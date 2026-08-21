# -*- coding: utf-8 -*-
"""
对话会话管理层 — GUI/TUI 共用入口

封装 LangGraph 图的调用,提供多轮对话、流式输出、会话历史管理。
支持图片输入(口罩检测场景)。
"""
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable, Generator, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

import config
from agent.graph import build_graph
from agent.state import AgentState

# 节点名 → 人类可读描述
_NODE_LABELS = {
    "planner": ("\U0001F914", "正在分析问题,规划执行路径..."),
    "mask_detect": ("\U0001F637", "正在执行口罩检测..."),
    "search": ("\U0001F50D", "正在联网搜索相关信息..."),
    "rag_search": ("\U0001F4DA", "正在检索本地知识库..."),
    "finalizer": ("\u270D\uFE0F", "正在生成最终回答..."),
}


def _extract_node_summary(node_name: str, node_output: dict) -> Optional[str]:
    """从节点输出中提取简要结果描述。"""
    if node_name == "planner":
        nxt = node_output.get("next_node", "")
        steps = node_output.get("steps", 0)
        label_map = {
            "mask_detect": "口罩检测",
            "search": "联网搜索",
            "rag_search": "本地知识库",
            "finish": "收尾",
        }
        target = label_map.get(nxt, nxt)
        return f"→ 决定走【{target}】(第 {steps} 步)"

    elif node_name == "mask_detect":
        det = node_output.get("detection_result", {})
        if det:
            return (f"检测到 {det.get('total_persons',0)} 人"
                    f"(戴口罩 {det.get('mask_count',0)}, 未戴 {det.get('no_mask_count',0)})")
        return "检测未返回结果"

    elif node_name == "search":
        results = node_output.get("search_results", [])
        n = len(results) if results else 0
        return f"找到 {n} 条搜索结果"

    elif node_name == "rag_search":
        results = node_output.get("rag_results", [])
        n = len(results) if results else 0
        return f"检索到 {n} 个知识片段"

    elif node_name == "finalizer":
        answer = node_output.get("verified_answer", "")
        return f"生成回答 {len(answer)} 字" if answer else "回答生成完成"

    return None


def _extract_node_fields(node_name: str, node_output: dict, state: dict) -> dict:
    """从节点输出中提取结构化字段,供前端以 JSON 代码块形式展示思考过程。

    暴露内容(按用户选择):
      - planner 路由理由(reason)
      - 工具输入参数(image_path / query 等)
      + 各节点输出摘要(几个关键字段,非完整逐人列表/全部搜索结果)
    """
    state = state or {}
    user_query = state.get("user_query", "")
    image_path = state.get("image_path", "")

    if node_name == "planner":
        nxt = node_output.get("next_node", "")
        label_map = {
            "mask_detect": "口罩检测",
            "search": "联网搜索",
            "rag_search": "本地知识库",
            "finish": "收尾",
        }
        return {
            "step": node_output.get("steps", 0),
            "next": nxt,
            "next_label": label_map.get(nxt, nxt),
            "reason": node_output.get("planner_reason", ""),
            "user_query": user_query[:200],
            "has_image": bool(image_path),
            "image_path": image_path or None,
        }

    elif node_name == "mask_detect":
        det = node_output.get("detection_result", {}) or {}
        if not det:
            return {
                "input_image": image_path or None,
                "status": "未返回结果",
            }
        return {
            "input_image": image_path or None,
            "total_persons": det.get("total_persons", 0),
            "mask_count": det.get("mask_count", 0),
            "no_mask_count": det.get("no_mask_count", 0),
            "compliance_rate": round(det.get("compliance_rate", 0) * 100, 1),
            "annotated_image": det.get("annotated_image", "") or None,
        }

    elif node_name == "search":
        results = node_output.get("search_results", []) or []
        top3 = [
            {"title": r.get("title", ""), "url": r.get("url", "")}
            for r in results[:3]
        ]
        return {
            "query": user_query[:200],
            "results_count": len(results),
            "top_3": top3,
        }

    elif node_name == "rag_search":
        results = node_output.get("rag_results", []) or []
        top3 = [
            {
                "source": r.get("source", ""),
                "page": r.get("page", ""),
                "score": round(r.get("score", 0), 3),
                "content_preview": (r.get("content", "") or "")[:120],
            }
            for r in results[:3]
        ]
        return {
            "query": user_query[:200],
            "results_count": len(results),
            "top_3": top3,
        }

    elif node_name == "finalizer":
        answer = node_output.get("verified_answer", "")
        sources = []
        if state.get("detection_result"):
            sources.append("口罩检测")
        if state.get("search_results"):
            sources.append("联网搜索")
        if state.get("rag_results"):
            sources.append("本地知识库")
        return {
            "answer_length": len(answer),
            "sources": sources,
        }

    return {}


logger = logging.getLogger("mask_agent.cli")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)
logger.propagate = False


@dataclass
class ConversationTurn:
    """单轮对话记录。"""
    user_input: str
    assistant_output: str
    steps: int
    route_log: List[str] = field(default_factory=list)
    image_path: str = ""
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        """序列化为可 JSON 持久化的 dict(只保留文本+元数据,不含 fields/完整检测结果)。"""
        return {
            "user_input": self.user_input,
            "assistant_output": self.assistant_output,
            "steps": self.steps,
            "route_log": list(self.route_log),
            "image_path": self.image_path,
            "elapsed_s": self.elapsed_s,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConversationTurn":
        """从 dict 反序列化(兼容旧数据缺字段的情况)。"""
        return cls(
            user_input=str(d.get("user_input", "")),
            assistant_output=str(d.get("assistant_output", "")),
            steps=int(d.get("steps", 0)),
            route_log=list(d.get("route_log", []) or []),
            image_path=str(d.get("image_path", "")),
            elapsed_s=float(d.get("elapsed_s", 0.0)),
        )


@dataclass
class ConversationSession:
    """多轮对话会话。"""
    session_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    _graph: object = None
    _last_turn: object = None

    def __post_init__(self):
        if self._graph is None:
            self._graph = build_graph()

    @property
    def history(self) -> List[dict]:
        result = []
        for t in self.turns:
            result.append({"role": "user", "content": t.user_input})
            result.append({"role": "assistant", "content": t.assistant_output})
        return result

    def chat(self, user_input: str, image_path: str = "") -> ConversationTurn:
        """同步调用:发送一条消息,返回完整回答。"""
        start = time.time()
        route_log = []

        state = self._build_state(user_input, image_path)
        result_state = self._graph.invoke(state)

        answer = result_state.get("verified_answer", "(无回答)")
        steps = result_state.get("steps", 0)

        for msg in result_state.get("messages", []):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if "[MaskDetect]" in content:
                route_log.append("\U0001F637 口罩检测")
            elif "[Search]" in content:
                route_log.append("\U0001F50D 联网搜索")
            elif "[RAG]" in content:
                route_log.append("\U0001F4DA 本地知识库")

        elapsed = time.time() - start
        turn = ConversationTurn(
            user_input=user_input,
            assistant_output=answer,
            steps=steps,
            route_log=route_log,
            image_path=image_path,
            elapsed_s=round(elapsed, 2),
        )
        self.turns.append(turn)
        # 同步写入长期记忆向量库(RAG 召回源);失败不阻塞对话
        try:
            from tools import memory_rag
            memory_rag.add_turn(self.session_id, turn.to_dict(), len(self.turns))
        except Exception as _e:
            logger.warning("写入长期记忆失败: %s", _e)
        return turn

    def chat_with_progress(
        self,
        user_input: str,
        image_path: str = "",
        on_step: Optional[Callable[[str, str, Optional[str]], None]] = None,
    ) -> Generator[dict, None, ConversationTurn]:
        """带进度的对话:逐节点产出进度事件 + finalizer 真流式 token。

        用 LangGraph 的 stream_mode=["updates","messages"]:
          - updates:节点完成事件(进度展示,同旧逻辑)
          - messages:LLM token 流(finalizer 生成回答时逐 token 推送)
        finalizer 节点内部仍是 invoke,LangGraph 会自动 hook 其 LLM 调用
        并通过 messages 流模式实时产出 token chunk。
        """
        start = time.time()
        route_log = []
        state = self._build_state(user_input, image_path)

        final_node_outputs = {}
        finalizer_chunks = []  # 累积 finalizer 的流式 token
        finalizer_streaming_started = False
        # finalizer 是否已通过流式输出完整 answer(用于决定是否还需要兜底)
        finalizer_answer_complete = False

        try:
            for event in self._graph.stream(
                state, stream_mode=["updates", "messages"]
            ):
                # stream_mode list 模式:event 是 (mode_name, data)
                mode, data = event

                if mode == "updates":
                    for node_name, node_output in (data or {}).items():
                        if node_name == "__end__":
                            continue

                        icon, label = _NODE_LABELS.get(
                            node_name, ("\u2699\uFE0F", f"正在执行 {node_name}...")
                        )
                        yield {
                            "type": "step_start",
                            "node": node_name,
                            "icon": icon,
                            "label": label,
                        }
                        summary = _extract_node_summary(node_name, node_output)
                        fields = _extract_node_fields(node_name, node_output, state)

                        if node_name == "mask_detect":
                            route_log.append("\U0001F637 口罩检测")
                        elif node_name == "search":
                            route_log.append("\U0001F50D 联网搜索")
                        elif node_name == "rag_search":
                            route_log.append("\U0001F4DA 本地知识库")

                        final_node_outputs[node_name] = node_output

                        yield {
                            "type": "step_done",
                            "node": node_name,
                            "summary": summary or "",
                            "fields": fields,
                        }
                        if on_step:
                            on_step(node_name, label, summary)

                elif mode == "messages":
                    # data 是 (message_chunk, metadata) 元组
                    try:
                        chunk, metadata = data
                    except (TypeError, ValueError):
                        continue
                    node = ""
                    if isinstance(metadata, dict):
                        node = metadata.get("langgraph_node", "") or metadata.get("name", "")
                    # 只推送 finalizer 节点的 LLM token(planner 的 JSON 不推给前端)
                    if node != "finalizer":
                        continue
                    content = getattr(chunk, "content", "")
                    if not content:
                        continue
                    if not finalizer_streaming_started:
                        finalizer_streaming_started = True
                        yield {"type": "answer_start"}
                    finalizer_chunks.append(content)
                    yield {"type": "answer_chunk", "text": content}

        except Exception as e:
            yield {"type": "error", "message": str(e)}
            logger.error("chat_with_progress 执行出错: %s", e)

        # 组装完整 answer:优先用流式累积的 token,其次从 finalizer 节点输出取
        answer = ""
        steps = 0

        if finalizer_chunks:
            answer = "".join(finalizer_chunks)
            finalizer_answer_complete = True

        if not answer and "finalizer" in final_node_outputs:
            answer = final_node_outputs["finalizer"].get("verified_answer", "")

        if "planner" in final_node_outputs:
            steps = final_node_outputs["planner"].get("steps", 0)

        if not answer:
            for node_name, node_out in reversed(final_node_outputs.items()):
                msgs = node_out.get("messages", [])
                for m in reversed(msgs):
                    content = m.content if isinstance(m.content, str) else str(m.content)
                    if content and content not in ("", "(无)"):
                        answer = content
                        break
                if answer:
                    break

        if not answer:
            answer = "(生成回答时出现异常,请重试)"

        # 若已通过流式推过 token,则只补推 answer_done(避免重复推全文)
        if finalizer_answer_complete:
            yield {"type": "answer_done", "answer": answer}
        else:
            # 流式未触发(如 finalizer 异常走了兜底),退化为切片假流式
            yield {"type": "answer_start"}
            for i in range(1, len(answer) + 1):
                yield {"type": "answer_chunk", "text": answer[:i]}
            yield {"type": "answer_done", "answer": answer}

        elapsed = time.time() - start
        turn = ConversationTurn(
            user_input=user_input,
            assistant_output=answer,
            steps=steps,
            route_log=route_log,
            image_path=image_path,
            elapsed_s=round(elapsed, 2),
        )
        self.turns.append(turn)
        # 同步写入长期记忆向量库(RAG 召回源);失败不阻塞对话
        try:
            from tools import memory_rag
            memory_rag.add_turn(self.session_id, turn.to_dict(), len(self.turns))
        except Exception as _e:
            logger.warning("写入长期记忆失败: %s", _e)
        self._last_turn = turn
        return turn

    def chat_stream(self, user_input: str, image_path: str = "") -> Generator[str, None, None]:
        """流式调用:逐 token 产出回答。"""
        start = time.time()
        route_log = []

        state = self._build_state(user_input, image_path)
        result_state = self._graph.invoke(state)

        answer = result_state.get("verified_answer", "(无回答)")
        steps = result_state.get("steps", 0)

        for msg in result_state.get("messages", []):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if "[MaskDetect]" in content:
                route_log.append("\U0001F637 口罩检测")
            elif "[Search]" in content:
                route_log.append("\U0001F50D 联网搜索")
            elif "[RAG]" in content:
                route_log.append("\U0001F4DA 本地知识库")

        for i in range(1, len(answer) + 1):
            yield answer[:i]

        elapsed = time.time() - start
        turn = ConversationTurn(
            user_input=user_input,
            assistant_output=answer,
            steps=steps,
            route_log=route_log,
            image_path=image_path,
            elapsed_s=round(elapsed, 2),
        )
        self.turns.append(turn)
        # 同步写入长期记忆向量库(RAG 召回源);失败不阻塞对话
        try:
            from tools import memory_rag
            memory_rag.add_turn(self.session_id, turn.to_dict(), len(self.turns))
        except Exception as _e:
            logger.warning("写入长期记忆失败: %s", _e)
        self._last_turn = turn

    async def achat_stream(self, user_input: str, image_path: str = "") -> AsyncGenerator[str, None]:
        """异步流式调用:供 Streamlit st.write_stream 使用。"""
        gen = self.chat_stream(user_input, image_path)
        for chunk in gen:
            yield chunk
        if self.turns:
            self._last_turn = self.turns[-1]
        else:
            self._last_turn = ConversationTurn(user_input=user_input, assistant_output="", steps=0)

    def _build_state(self, user_input: str, image_path: str = "") -> dict:
        """构建图的初始状态:短期记忆(本会话近邻)+ 长期记忆(RAG 召回)。

        记忆分层:
        - 短期记忆:取本会话最近 N 轮(config.SHORT_TERM_TURNS,默认 8)直接拼到
          messages,作为 LLM context。精确,但受窗口限制。
        - 长期记忆:用当前 user_query 去 tools/memory_rag.recall 检索向量库,
          召回"与当前问题相关的早期/跨会话历史片段",以 [长期记忆] 前缀注入。
          过滤掉当前会话自身所有 turn(它们已在 self.turns / 短期窗口中,避免重复)。
        """
        messages = []

        # ---- 短期记忆:本会话最近 N 轮直接进 LLM context ----
        short_window = int(getattr(config, "SHORT_TERM_TURNS", 8))
        recent_turns = self.turns[-short_window:] if self.turns else []
        total_turns = len(self.turns)
        start_idx = max(0, total_turns - short_window)
        for offset, turn in enumerate(recent_turns):
            global_idx = start_idx + offset + 1  # 全局轮号(从1开始,便于长期记忆引用对齐)
            truncated_q = turn.user_input[:200]
            truncated_a = turn.assistant_output[:500]
            suffix_a = "..." if len(turn.assistant_output) > 500 else ""
            messages.append(HumanMessage(
                content=f"[历史对话第{global_idx}轮] 用户: {truncated_q}"
            ))
            messages.append(AIMessage(
                content=f"[历史对话第{global_idx}轮] 助手: {truncated_a}{suffix_a}"
            ))

        # ---- 长期记忆:RAG 召回相关早期对话(跨会话/超窗口) ----
        # 排除本会话所有 turn(它们已在 self.turns / 短期窗口覆盖,避免重复召回)
        try:
            from tools import memory_rag
            recalled = memory_rag.recall(
                user_input,
                top_k=getattr(config, "MEMORY_RECALL_TOP_K", 3),
                exclude_sid=self.session_id,
                exclude_recent=total_turns,  # >0 即触发过滤本会话所有轮
            )
            if recalled:
                memory_lines = []
                for r in recalled:
                    r_sid = r.get("sid", "?")
                    r_turn = r.get("turn", "?")
                    r_content = (r.get("content") or "").strip().replace("\n", " ")[:280]
                    memory_lines.append(f"[会话{r_sid}第{r_turn}轮] {r_content}")
                messages.append(SystemMessage(
                    content="[长期记忆-相关历史召回]\n" + "\n".join(memory_lines)
                ))
        except Exception as e:
            # 长期记忆召回失败不应阻塞对话主流程
            logger.warning("长期记忆召回失败: %s", e)

        messages.append(HumanMessage(content=user_input))

        return {
            "messages": messages,
            "user_query": user_input,
            "image_path": image_path,
            "detection_result": {},
            "search_results": [],
            "rag_results": [],
            "verified_answer": "",
            "next_node": "",
            "planner_reason": "",
            "steps": 0,
        }

    def reset(self):
        """重置会话。"""
        self.turns = []

    def save_to_disk(self) -> str:
        """把当前会话(含所有 turn)持久化到磁盘 JSON 文件。

        保留对话文本 + 元数据(路由、耗时、步数、图片路径),
        不含 fields 结构化思考过程和完整检测结果(体积控制)。
        在每轮对话完成后调用。
        """
        # 延迟导入避免循环依赖
        from agent.session_store import save_session
        turns_data = [t.to_dict() for t in self.turns]
        return save_session(self.session_id, turns_data)

    @classmethod
    def load_from_disk(cls, sid: str) -> Optional["ConversationSession"]:
        """从磁盘加载会话,返回 ConversationSession 实例;不存在返回 None。"""
        from agent.session_store import load_session
        data = load_session(sid)
        if data is None:
            return None
        sess = cls(session_id=sid)
        sess.turns = [ConversationTurn.from_dict(t) for t in (data.get("turns") or [])]
        return sess

    def export_history(self) -> str:
        """导出对话历史为 JSON 字符串。"""
        import json
        return json.dumps(
            [
                {
                    "turn": i + 1,
                    "user": t.user_input,
                    "assistant": t.assistant_output,
                    "image": t.image_path,
                    "steps": t.steps,
                    "route": t.route_log,
                    "elapsed_s": t.elapsed_s,
                }
                for i, t in enumerate(self.turns)
            ],
            ensure_ascii=False,
            indent=2,
        )
