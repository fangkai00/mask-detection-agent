# -*- coding: utf-8 -*-
"""
LangGraph 编排层

状态图节点:planner / mask_detect / search / rag_search / finalizer
边:
  - planner → {mask_detect | search | rag_search | finish}(条件路由)
  - mask_detect / search / rag_search → planner(循环)
  - finish → finalizer → END

planner 用主 LLM(qwen-max,DashScope OpenAI 兼容)输出路由 JSON。
"""
import json
import logging
import os
import re
import sys
from typing import List, Literal

# 项目根入 sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import compat  # noqa: E402

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

import config
from agent.error_feedback import (
    format_error_feedback_message,
    get_max_tool_errors,
    safe_tool_call,
)
from agent.state import AgentState
from prompts.planner_prompt import format_finalizer_prompt, format_planner_prompt
from tools.mask_detection import get_mask_detection_tool
from tools.rag_knowledge import RAGKnowledgeTool
from tools.web_search import WebSearchTool

logger = logging.getLogger("mask_agent")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)
logger.propagate = False

# ============================================================
# 主决策 LLM(懒初始化,按 LLM_PROVIDER 切换 dashscope/ollama)
# ============================================================
_llm = None


def _resolve_llm_params() -> dict:
    """根据 config.LLM_PROVIDER 解析当前生效的 LLM 参数。

    返回 {model, base_url, api_key}。支持:
      - "dashscope":阿里云百炼(云端)
      - "ollama":本地 Ollama(OpenAI 兼容接口)
    其他值按 dashscope 兼容处理。
    """
    provider = (getattr(config, "LLM_PROVIDER", "dashscope") or "dashscope").lower()
    if provider == "ollama":
        model = getattr(config, "OLLAMA_MODEL", "") or "qwen2.5:7b"
        base_url = getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434/v1")
        api_key = getattr(config, "OLLAMA_API_KEY", "ollama") or "ollama"
        logger.info("[llm] provider=ollama, model=%s, base_url=%s", model, base_url)
        return {"model": model, "base_url": base_url, "api_key": api_key}
    # 默认 dashscope
    model = getattr(config, "MAIN_LLM_MODEL", "qwen-max")
    base_url = getattr(config, "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    api_key = getattr(config, "DASHSCOPE_API_KEY", "")
    logger.info("[llm] provider=dashscope, model=%s, base_url=%s", model, base_url)
    return {"model": model, "base_url": base_url, "api_key": api_key}


def get_main_llm() -> ChatOpenAI:
    """主决策 LLM,按 config.LLM_PROVIDER 在 DashScope / Ollama 间切换。"""
    global _llm
    if _llm is None:
        p = _resolve_llm_params()
        kwargs = dict(
            model=p["model"],
            base_url=p["base_url"],
            api_key=p["api_key"],
            temperature=config.MAIN_LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )
        # Ollama:限制 num_ctx 防止模型用过大 context(如 256K)导致 prefill 爆慢。
        # 通过 model_kwargs 传 Ollama options(OpenAI 兼容接口支持)。
        if (getattr(config, "LLM_PROVIDER", "dashscope") or "").lower() == "ollama":
            num_ctx = int(getattr(config, "OLLAMA_NUM_CTX", 4096))
            kwargs["model_kwargs"] = {"options": {"num_ctx": num_ctx}}
        _llm = ChatOpenAI(**kwargs)
    return _llm


# ============================================================
# 路由决策模型(Pydantic,用于生成 JSON Schema 强约束)
# ============================================================
class RouteDecision(BaseModel):
    next: Literal["mask_detect", "search", "rag_search", "finish"]
    reason: str = ""


def _build_route_decision_response_format() -> dict:
    """基于 RouteDecision Pydantic 模型构造 DashScope/OpenAI 兼容的
    response_format(json_schema strict)。

    strict 模式要求:
      - additionalProperties=false
      - 所有字段在 required 中(含带默认值的字段,否则服务端可能拒绝)
    """
    schema = RouteDecision.model_json_schema()
    # 清理 $ 开头的 JSON Schema 元字段(strict 模式不接受 $defs/$ref)
    schema.pop("$defs", None)
    schema.pop("title", None)
    schema["additionalProperties"] = False
    # next 用 enum 描述;reason 强制 required(strict 不允许 optional)
    props = schema.get("properties", {})
    if "next" in props:
        # Pydantic Literal 已生成 enum,保留即可;确保 type
        props["next"].setdefault("type", "string")
    if "reason" in props:
        props["reason"].setdefault("type", "string")
        # 去掉 default(strict 模式下 default 字段不算 required 会报错,统一 required)
        props["reason"].pop("default", None)
    schema["required"] = ["next", "reason"]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "route_decision",
            "schema": schema,
            "strict": True,
        },
    }


# 绑定了 response_format 的 planner LLM(懒初始化)
_planner_llm = None


def get_planner_llm():
    """返回绑定了 response_format json_schema 的 planner LLM。

    容错:Ollama 部分模型不支持 response_format json_schema strict,
    bind 不报错但运行时可能 400;此处 bind 阶段不抛错(LangChain 是延迟绑定),
    真正失败由 planner 节点的 try/except 兜底退回 finish。
    另:对 Ollama 老模型,可手动把 LLM_PROVIDER 改回 dashscope 或换支持 json_schema 的模型。
    """
    global _planner_llm
    if _planner_llm is None:
        try:
            _planner_llm = get_main_llm().bind(
                response_format=_build_route_decision_response_format()
            )
        except Exception as e:
            # bind 阶段失败(极罕见),退回普通 LLM,靠正则兜底解析
            logger.warning("[planner] bind response_format 失败,退回普通 LLM: %s", e)
            _planner_llm = get_main_llm()
    return _planner_llm


def _parse_route(content: str) -> RouteDecision:
    """从 LLM 文本中解析路由 JSON,失败回退 finish。

    response_format 已强约束输出为合法 JSON,但仍保留正则提取 + 容错,
    以应对服务端偶发不遵守 schema 的情况。
    """
    _VALID = ("mask_detect", "search", "rag_search", "finish")
    if content:
        # 优先直接解析(response_format 下通常整段就是合法 JSON)
        try:
            d = json.loads(content)
            nxt = d.get("next", "finish")
            if nxt not in _VALID:
                nxt = "finish"
            return RouteDecision(next=nxt, reason=str(d.get("reason", "")))
        except Exception:
            pass
        # 兜底:正则提取首个 JSON 对象
        m = re.search(r"\{[\s\S]*\}", content)
        if m:
            try:
                d = json.loads(m.group(0))
                nxt = d.get("next", "finish")
                if nxt not in _VALID:
                    nxt = "finish"
                return RouteDecision(next=nxt, reason=str(d.get("reason", "")))
            except Exception:
                pass
    return RouteDecision(next="finish")


def _summarize_gathered(state: AgentState) -> str:
    """汇总已收集的检测结果和搜索信息,供 planner 与 finalizer 使用。"""
    parts = []

    # 短期记忆:最近 4 轮对话历史
    for m in state.get("messages") or []:
        content = m.content if isinstance(m.content, str) else str(m.content)
        if "[历史对话" in content:
            parts.append(content[:300])

    # 检测结果
    det = state.get("detection_result")
    if det:
        parts.append(
            f"[MaskDetect] 检测完成:共{det.get('total_persons',0)}人,"
            f"戴口罩{det.get('mask_count',0)}人,"
            f"未戴{det.get('no_mask_count',0)}人,"
            f"合规率{det.get('compliance_rate',0)*100:.1f}%"
        )
        # 逐人详情(截断)
        detections = det.get("detections", [])
        if detections:
            det_lines = [
                f"  - {d['cls_name']} (conf={d['confidence']:.2f})"
                for d in detections[:10]
            ]
            parts.append("检测结果明细:\n" + "\n".join(det_lines))
        if det.get("annotated_image"):
            parts.append(f"标注图: {det['annotated_image']}")

    # 搜索结果
    sr = state.get("search_results") or []
    if sr:
        parts.append(
            "[Search]\n"
            + "\n".join(f"{r.get('title', '')}: {r.get('snippet', '')}" for r in sr)
        )

    # RAG 本地知识库检索结果
    rr = state.get("rag_results") or []
    if rr:
        rag_lines = []
        for i, r in enumerate(rr, 1):
            content = (r.get("content") or "").strip()
            source = r.get("source") or "(未知来源)"
            page = r.get("page") or ""
            score = r.get("score", 0)
            rag_lines.append(
                f"[{i}] 来源: {source} 第{page}页 (score={score:.3f})\n{content[:300]}"
            )
        parts.append("[RAG 知识库]\n" + "\n".join(rag_lines))

    # 错误反馈
    err_parts = []
    for m in state.get("messages") or []:
        content = m.content if isinstance(m.content, str) else str(m.content)
        if "[ToolError]" in content:
            idx = content.find("[ToolError]")
            err_parts.append(content[idx: idx + 400])
    if err_parts:
        parts.append("[错误反馈]\n" + "\n---\n".join(err_parts))

    return "\n\n".join(parts) if parts else "(暂无)"


def _count_tool_errors(state: AgentState, tool_name: str) -> int:
    """统计 messages 中 [ToolError] 来自 tool_name 的次数。"""
    cnt = 0
    for m in state.get("messages") or []:
        content = m.content if isinstance(m.content, str) else str(m.content)
        if "[ToolError]" in content and f"工具 {tool_name}" in content:
            cnt += 1
    return cnt


def _enforce_no_repeat(decision: RouteDecision, state: AgentState) -> RouteDecision:
    """代码级防死循环:LLM 若重复调用已产出结果的工具,强制推进或收尾。"""
    ran = set()
    if state.get("detection_result"):
        ran.add("mask_detect")
    if state.get("search_results"):
        ran.add("search")
    if state.get("rag_results"):
        ran.add("rag_search")
    if decision.next not in ran:
        return decision

    # 错误反馈例外:若该工具上次返回的是错误(且未达上限),允许重试
    tool_err_cnt = _count_tool_errors(state, decision.next)
    last_msg = ""
    for m in reversed(state.get("messages") or []):
        last_msg = m.content if isinstance(m.content, str) else str(m.content)
        break
    last_is_error = "[ToolError]" in last_msg and f"工具 {decision.next}" in last_msg
    if last_is_error and tool_err_cnt < get_max_tool_errors():
        logger.info(
            "[enforce_no_repeat] %s 上次失败(第 %s/%s 次),允许 LLM 修正重试",
            decision.next, tool_err_cnt, get_max_tool_errors(),
        )
        return decision
    if tool_err_cnt >= get_max_tool_errors():
        logger.warning(
            "[enforce_no_repeat] %s 失败 %s 次超限,强制改 finish",
            decision.next, tool_err_cnt,
        )
        return RouteDecision(next="finish", reason=f"工具 {decision.next} 失败超限,放弃")

    return RouteDecision(next="finish", reason="代码级收尾:避免重复工具导致死循环")


# ============================================================
# 节点函数
# ============================================================
def planner(state: AgentState) -> dict:
    """主决策节点:决定下一步路由。"""
    steps = (state.get("steps") or 0) + 1
    if steps > config.MAX_PLANNER_STEPS:
        logger.info("[planner] 达到步数上限 %s → finish", config.MAX_PLANNER_STEPS)
        return {"next_node": "finish", "steps": steps}

    gathered = _summarize_gathered(state)
    image_path = state.get("image_path", "")

    sys = format_planner_prompt(
        steps=steps,
        max_steps=config.MAX_PLANNER_STEPS,
        gathered=gathered,
        image_path=image_path,
        max_tool_errors=get_max_tool_errors(),
    )
    msgs = [SystemMessage(content=sys), HumanMessage(content=state["user_query"])]
    try:
        # 用绑定了 response_format json_schema 的 planner LLM,
        # 输出被强约束为 {"next": ..., "reason": ...}
        resp = get_planner_llm().invoke(msgs)
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        decision = _parse_route(content)
    except Exception as e:
        logger.error("[planner] LLM 调用失败,回退 finish: %s", e)
        decision = RouteDecision(next="finish")

    decision = _enforce_no_repeat(decision, state)

    logger.info("[planner] step=%s → next=%s (%s)", steps, decision.next, decision.reason)
    return {"next_node": decision.next, "steps": steps, "planner_reason": decision.reason}


def mask_detect_node(state: AgentState) -> dict:
    """口罩检测节点:调用 YOLOv8 模型对图片执行检测。

    错误反馈闭环:用 safe_tool_call 包装,失败时不崩溃,把错误描述
    回传 LLM,让 planner 下一轮自我修正。
    """
    image_path = state.get("image_path", "")
    if not image_path:
        logger.warning("[mask_detect] 未提供图片路径,跳过检测")
        return {
            "messages": [AIMessage(content="[MaskDetect] 未提供图片路径,无法执行口罩检测。")],
        }

    args = {"image_path": image_path}
    result = safe_tool_call(get_mask_detection_tool(), args, tool_name="mask_detect")

    if result.success:
        det = result.result or {}
        total = det.get("total_persons", 0)
        mask_n = det.get("mask_count", 0)
        nomask_n = det.get("no_mask_count", 0)
        rate = det.get("compliance_rate", 0)
        snippet = (
            f"[MaskDetect] 检测完成:共{total}人,"
            f"戴口罩{mask_n}人,未戴{nomask_n}人,合规率{rate*100:.1f}%"
        )
        logger.info("[mask_detect] 检测成功(attempts=%s)", result.attempts)
        return {
            "detection_result": det,
            "messages": [AIMessage(content=snippet)],
        }

    # 失败:回传错误描述给 LLM
    err_attempt = _count_tool_errors(state, "mask_detect") + 1
    err_msg = format_error_feedback_message(
        tool_name="mask_detect",
        error_message=result.error_message,
        attempt=err_attempt,
    )
    logger.warning("[mask_detect] 检测失败,回传 LLM 反思: %s", result.error_type)
    return {
        "detection_result": {},
        "messages": [AIMessage(content=err_msg)],
    }


def search_node(state: AgentState) -> dict:
    """搜索节点:DashScope Qwen enable_search 联网搜索。"""
    args = {"query": state["user_query"]}
    result = safe_tool_call(WebSearchTool(), args, tool_name="web_search")

    if result.success:
        results = result.result or []
        snippet = (
            "\n".join(f"- {r.get('title', '')} {r.get('url', '')}" for r in results)
            if results else "(搜索失败或无结果)"
        )
        logger.info("[search] 返回 %s 条结果(attempts=%s)", len(results), result.attempts)
        return {
            "search_results": results,
            "messages": [AIMessage(content=f"[Search]\n{snippet}")],
        }

    err_attempt = _count_tool_errors(state, "web_search") + 1
    err_msg = format_error_feedback_message(
        tool_name="web_search",
        error_message=result.error_message,
        attempt=err_attempt,
    )
    logger.warning("[search] 失败,回传 LLM 反思: %s", result.error_type)
    return {
        "search_results": [],
        "messages": [AIMessage(content=err_msg)],
    }


def rag_search_node(state: AgentState) -> dict:
    """本地知识库检索节点:基于 LlamaIndex + DashScope Embedding 检索 rag_data 中的文档。

    首次调用时自动构建向量索引(可能耗时 30s+),后续复用持久化索引。
    错误反馈闭环同 search_node。
    """
    args = {"query": state["user_query"]}
    result = safe_tool_call(RAGKnowledgeTool(), args, tool_name="rag_search")

    if result.success:
        results = result.result or []
        if results and not any(r.get("error") for r in results):
            snippet = (
                "\n".join(
                    f"- 来源: {r.get('source','')} 第{r.get('page','?')}页 "
                    f"(score={r.get('score',0):.3f})"
                    for r in results
                )
                if results else "(知识库无结果)"
            )
            logger.info("[rag_search] 返回 %s 条片段(attempts=%s)", len(results), result.attempts)
            return {
                "rag_results": results,
                "messages": [AIMessage(content=f"[RAG]\n{snippet}")],
            }
        # 检索结果含 error 字段(如索引未构建),按失败处理
        err_msg_in_result = next((r.get("error") for r in results if r.get("error")), "")
        if err_msg_in_result:
            err_attempt = _count_tool_errors(state, "rag_search") + 1
            err_msg = format_error_feedback_message(
                tool_name="rag_search",
                error_message=err_msg_in_result,
                attempt=err_attempt,
            )
            logger.warning("[rag_search] 检索返回错误,回传 LLM 反思")
            return {
                "rag_results": [],
                "messages": [AIMessage(content=err_msg)],
            }

    err_attempt = _count_tool_errors(state, "rag_search") + 1
    err_msg = format_error_feedback_message(
        tool_name="rag_search",
        error_message=result.error_message,
        attempt=err_attempt,
    )
    logger.warning("[rag_search] 失败,回传 LLM 反思: %s", result.error_type)
    return {
        "rag_results": [],
        "messages": [AIMessage(content=err_msg)],
    }


def finalizer(state: AgentState) -> dict:
    """终答节点:基于已收集信息生成回答。"""
    gathered = _summarize_gathered(state)
    sys = format_finalizer_prompt(
        gathered=gathered,
        disclaimer_triggers=config.DISCLAIMER_TRIGGERS,
    )
    msgs = [
        SystemMessage(content=sys),
        HumanMessage(content=state["user_query"]),
    ]
    try:
        resp = get_main_llm().invoke(msgs)
        answer = resp.content if isinstance(resp.content, str) else str(resp.content)
    except Exception as e:
        logger.error("[finalizer] LLM 失败,回退已收集信息: %s", e)
        answer = gathered if gathered != "(暂无)" else "(无可用信息)"

    # 检测到未戴口罩时自动追加提示
    _DISCLAIMER_CORE = "AI 模型自动生成"
    if (
        any(k in answer for k in config.DISCLAIMER_TRIGGERS)
        and _DISCLAIMER_CORE not in answer
    ):
        answer = answer.rstrip() + "\n" + config.DISCLAIMER_TEXT
    logger.info("[finalizer] 生成最终回答(len=%s)", len(answer))
    return {"verified_answer": answer}


# ============================================================
# 构建图
# ============================================================
def build_graph():
    """构建并编译口罩检测 LangGraph。"""
    g = StateGraph(AgentState)
    g.add_node("planner", planner)
    g.add_node("mask_detect", mask_detect_node)
    g.add_node("search", search_node)
    g.add_node("rag_search", rag_search_node)
    g.add_node("finalizer", finalizer)

    g.set_entry_point("planner")
    g.add_conditional_edges(
        "planner",
        lambda s: s.get("next_node") or "finish",
        {
            "mask_detect": "mask_detect",
            "search": "search",
            "rag_search": "rag_search",
            "finish": "finalizer",
        },
    )
    g.add_edge("mask_detect", "planner")
    g.add_edge("search", "planner")
    g.add_edge("rag_search", "planner")
    g.add_edge("finalizer", END)
    return g.compile()


def build_initial_state(query: str, image_path: str = "") -> dict:
    """构建图的初始状态。"""
    return {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "image_path": image_path,
        "detection_result": {},
        "search_results": [],
        "rag_results": [],
        "verified_answer": "",
        "next_node": "",
        "planner_reason": "",
        "steps": 0,
    }


def run(query: str, image_path: str = "") -> dict:
    """运行一次完整问答(同步)。"""
    return build_graph().invoke(build_initial_state(query, image_path))


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "口罩怎么戴才正确?"
    img = sys.argv[2] if len(sys.argv) > 2 else ""
    print(f"\n用户问题: {q}")
    if img:
        print(f"图片路径: {img}")
    print("=" * 60)
    result = run(q, img)
    print("=" * 60)
    print("路由步数:", result.get("steps"))
    if result.get("detection_result"):
        det = result["detection_result"]
        print(f"检测结果: {det.get('mask_count',0)}戴 / {det.get('no_mask_count',0)}未戴 / 共{det.get('total_persons',0)}人")
    print("\n=== 最终回答 ===\n")
    print(result.get("verified_answer", "(无)"))
