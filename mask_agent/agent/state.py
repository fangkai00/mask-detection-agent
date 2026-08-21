# -*- coding: utf-8 -*-
"""
LangGraph 状态定义

记忆分层(本状态只承载"短期记忆"+ 本轮工具产出,长期记忆在 tools/memory_rag.py):
  - messages: 短期记忆载体。每轮 chat 由 cli._build_state 把当前会话最近 N 轮
              (config.SHORT_TERM_TURNS)直接拼进来作为 LLM context,精确但有限。
              超出窗口的早期对话靠长期记忆 RAG 召回注入。
  - user_query: 用户原始问题
  - image_path: 用户上传的待检测图片路径(可空)
  - detection_result: 口罩检测结果(dict,含 total/mask_count/no_mask_count 等)
  - search_results: 联网搜索结果(list[{url,title,snippet}])
  - rag_results: 文档知识库检索结果(list[{content,source,page,score}])。
                 注意:这是 tools/rag_knowledge.py 对 rag_data/ PDF/MD 的检索产出,
                 属"文档知识库"而非"长期对话记忆";长期对话记忆见 tools/memory_rag.py。
  - verified_answer: 最终回答
  - next_node: planner 路由决策(mask_detect / search / rag_search / finish)
  - steps: 已执行 planner 步数(防死循环)
"""
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """口罩检测智能体 LangGraph 状态。"""

    # 短期记忆:本会话最近 N 轮对话直接进 LLM context(用 add_messages 归约累加)
    # 长期记忆(早期对话)由 cli._build_state 调 memory_rag.recall 后注入 [长期记忆] 片段
    messages: Annotated[List[BaseMessage], add_messages]

    # 用户原始问题
    user_query: str

    # 用户上传的待检测图片路径(无图片时为空串)
    image_path: str

    # 口罩检测结果(mask_detect 节点产出)
    # {total_persons, mask_count, no_mask_count, compliance_rate, detections, annotated_image}
    detection_result: dict

    # 联网搜索结果(search 节点产出)
    search_results: List

    # 文档知识库检索结果(rag_search 节点产出,来自 tools/rag_knowledge.py 检索 PDF/MD;
    # 注意:这不是"长期对话记忆",长期对话记忆在 tools/memory_rag.py)
    # [{content, source, page, score}, ...]
    rag_results: List

    # 最终回答(finalizer 产出)
    verified_answer: str

    # planner 路由决策:mask_detect / search / rag_search / finish
    next_node: str

    # planner 路由理由(LLM 给出的决策原因,展示给用户的思考过程用)
    planner_reason: str

    # 已执行 planner 步数(防死循环)
    steps: int
