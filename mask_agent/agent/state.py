# -*- coding: utf-8 -*-
"""
LangGraph 状态定义

State 包含:
  - messages: 消息历史(含用户问题、各节点产出摘要)
  - user_query: 用户原始问题
  - image_path: 用户上传的待检测图片路径(可空)
  - detection_result: 口罩检测结果(dict,含 total/mask_count/no_mask_count 等)
  - search_results: 联网搜索结果(list[{url,title,snippet}])
  - rag_results: 本地知识库检索结果(list[{content,source,page,score}])
  - verified_answer: 最终回答
  - next_node: planner 路由决策(mask_detect / search / rag_search / finish)
  - steps: 已执行 planner 步数(防死循环)
"""
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """口罩检测智能体 LangGraph 状态。"""

    # 消息历史(用 add_messages 归约累加)
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

    # 本地知识库检索结果(rag_search 节点产出)
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
