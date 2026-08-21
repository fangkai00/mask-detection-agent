# -*- coding: utf-8 -*-
"""
工具提示词(联网搜索 user prompt)

口罩检测工具无需 LLM 提示词(YOLOv8 模型直接推理),仅 WebSearch 需要
格式化 user prompt 让 Qwen 返回结构化 JSON。
"""

WEB_SEARCH_USER_PROMPT = """\
请针对以下问题联网搜索,返回一个 JSON 对象(最多 {num_results} 条结果),格式为 {{"results": [...]}},每条结果包含 url、title、snippet 三个字段。
不要输出任何其他文字,只输出符合 schema 的 JSON 对象。

问题: {query}

输出格式示例:
{{"results": [{{"url": "https://...", "title": "...", "snippet": "..."}}]}}
"""


def format_web_search_user_prompt(query: str, num_results: int = 5) -> str:
    """格式化联网搜索 user prompt。"""
    return WEB_SEARCH_USER_PROMPT.format(query=query, num_results=num_results)
