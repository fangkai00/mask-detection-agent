# -*- coding: utf-8 -*-
"""提示词模板包(planner / finalizer / 工具提示词)。

子模块划分:
  - planner_prompt : LangGraph planner(路由)与 finalizer(终答)提示词
  - tools_prompts  : 工具提示词(WebSearch user prompt)
"""
from prompts.planner_prompt import (
    PLANNER_SYSTEM_PROMPT,
    FINALIZER_SYSTEM_PROMPT,
    format_planner_prompt,
    format_finalizer_prompt,
)
from prompts.tools_prompts import (
    WEB_SEARCH_USER_PROMPT,
    format_web_search_user_prompt,
)

__all__ = [
    "PLANNER_SYSTEM_PROMPT",
    "FINALIZER_SYSTEM_PROMPT",
    "format_planner_prompt",
    "format_finalizer_prompt",
    "WEB_SEARCH_USER_PROMPT",
    "format_web_search_user_prompt",
]
