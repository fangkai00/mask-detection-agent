# -*- coding: utf-8 -*-
"""兼容层:修补环境里 langchain 与 langchain-core / langgraph-sdk 的版本错配。

问题1:langchain-core 0.3.x 的回调管理器会读取 `langchain.debug`，但某些版本的
langchain 不再暴露该属性，导致 BaseTool.invoke / LLM.invoke 报
`AttributeError: module 'langchain' has no attribute 'debug'`。

问题2:langgraph-sdk 期望 langchain-core 1.0+ 的
`langchain_core.language_models.chat_model_stream.AsyncChatModelStream`，
但当前环境 langchain-core 为 0.3.x，无此模块，导致 `from langgraph.graph import ...`
在导入链中触发 `ModuleNotFoundError`。本地 StateGraph 编译/调用不依赖该流式客户端类，
因此注入一个可接受任意参数的桩类即可让导入通过。

用法:在各业务模块顶部(sys.path 就绪后)`import compat` 即可，无需显式调用。
"""
import sys
import types

import langchain as _langchain

# ---- 问题1:补 langchain.debug / verbose 属性 ----
if not hasattr(_langchain, "debug"):
    _langchain.debug = False
if not hasattr(_langchain, "verbose"):
    _langchain.verbose = False
if not hasattr(_langchain, "llm_cache"):
    _langchain.llm_cache = None

# ---- 问题2:为 langchain_core.language_models.chat_model_stream 注入桩模块 ----
_shim_name = "langchain_core.language_models.chat_model_stream"
if _shim_name not in sys.modules:
    _shim = types.ModuleType(_shim_name)

    class _StreamStub:
        """桩类:仅满足 langgraph-sdk 导入，本地图编译不实例化此类型。"""

        def __init__(self, *args, **kwargs):
            pass

    _shim.AsyncChatModelStream = _StreamStub
    _shim.ChatModelStream = _StreamStub
    sys.modules[_shim_name] = _shim

# ---- 问题3:为 langchain_core.language_models._compat_bridge 注入桩模块 ----
_bridge_name = "langchain_core.language_models._compat_bridge"
if _bridge_name not in sys.modules:
    _bridge = types.ModuleType(_bridge_name)

    def message_to_events(*args, **kwargs):
        """桩函数:本地 StateGraph 不触发流式转换，返回空列表即可。"""
        return []

    _bridge.message_to_events = message_to_events
    sys.modules[_bridge_name] = _bridge
