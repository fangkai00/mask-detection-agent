# -*- coding: utf-8 -*-
"""
错误反馈闭环模块(Error Feedback Loop)

核心机制:
1. 不崩溃 + 错误回传 LLM:
   - safe_tool_call 包装工具调用,捕获所有异常
   - 异常转自然语言描述,回传为 AIMessage
   - LLM 在下一轮看到错误描述,自我修正参数后重试
   - 同一工具连续失败超 MAX_TOOL_ERRORS 后,planner 应放弃该工具

2. Few-Shot 记忆库(ErrorMemory):
   - 持久化到 logs/error_cases.jsonl
   - 每次工具失败 + LLM 修正成功,记录错误信息与解决思路
"""
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

import config

logger = logging.getLogger("mask_agent.error_feedback")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)
logger.propagate = False


def _cfg(key: str, default):
    return getattr(config, key, default)


def get_max_tool_retries() -> int:
    return int(_cfg("TOOL_MAX_RETRIES", 2))


def get_max_tool_errors() -> int:
    return int(_cfg("MAX_TOOL_ERRORS", 3))


def get_error_memory_file() -> str:
    rel = _cfg("ERROR_MEMORY_FILE", "logs/error_cases.jsonl")
    if os.path.isabs(rel):
        return rel
    return os.path.join(config.PROJECT_ROOT, rel)


def get_fewshot_max_cases() -> int:
    return int(_cfg("ERROR_FEWSHOT_MAX_CASES", 3))


# 瞬时错误判定(可代码级重试)
RETRIABLE_EXCEPTION_NAMES = {
    "ConnectionError", "TimeoutError", "ConnectTimeout", "ReadTimeout",
    "RateLimitError", "APITimeoutError", "APIConnectionError",
    "InternalServerError", "ServiceUnavailable",
}


def _is_retriable(exc: Exception) -> bool:
    for cls in type(exc).__mro__:
        if cls.__name__ in RETRIABLE_EXCEPTION_NAMES:
            return True
    return False


def _truncate(s: Any, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 3] + "..."


def error_to_natural_language(exc: Exception, tool_name: str, args: dict) -> str:
    """把异常转成 LLM 可理解的中文描述 + 修复建议。"""
    exc_type = type(exc).__name__
    msg = str(exc) or "(无错误消息)"
    args_brief = ", ".join(f"{k}={_truncate(v, 60)}" for k, v in args.items() if v is not None)

    if _is_retriable(exc):
        suggestion = "这是瞬时错误(网络/服务端),建议稍后重试同一调用,参数不变。"
    elif exc_type in ("ValueError", "KeyError", "TypeError"):
        suggestion = f"这通常是参数格式或类型错误,请检查参数:{args_brief}。建议修正参数后重试。"
    elif "FileNotFoundError" in exc_type:
        suggestion = "文件未找到,请检查 image_path 是否正确,或模型权重是否已训练完成。"
    elif "JSON" in msg.upper() or "PARSE" in msg.upper():
        suggestion = "返回内容解析失败,可能是 LLM 输出格式不符合预期。建议强化 JSON schema 约束。"
    else:
        suggestion = f"未识别的错误类型({exc_type})。建议检查参数({args_brief})或换用其他工具。"

    return (
        f"调用工具 {tool_name} 时发生错误:{exc_type} - {_truncate(msg, 300)}。"
        f"修复建议:{suggestion}"
    )


@dataclass
class ToolCallResult:
    """工具调用结果(成功或失败)。"""
    success: bool
    result: Any = None
    error_message: str = ""
    error_type: str = ""
    raw_exception: Optional[Exception] = None
    attempts: int = 1
    args: dict = field(default_factory=dict)


def safe_tool_call(
    tool,
    args: dict,
    tool_name: Optional[str] = None,
    max_retries: Optional[int] = None,
    retry_delay: float = 1.0,
) -> ToolCallResult:
    """通用工具调用包装器:捕获异常 → 转自然语言 → 返回 ToolCallResult。

    重试策略:
      - 瞬时错误(网络/超时/限流):代码级重试 max_retries 次
      - 参数/逻辑错误:立即返回(不代码重试,让 LLM 修正参数后下一轮重试)
    """
    name = tool_name or getattr(tool, "name", "unknown_tool")
    retries = max_retries if max_retries is not None else get_max_tool_retries()
    last_exc: Optional[Exception] = None
    attempts = 0

    for attempt in range(1, retries + 2):
        attempts = attempt
        try:
            if hasattr(tool, "invoke"):
                result = tool.invoke(args)
            elif hasattr(tool, "_run"):
                result = tool._run(**args)
            else:
                result = tool(**args)
            return ToolCallResult(success=True, result=result, attempts=attempts, args=args)
        except Exception as e:
            last_exc = e
            logger.warning(
                "[safe_tool_call] %s 第 %s/%s 次调用失败: %s: %s",
                name, attempt, retries + 1, type(e).__name__, _truncate(str(e), 120),
            )
            if _is_retriable(e) and attempt <= retries:
                time.sleep(retry_delay)
                continue
            break

    error_desc = error_to_natural_language(last_exc, name, args)
    return ToolCallResult(
        success=False,
        error_message=error_desc,
        error_type=type(last_exc).__name__,
        raw_exception=last_exc,
        attempts=attempts,
        args=args,
    )


def format_error_feedback_message(
    tool_name: str,
    error_message: str,
    attempt: int,
    max_tool_errors: Optional[int] = None,
) -> str:
    """构造回传给 LLM 的消息内容(自然语言错误描述)。"""
    mte = max_tool_errors if max_tool_errors is not None else get_max_tool_errors()
    return (
        f"[ToolError] 工具 {tool_name} 调用失败(第 {attempt}/{mte} 次)。\n"
        f"错误描述:{error_message}\n"
        f"请根据上述错误信息反思:1) 参数是否正确(image_path、数据类型)?"
        f"2) 是否需要换用其他工具(如 mask_detect 失败→改 search)?"
        f"3) 若属网络瞬时错误,可直接重试同一调用。"
        f"若同一工具累计失败 {mte} 次,请放弃该工具并选 finish 或换路径。"
    )


class ErrorMemory:
    """错误案例库:jsonl 持久化 + 查询。"""

    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path or get_error_memory_file()
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", encoding="utf-8") as f:
                pass

    def append(self, tool, error_type, error_message, args, fixed_args=None,
               resolution="", success_after_fix=False) -> dict:
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tool": tool,
            "error_type": error_type,
            "error_message": _truncate(error_message, 500),
            "args": {k: _truncate(v, 200) for k, v in (args or {}).items()},
            "fixed_args": (
                {k: _truncate(v, 200) for k, v in (fixed_args or {}).items()}
                if fixed_args else None
            ),
            "resolution": _truncate(resolution, 500),
            "success_after_fix": success_after_fix,
        }
        with self._lock:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("[ErrorMemory] 追加案例: tool=%s error=%s", tool, error_type)
        return record

    def load_all(self) -> List[dict]:
        cases = []
        if not os.path.exists(self.file_path):
            return cases
        with self._lock:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        cases.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return cases

    def query(self, tool=None, error_type=None, limit=None, only_successful=True) -> List[dict]:
        if limit is None:
            limit = get_fewshot_max_cases()
        cases = self.load_all()
        filtered = []
        for c in cases:
            if tool and c.get("tool") != tool:
                continue
            if error_type and c.get("error_type") != error_type:
                continue
            if only_successful and not c.get("success_after_fix"):
                continue
            filtered.append(c)
        filtered.sort(key=lambda c: c.get("timestamp", ""), reverse=True)
        return filtered[:limit]

    def stats(self) -> dict:
        cases = self.load_all()
        stats = {"total": len(cases), "by_tool": {}}
        for c in cases:
            t = c.get("tool", "unknown")
            stats["by_tool"][t] = stats["by_tool"].get(t, 0) + 1
        return stats


_memory_singleton: Optional[ErrorMemory] = None
_memory_lock = threading.Lock()


def get_error_memory() -> ErrorMemory:
    global _memory_singleton
    if _memory_singleton is None:
        with _memory_lock:
            if _memory_singleton is None:
                _memory_singleton = ErrorMemory()
    return _memory_singleton
