# -*- coding: utf-8 -*-
"""
会话历史持久化(JSON 文件,每会话一文件)

存储位置:config.MEMORY_DIR(默认 data/memory/)
文件名:  session_{sid}.json
内容:    {session_id, created_at, updated_at, turns: [{...}, ...]}

保留范围:对话文本 + 元数据(用户输入、助手回答、图片路径、路由、耗时、步数)
         不含 fields 结构化思考过程和完整检测结果(体积控制)

用途:Streamlit session_state 是进程内内存,重启 GUI 进程即丢失。
      本模块把会话落盘,GUI 启动时加载,实现跨重启的历史对话查看。
"""
import json
import os
import re
import sys
import time
from typing import List, Optional

# 项目根入 sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config


def _store_dir() -> str:
    """返回会话存储目录(绝对路径),不存在则创建。"""
    d = getattr(config, "MEMORY_DIR", "data/memory")
    if not os.path.isabs(d):
        d = os.path.join(config.PROJECT_ROOT, d)
    os.makedirs(d, exist_ok=True)
    return d


def _safe_sid(sid: str) -> str:
    """校验 sid 只含安全字符(防路径穿越),非法字符替换为 _。"""
    if not sid:
        return "unknown"
    return re.sub(r"[^A-Za-z0-9_-]", "_", sid)


def _session_path(sid: str) -> str:
    """返回指定会话的 JSON 文件路径。"""
    return os.path.join(_store_dir(), f"session_{_safe_sid(sid)}.json")


def save_session(sid: str, turns_data: list, created_at: Optional[float] = None) -> str:
    """保存会话到磁盘(覆盖写)。

    Args:
        sid: 会话 ID
        turns_data: turns 序列化后的 list(每个元素是 dict,见 ConversationTurn.to_dict)
        created_at: 创建时间戳;None 表示首次保存,用当前时间

    Returns:
        保存的文件路径
    """
    path = _session_path(sid)
    now = time.time()
    # 首次保存或文件不存在时记录 created_at;否则保留原 created_at
    if created_at is None:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    old = json.load(f)
                created_at = old.get("created_at", now)
            except Exception:
                created_at = now
        else:
            created_at = now

    payload = {
        "session_id": sid,
        "created_at": created_at,
        "updated_at": now,
        "turns": turns_data,
    }
    # 原子写:先写临时文件再 rename,避免半截写坏
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return path


def load_session(sid: str) -> Optional[dict]:
    """加载单个会话,返回 {session_id, created_at, updated_at, turns} 或 None。"""
    path = _session_path(sid)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def list_sessions() -> List[dict]:
    """列出所有持久化会话,按 updated_at 降序(最近更新在前)。

    返回每个会话的摘要:[{session_id, created_at, updated_at, turn_count, first_query}, ...]
    """
    d = _store_dir()
    items = []
    for name in os.listdir(d):
        if not name.startswith("session_") or not name.endswith(".json"):
            continue
        path = os.path.join(d, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        sid = data.get("session_id") or name[len("session_"):-len(".json")]
        turns = data.get("turns", []) or []
        first_query = turns[0].get("user_input", "")[:20] if turns else ""
        items.append({
            "session_id": sid,
            "created_at": data.get("created_at", 0),
            "updated_at": data.get("updated_at", 0),
            "turn_count": len(turns),
            "first_query": first_query,
        })
    # 按 updated_at 降序
    items.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
    return items
