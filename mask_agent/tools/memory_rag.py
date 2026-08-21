# -*- coding: utf-8 -*-
"""
对话长期记忆 —— 基于 LlamaIndex 的对话历史向量索引

职责定位(与短期记忆、文档知识库区分):
- 短期记忆:agent/cli.py `_build_state` 把当前会话最近 N 轮直接拼到
  `state.messages`,作为 LLM 上下文。窗口有限,只覆盖近邻对话。
- 长期记忆(本模块):把每一轮对话(user/assistant)向量化后写入持久化向量库,
  下一轮对话开始前用 user_query 语义检索召回"与当前问题相关的早期历史片段",
  突破短期窗口与时间衰减限制,实现跨会话、跨重启的长期记忆。
- 文档知识库:tools/rag_knowledge.py 检索 rag_data/ 下 PDF/MD 专业文档,
  与本模块完全独立,不要把 RAG 工具误当作长期记忆。

设计要点:
- 复用 DashScope text-embedding-v2(与 rag_knowledge.py 同一套 embedding),
  但持久化目录独立(config.MEMORY_INDEX_DIR),避免与文档知识库向量混淆。
- 增量写入:每轮对话结束由 ConversationSession 调 `add_turn`,
  以 `sid#turn_idx` 作为稳定 ref_doc_id 幂等 upsert。
- 召回过滤:recall 时可传 `exclude_sid` / `exclude_recent` 跳过当前会话最近
  N 轮(它们已在短期窗口中),避免短期/长期记忆内容重复。
- 进程内单例 + 线程锁,与 rag_knowledge.py 风格一致。
- 失败不阻塞主流程:任何异常捕获后返回空结果或默认值,对话继续。
"""
import logging
import os
import sys
import threading
from pathlib import Path
from typing import List, Optional

# 项目根入 sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# vendor 目录(llama-index-llms-dashscope 等装在此)
_VENDOR = os.path.join(_ROOT, "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

import compat  # noqa: E402

import config
from tools.rag_knowledge import _import_llamaindex  # 复用 LlamaIndex 延迟导入

logger = logging.getLogger("mask_agent.memory_rag")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)
logger.propagate = False


# ============================================================
# 单例管理(线程安全)
# ============================================================
_index_lock = threading.Lock()
_index_singleton = None  # VectorStoreIndex 实例
_storage_context_singleton = None
_init_error: Optional[str] = None  # 首次初始化失败的错误信息


def _setup_embedding(lib: dict):
    """配置 LlamaIndex 全局 Settings.embed_model(仅 embedding,长期记忆召回不需要 LLM 合成)。

    复用 rag_knowledge.py 用的 DashScope text-embedding-v2;若 Settings 已被
    rag_knowledge 配置过则直接沿用,避免重复创建 embedding 客户端。
    """
    api_key = config.DASHSCOPE_API_KEY
    if not api_key:
        raise ValueError("未配置 DASHSCOPE_API_KEY,长期记忆向量索引无法使用 DashScope Embedding")

    # 若 rag_knowledge 已注入过 embed_model,直接复用
    existing = getattr(lib["Settings"], "embed_model", None)
    if existing is not None:
        return existing

    embed_model_name = getattr(config, "RAG_EMBED_MODEL", "text-embedding-v2")
    embed_model = lib["DashScopeEmbedding"](
        model_name=lib["DashScopeTextEmbeddingModels"].TEXT_EMBEDDING_V2
        if embed_model_name == "text-embedding-v2"
        else embed_model_name,
        api_key=api_key,
        embed_batch_size=10,  # DashScope API 限制:单批 ≤ 20,留余量取 10
    )
    lib["Settings"].embed_model = embed_model
    return embed_model


def _load_or_create_index(lib: dict):
    """加载持久化向量索引;不存在则创建空索引并持久化目录。"""
    storage_dir = Path(getattr(config, "MEMORY_INDEX_DIR",
                               os.path.join(config.PROJECT_ROOT, "data", "memory_index")))
    storage_dir.mkdir(parents=True, exist_ok=True)

    # 1. 优先从持久化目录加载
    if storage_dir.exists() and any(storage_dir.iterdir()):
        try:
            storage_context = lib["StorageContext"].from_defaults(persist_dir=str(storage_dir))
            index = lib["load_index_from_storage"](storage_context)
            logger.info("[memory_rag] 从持久化目录加载长期记忆索引: %s", storage_dir)
            return index, storage_context
        except Exception as e:
            logger.warning("[memory_rag] 加载持久化索引失败,将重建空索引: %s", e)

    # 2. 首次:建空索引(无文档)并持久化
    index = lib["VectorStoreIndex"]([])
    index.storage_context.persist(persist_dir=str(storage_dir))
    logger.info("[memory_rag] 初始化空长期记忆索引,持久化到 %s", storage_dir)
    return index, index.storage_context


def _ensure_index():
    """获取(必要时构建)长期记忆向量索引。线程安全单例。"""
    global _index_singleton, _storage_context_singleton, _init_error

    if _index_singleton is not None:
        return _index_singleton
    if _init_error is not None:
        raise RuntimeError(f"长期记忆索引初始化失败(不会重试,需重启服务): {_init_error}")

    with _index_lock:
        if _index_singleton is not None:
            return _index_singleton
        if _init_error is not None:
            raise RuntimeError(f"长期记忆索引初始化失败(不会重试,需重启服务): {_init_error}")

        try:
            lib = _import_llamaindex()
            _setup_embedding(lib)
            index, storage_ctx = _load_or_create_index(lib)
            _index_singleton = index
            _storage_context_singleton = storage_ctx
            logger.info("[memory_rag] 长期记忆索引就绪")
            return _index_singleton
        except Exception as e:
            _init_error = str(e)
            logger.error("[memory_rag] 初始化失败: %s", e)
            raise


def _persist():
    """把当前索引状态持久化到 MEMORY_INDEX_DIR。"""
    if _storage_context_singleton is None:
        return
    storage_dir = Path(getattr(config, "MEMORY_INDEX_DIR",
                               os.path.join(config.PROJECT_ROOT, "data", "memory_index")))
    _storage_context_singleton.persist(persist_dir=str(storage_dir))


def _ref_id(sid: str, turn_idx: int) -> str:
    """生成稳定的 ref_doc_id,用于幂等 upsert。"""
    safe_sid = (sid or "unknown").replace("/", "_").replace("#", "_")
    return f"{safe_sid}#{turn_idx}"


def add_turn(sid: str, turn: dict, turn_idx: int) -> None:
    """把一轮对话作为 Document 写入长期记忆向量索引(幂等 upsert)。

    Args:
        sid: 会话 ID
        turn: ConversationTurn.to_dict() 输出,至少含 user_input/assistant_output
        turn_idx: 该轮在会话中的全局序号(从 1 开始)

    Note:
        - 同一 (sid, turn_idx) 重复写入会先删旧版再插入,保证幂等。
        - 任何异常被捕获并记日志,不抛出,避免阻塞对话主流程。
    """
    user_text = (turn.get("user_input") or "").strip()
    assistant_text = (turn.get("assistant_output") or "").strip()
    if not user_text and not assistant_text:
        return

    # 截断,避免单条向量过长(召回时再二次截断)
    if len(user_text) > 800:
        user_text = user_text[:800] + "..."
    if len(assistant_text) > 1500:
        assistant_text = assistant_text[:1500] + "..."

    doc_text = (
        f"[会话{sid} 第{turn_idx}轮]\n"
        f"用户: {user_text}\n"
        f"助手: {assistant_text}"
    )
    ref_id = _ref_id(sid, turn_idx)

    try:
        index = _ensure_index()
        from llama_index.core import Document as LIDocument
        doc = LIDocument(
            text=doc_text,
            metadata={
                "sid": sid or "unknown",
                "turn": turn_idx,
                "ref_doc_id": ref_id,
            },
        )
        # 幂等 upsert:先删旧版(忽略 not found),再插入新版
        try:
            index.delete(ref_doc_id=ref_id)
        except Exception:
            pass  # 不存在或删除失败,忽略
        index.insert(doc)
        _persist()
        logger.debug("[memory_rag] 写入长期记忆: %s", ref_id)
    except Exception as e:
        logger.warning("[memory_rag] 写入长期记忆失败(sid=%s, turn=%s): %s", sid, turn_idx, e)


def recall(query: str,
           top_k: Optional[int] = None,
           exclude_sid: Optional[str] = None,
           exclude_recent: int = 0) -> List[dict]:
    """用当前 query 检索长期记忆,返回相关历史对话片段。

    Args:
        query: 当前轮用户问题(用于语义检索)
        top_k: 返回片段数;None 用 config.MEMORY_RECALL_TOP_K
        exclude_sid: 若提供且 exclude_recent>0,过滤掉该会话的所有轮次
                     (它们已在 ConversationSession.turns / 短期窗口中,避免重复)
        exclude_recent: >0 时与 exclude_sid 配合启用过滤(语义见 exclude_sid)

    Returns:
        [{content, sid, turn, score}, ...] 按 score 降序;失败返回空列表。
    """
    if not query or not query.strip():
        return []
    k = int(top_k or getattr(config, "MEMORY_RECALL_TOP_K", 3))
    if k <= 0:
        return []

    try:
        index = _ensure_index()
        retriever = index.as_retriever(similarity_top_k=k * 3 if exclude_sid else k)
        nodes = retriever.retrieve(query)
    except Exception as e:
        logger.warning("[memory_rag] 召回失败(query=%s): %s", query[:50], e)
        return []

    # 结果层过滤:排除当前会话已在本会话 self.turns 内的轮次
    # (本会话所有 turn 已由 ConversationSession 在内存持有,短期窗口或本会话上下文
    #  已覆盖,无需长期记忆再召回一次,避免重复)
    results = []
    for n in nodes:
        meta = n.metadata or {}
        n_sid = meta.get("sid", "")
        n_turn = meta.get("turn", 0)
        if exclude_sid and n_sid == exclude_sid and exclude_recent > 0:
            continue
        score = float(getattr(n, "score", 0.0)) if getattr(n, "score", None) is not None else 0.0
        results.append({
            "content": (n.text or "")[:600],
            "sid": n_sid,
            "turn": n_turn,
            "score": score,
        })
        if len(results) >= k:
            break

    return results


def reset_memory():
    """清空长期记忆索引(主要用于调试/重置)。"""
    global _index_singleton, _storage_context_singleton, _init_error
    with _index_lock:
        _index_singleton = None
        _storage_context_singleton = None
        _init_error = None
    # 物理清空持久化目录
    storage_dir = Path(getattr(config, "MEMORY_INDEX_DIR",
                               os.path.join(config.PROJECT_ROOT, "data", "memory_index")))
    if storage_dir.exists():
        for f in storage_dir.iterdir():
            try:
                if f.is_file():
                    f.unlink()
            except Exception:
                pass
    logger.info("[memory_rag] 长期记忆已清空")


if __name__ == "__main__":
    # 独立测试:python tools/memory_rag.py "查询问题"
    logging.basicConfig(level=logging.INFO)
    q = sys.argv[1] if len(sys.argv) > 1 else "口罩适合性检测的标准是什么"
    print(f"\n查询: {q}\n" + "=" * 60)
    try:
        add_turn("test_sid", {"user_input": "什么是口罩适合性检测", "assistant_output": "..."}, 1)
        for i, r in enumerate(recall(q), 1):
            print(f"\n[{i}] sid={r.get('sid')} turn={r.get('turn')} score={r.get('score',0):.3f}")
            print(f"    {r.get('content','')[:200]}...")
    except Exception as e:
        print(f"失败: {e}")
