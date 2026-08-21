# -*- coding: utf-8 -*-
"""
RAGKnowledgeTool —— 基于 LlamaIndex 的本地知识库检索工具

复用项目内已训练 PDF(rag_data/):
1. 启动时检查 rag_data/md 是否有 .md 文件,无则调 pdf_to_md 自动转换
2. 加载 .md → 构建 VectorStoreIndex(首次) / 从持久化目录加载(后续)
3. 作为 LangGraph 节点工具:输入 query,返回检索到的文档片段 + 来源元数据

设计要点:
- 复用 DashScope LLM + DashScope Embedding(text-embedding-v2)
- 索引持久化到 config.RAG_STORAGE_DIR,避免每次重建
- 进程内单例:首次访问时构建,后续复用(避免每次查询重建索引)
- 兼容 vendor 目录安装的 llama-index-llms-dashscope(自动加 sys.path)
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

# vendor 目录(streamlit-paste-button / llama-index-llms-dashscope 等装在此)
_VENDOR = os.path.join(_ROOT, "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

import compat  # noqa: E402

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

import config
from tools.pdf_to_md import ensure_md_files

logger = logging.getLogger("mask_agent.rag")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)
logger.propagate = False


# ============================================================
# LlamaIndex 懒加载(避免无 RAG 需求时也加载全部依赖)
# ============================================================
def _import_llamaindex():
    """延迟导入 LlamaIndex,失败时给出清晰报错。"""
    try:
        from llama_index.core import (
            VectorStoreIndex,
            SimpleDirectoryReader,
            Settings,
            StorageContext,
            load_index_from_storage,
        )
        from llama_index.embeddings.dashscope import (
            DashScopeEmbedding,
            DashScopeTextEmbeddingModels,
        )
        from llama_index.llms.dashscope import DashScope
        return {
            "VectorStoreIndex": VectorStoreIndex,
            "SimpleDirectoryReader": SimpleDirectoryReader,
            "Settings": Settings,
            "StorageContext": StorageContext,
            "load_index_from_storage": load_index_from_storage,
            "DashScope": DashScope,
            "DashScopeEmbedding": DashScopeEmbedding,
            "DashScopeTextEmbeddingModels": DashScopeTextEmbeddingModels,
        }
    except ImportError as e:
        raise ImportError(
            f"加载 LlamaIndex 失败: {e}。"
            "请安装: pip install llama-index llama-index-embeddings-dashscope "
            "llama-index-llms-dashscope"
        ) from e


class RAGSearchInput(BaseModel):
    query: str = Field(description="知识库检索查询,如:口罩适合性检测的标准是什么")


# ============================================================
# 索引管理(单例,线程安全)
# ============================================================
_index_lock = threading.Lock()
_index_singleton = None  # VectorStoreIndex 实例
_query_engine_singleton = None
_init_error: Optional[str] = None  # 首次初始化失败的错误信息


def _setup_llm_and_embedding(lib: dict):
    """配置 LlamaIndex 全局 Settings.llm 与 Settings.embed_model。

    使用 DashScope(text-embedding-v2 + qwen),API Key 从 config.DASHSCOPE_API_KEY 取。
    """
    api_key = config.DASHSCOPE_API_KEY
    if not api_key:
        raise ValueError("未配置 DASHSCOPE_API_KEY,RAG 工具无法使用 DashScope Embedding")

    # Embedding 模型(默认 text-embedding-v2,中英文兼容)
    # 关键:DashScope API 单次最多 20 条,默认 embed_batch_size=25 会超限,需 ≤ 20
    embed_model_name = getattr(config, "RAG_EMBED_MODEL", "text-embedding-v2")
    embed_model = lib["DashScopeEmbedding"](
        model_name=lib["DashScopeTextEmbeddingModels"].TEXT_EMBEDDING_V2
        if embed_model_name == "text-embedding-v2"
        else embed_model_name,
        api_key=api_key,
        embed_batch_size=10,  # DashScope API 限制:单批 ≤ 20,留余量取 10
    )

    # 查询引擎用的 LLM(默认与主决策 LLM 一致)
    llm_model = getattr(config, "RAG_LLM_MODEL", "") or getattr(config, "MAIN_LLM_MODEL", "qwen-max")
    llm = lib["DashScope"](
        model=llm_model,
        api_key=api_key,
        temperature=0.3,  # 知识问答略高温度,允许一定归纳
        top_p=0.8,
    )

    # 注入全局 Settings(LlamaIndex 内部检索/合成答案时用)
    lib["Settings"].llm = llm
    lib["Settings"].embed_model = embed_model
    # 文档分块大小(默认 1024,中文 PDF 文本较稀疏,可适当增大减少分块数)
    lib["Settings"].chunk_size = 512
    lib["Settings"].chunk_overlap = 50
    return llm, embed_model


def _load_or_build_index(lib: dict):
    """加载持久化索引;不存在则从 MD 文件构建并持久化。

    返回 VectorStoreIndex。
    """
    md_dir = Path(config.RAG_MD_DIR)
    storage_dir = Path(config.RAG_STORAGE_DIR)

    # 1. 优先从持久化目录加载(快)
    if storage_dir.exists() and any(storage_dir.iterdir()):
        try:
            storage_context = lib["StorageContext"].from_defaults(persist_dir=str(storage_dir))
            index = lib["load_index_from_storage"](storage_context)
            logger.info("[rag] 从持久化目录加载索引: %s", storage_dir)
            return index
        except Exception as e:
            logger.warning("[rag] 加载持久化索引失败,将重建: %s", e)

    # 2. 检查 MD 目录是否有文件
    if not md_dir.exists() or not any(md_dir.glob("*.md")):
        logger.info("[rag] MD 目录无 .md 文件,触发 PDF→MD 转换: %s", md_dir)
        ensure_md_files(config.RAG_PDF_DIR, config.RAG_MD_DIR)

    md_files = list(md_dir.glob("*.md")) if md_dir.exists() else []
    if not md_files:
        raise FileNotFoundError(
            f"RAG 知识库无可用 MD 文件(已尝试 PDF→MD 转换)。"
            f"请把 PDF 放到 {config.RAG_PDF_DIR} 或 MD 放到 {config.RAG_MD_DIR}"
        )

    logger.info("[rag] 构建向量索引: %s 个 MD 文件", len(md_files))
    reader = lib["SimpleDirectoryReader"](str(md_dir))
    documents = reader.load_data()
    if not documents:
        raise RuntimeError(f"SimpleDirectoryReader 未从 {md_dir} 加载到任何文档")

    index = lib["VectorStoreIndex"].from_documents(documents)
    storage_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(storage_dir))
    logger.info("[rag] 索引已持久化到 %s", storage_dir)
    return index


def get_query_engine():
    """获取(必要时构建)LlamaIndex 查询引擎。线程安全单例。"""
    global _index_singleton, _query_engine_singleton, _init_error

    if _query_engine_singleton is not None:
        return _query_engine_singleton
    if _init_error is not None:
        raise RuntimeError(f"RAG 索引初始化失败(不会重试,需重启服务): {_init_error}")

    with _index_lock:
        # 双重检查
        if _query_engine_singleton is not None:
            return _query_engine_singleton
        if _init_error is not None:
            raise RuntimeError(f"RAG 索引初始化失败(不会重试,需重启服务): {_init_error}")

        try:
            lib = _import_llamaindex()
            _setup_llm_and_embedding(lib)
            index = _load_or_build_index(lib)
            top_k = int(getattr(config, "RAG_TOP_K", 4))
            query_engine = index.as_query_engine(similarity_top_k=top_k)
            _index_singleton = index
            _query_engine_singleton = query_engine
            logger.info("[rag] 查询引擎就绪(top_k=%s)", top_k)
            return _query_engine_singleton
        except Exception as e:
            _init_error = str(e)
            logger.error("[rag] 初始化失败: %s", e)
            raise


def reset_rag_state():
    """重置单例(配置变更或重新构建索引时调用)。"""
    global _index_singleton, _query_engine_singleton, _init_error
    with _index_lock:
        _index_singleton = None
        _query_engine_singleton = None
        _init_error = None


def retrieve_documents(query: str, top_k: Optional[int] = None) -> List[dict]:
    """同步检索文档,返回结构化片段列表。

    Returns:
        [{content, source, page, score}, ...]
    """
    engine = get_query_engine()
    # 注:retrieve 只返回节点(不调用 LLM 合成),成本更低
    retriever = engine.retriever if hasattr(engine, "retriever") else None
    if retriever is None:
        # query_engine.as_retriever() 在 LlamaIndex 中是索引方法,这里从 engine 取
        # 多数 query_engine 实例有 .retriever 属性
        from llama_index.core import Settings  # noqa
        retriever = _index_singleton.as_retriever(similarity_top_k=top_k or int(getattr(config, "RAG_TOP_K", 4)))

    nodes = retriever.retrieve(query)
    results = []
    for n in nodes:
        meta = n.metadata or {}
        results.append({
            "content": (n.text or "")[:800],  # 截断,避免回传给 LLM 过长
            "source": meta.get("file_name") or meta.get("file_path") or "(未知来源)",
            "page": meta.get("page_label") or meta.get("page") or "",
            "score": float(getattr(n, "score", 0.0)) if getattr(n, "score", None) is not None else 0.0,
        })
    return results


# ============================================================
# LangChain BaseTool 封装(供 graph.py 节点调用)
# ============================================================
class RAGKnowledgeTool(BaseTool):
    """本地知识库检索工具(基于 LlamaIndex + DashScope)。

    首次调用时自动构建索引(可能耗时 30s+),后续调用复用持久化索引。
    返回检索到的文档片段 JSON 数组,供 finalizer 综合归纳。
    """

    name: str = "rag_search"
    description: str = (
        "本地知识库检索(LlamaIndex + DashScope Embedding),从已加载的 PDF/MD 文档"
        "中检索口罩、生物安全、实验室防护等专业领域知识。"
        "用于口罩适合性、佩戴标准、实验室规范等专业问题。"
    )
    args_schema: type = RAGSearchInput

    def _run(self, query: str) -> List[dict]:
        try:
            results = retrieve_documents(query)
            return results
        except Exception as e:
            logger.error("[rag_search] 检索失败: %s", e)
            return [{
                "content": "",
                "source": "(检索失败)",
                "page": "",
                "score": 0.0,
                "error": str(e)[:200],
            }]


if __name__ == "__main__":
    # 独立测试:python tools/rag_knowledge.py "查询问题"
    logging.basicConfig(level=logging.INFO)
    q = sys.argv[1] if len(sys.argv) > 1 else "口罩适合性检测的标准是什么"
    print(f"\n查询: {q}\n" + "=" * 60)
    try:
        for i, r in enumerate(retrieve_documents(q), 1):
            print(f"\n[{i}] 来源: {r.get('source')} 第{r.get('page','?')}页 (score={r.get('score',0):.3f})")
            print(f"    {r.get('content','')[:200]}...")
    except Exception as e:
        print(f"失败: {e}")
