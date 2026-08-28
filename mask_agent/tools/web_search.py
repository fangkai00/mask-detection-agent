# -*- coding: utf-8 -*-
"""
WebSearchTool —— 基于 Tavily 的联网搜索

后端选择逻辑(由 config.SEARCH_PROVIDER 控制):
  auto        : Tavily(有 key 且库可用)→ DuckDuckGo(库可用)→ 不可用提示
  tavily      : 强制 Tavily(运行时不可用则回退 DuckDuckGo)
  duckduckgo  : 强制 DuckDuckGo(运行时不可用则回退 Tavily)
  none        : 不联网,返回不可用提示

Tavily 不依赖 LLM_PROVIDER(DashScope/Ollama 均可),与主决策 LLM 解耦。
返回 List[dict],每条含 {url, title, snippet},供 search_node 拼装消息。

依赖(任选其一,均装更稳):
  pip install tavily-python            # 推荐,需 API Key
  pip install duckduckgo-search       # 无需 Key,国内网络可能不可用
  国内镜像: -i https://mirrors.aliyun.com/pypi/simple/
"""
import asyncio
import logging
import os
import sys
from typing import List, Optional

# 项目根入 sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import compat  # noqa: E402

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

import config

logger = logging.getLogger("mask_agent")

# 单条结果摘要截断上限
MAX_SNIPPET_CHARS = 500
# 兜底默认返回结果数(配置缺失时使用)
_DEFAULT_MAX_RESULTS = 5


class WebSearchInput(BaseModel):
    query: str = Field(description="搜索查询,如:2024年口罩佩戴政策")
    num_results: int = Field(default=5, description="期望返回结果数")


class WebSearchTool(BaseTool):
    """联网搜索工具,基于 Tavily(可回退 DuckDuckGo)。

    与主 LLM 提供方解耦:无论 LLM_PROVIDER=dashscope 或 ollama,
    联网搜索均走 Tavily/ DuckDuckGo,不再依赖模型的 enable_search 能力。

    返回 List[dict],每条含 url/title/snippet。
    """

    name: str = "web_search"
    description: str = (
        "联网搜索(Tavily,可回退 DuckDuckGo),返回包含 url/title/snippet 的结果列表。"
        "用于口罩政策、防护知识、公共卫生新闻等时效性信息查询。"
    )
    args_schema: type = WebSearchInput

    # ============================================================
    # BaseTool 入口
    # ============================================================
    def _run(self, query: str, num_results: Optional[int] = None) -> List[dict]:
        return self._do_search(query, num_results)

    async def _arun(self, query: str, num_results: Optional[int] = None) -> List[dict]:
        # Tavily / DDG 客户端均为同步阻塞,放到线程池避免卡住事件循环
        return await asyncio.to_thread(self._do_search, query, num_results)

    # ============================================================
    # 配置读取
    # ============================================================
    def _provider(self) -> str:
        return str(getattr(config, "SEARCH_PROVIDER", "auto") or "auto").lower().strip()

    def _tavily_key(self) -> str:
        return str(getattr(config, "TAVILY_API_KEY", "") or "").strip()

    def _max_results(self, num_results: Optional[int]) -> int:
        """确定本次搜索返回条数上限。

        优先级:显式传入 > config.SEARCH_NUM_RESULTS > config.TAVILY_MAX_RESULTS > 默认值。
        """
        for v in (num_results,
                  getattr(config, "SEARCH_NUM_RESULTS", None),
                  getattr(config, "TAVILY_MAX_RESULTS", None)):
            if v is None:
                continue
            try:
                return max(1, int(v))
            except (TypeError, ValueError):
                continue
        return _DEFAULT_MAX_RESULTS

    # ============================================================
    # 后端探测
    # ============================================================
    @staticmethod
    def _tavily_available() -> bool:
        try:
            import tavily  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _import_ddgs():
        """导入 DDGS 类,兼容新包名 ddgs 与旧包名 duckduckgo_search。"""
        try:
            from ddgs import DDGS
            return DDGS
        except ImportError:
            pass
        try:
            from duckduckgo_search import DDGS
            return DDGS
        except ImportError:
            return None

    def _ddg_available(self) -> bool:
        return self._import_ddgs() is not None

    def _chosen_backend(self) -> str:
        """在每次调用时确定使用的后端(auto 按 Tavily→DDG 顺序探测)。"""
        provider = self._provider()
        tavily_ok = bool(self._tavily_key()) and self._tavily_available()
        ddg_ok = self._ddg_available()

        if provider == "tavily":
            return "tavily" if tavily_ok else ("duckduckgo" if ddg_ok else "none")
        if provider == "duckduckgo":
            return "duckduckgo" if ddg_ok else ("tavily" if tavily_ok else "none")
        if provider == "none":
            return "none"
        # auto:Tavily 优先(质量高),不可用回退 DDG
        if tavily_ok:
            return "tavily"
        if ddg_ok:
            return "duckduckgo"
        return "none"

    # ============================================================
    # 调度
    # ============================================================
    def _do_search(self, query: str, num_results: Optional[int]) -> List[dict]:
        chosen = self._chosen_backend()
        n = self._max_results(num_results)

        if chosen == "tavily":
            res = self._search_tavily(query, n)
            if res is not None:
                return res
            # Tavily 运行时不可用 → 回退 DDG
            if self._ddg_available():
                res = self._search_duckduckgo(query, n)
                if res is not None:
                    return res
            return self._fallback_msg(query)

        if chosen == "duckduckgo":
            res = self._search_duckduckgo(query, n)
            if res is not None:
                return res
            # DDG 运行时不可用 → 回退 Tavily(若有 key)
            if self._tavily_key() and self._tavily_available():
                res = self._search_tavily(query, n)
                if res is not None:
                    return res
            return self._fallback_msg(query)

        return self._fallback_msg(query)

    # ============================================================
    # Tavily
    # ============================================================
    def _search_tavily(self, query: str, max_results: int) -> Optional[List[dict]]:
        """使用 Tavily 搜索,返回结构化结果;库缺失/异常返回 None(触发上层回退)。"""
        api_key = self._tavily_key()
        if not api_key:
            logger.warning("[web_search] 未配置 TAVILY_API_KEY,跳过 Tavily。")
            return None
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_key)
            resp = client.search(query=query, max_results=max_results)
        except ImportError:
            return None
        except Exception as e:
            logger.warning("[web_search] Tavily 搜索失败: %s: %s", type(e).__name__, e)
            return None

        raw = resp.get("results", []) if isinstance(resp, dict) else []
        results: List[dict] = []
        for r in raw:
            snippet = str(r.get("content", "") or r.get("snippet", "")).strip()
            if len(snippet) > MAX_SNIPPET_CHARS:
                snippet = snippet[:MAX_SNIPPET_CHARS] + "..."
            results.append({
                "url": str(r.get("url", "")).strip(),
                "title": str(r.get("title", "")).strip(),
                "snippet": snippet,
            })
        logger.info("[web_search] Tavily 返回 %s 条结果(query=%r)", len(results), query)
        return results

    # ============================================================
    # DuckDuckGo(兜底,无需 Key)
    # ============================================================
    def _search_duckduckgo(self, query: str, max_results: int) -> Optional[List[dict]]:
        """使用 DuckDuckGo 搜索,返回结构化结果;库缺失/异常返回 None。"""
        DDGS = self._import_ddgs()
        if DDGS is None:
            return None
        try:
            ddgs = DDGS()
            try:
                raw = ddgs.text(query, max_results=max_results)
            finally:
                close = getattr(ddgs, "close", None)
                if callable(close):
                    close()
        except Exception as e:
            logger.warning("[web_search] DuckDuckGo 搜索失败: %s: %s", type(e).__name__, e)
            return None

        results: List[dict] = []
        for r in (raw or []):
            snippet = str(r.get("body", r.get("snippet", ""))).strip()
            if len(snippet) > MAX_SNIPPET_CHARS:
                snippet = snippet[:MAX_SNIPPET_CHARS] + "..."
            results.append({
                "url": str(r.get("href", r.get("url", ""))).strip(),
                "title": str(r.get("title", "")).strip(),
                "snippet": snippet,
            })
        logger.info("[web_search] DuckDuckGo 返回 %s 条结果(query=%r)", len(results), query)
        return results

    # ============================================================
    # 兜底提示
    # ============================================================
    @staticmethod
    def _fallback_msg(query: str) -> List[dict]:
        """后端均不可用时返回友好提示(不抛异常,保证图继续流转)。"""
        msg = (
            "搜索功能不可用。请安装搜索依赖之一:\n"
            "  Tavily(推荐,需 API Key): pip install tavily-python\n"
            "  DuckDuckGo(无需 Key):     pip install duckduckgo-search\n"
            "国内镜像: -i https://mirrors.aliyun.com/pypi/simple/\n"
            "并在 config.yaml 中填入 TAVILY_API_KEY。"
        )
        logger.warning("[web_search] 联网搜索不可用(query=%r)", query)
        return [{"url": "", "title": f"搜索不可用: {query}", "snippet": msg}]


if __name__ == "__main__":
    q = "2024年口罩佩戴政策"
    print(f"查询: {q}")
    print(f"后端选择: {WebSearchTool()._chosen_backend()}")
    for i, r in enumerate(WebSearchTool().invoke({"query": q}), 1):
        print(f"\n[{i}] {r.get('title','')}\n    {r.get('url','')}\n    {r.get('snippet','')[:80]}")
