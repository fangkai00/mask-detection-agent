# -*- coding: utf-8 -*-
"""
WebSearchTool —— 基于 DashScope Qwen enable_search 的联网搜索

复用 DashScope OpenAI 兼容接口的 enable_search 能力。
用于口罩政策、防护知识、公共卫生新闻等时效性信息查询。

流程:
1. 在 extra_body 中传 enable_search=True,触发服务端联网搜索
2. 提示 Qwen 将搜索结果整理为 JSON 数组 [{url, title, snippet}]
3. 解析返回文本,提取结构化结果
"""
import asyncio
import json
import os
import re
import sys
from typing import List, Optional

# 项目根入 sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import compat  # noqa: E402

from openai import AsyncOpenAI
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

import config
from prompts.tools_prompts import format_web_search_user_prompt


class WebSearchInput(BaseModel):
    query: str = Field(description="搜索查询,如:2024年口罩佩戴政策")
    num_results: int = Field(default=5, description="期望返回结果数")


# ============================================================
# 结构化输出模型(用于 response_format json_schema 强约束)
# ============================================================
class SearchResultItem(BaseModel):
    url: str = ""
    title: str = ""
    snippet: str = ""


class SearchResultList(BaseModel):
    results: List[SearchResultItem] = Field(default_factory=list)


def _build_search_response_format() -> dict:
    """基于 SearchResultList Pydantic 模型构造 DashScope/OpenAI 兼容的
    response_format(json_schema strict)。

    strict 模式要求:
      - 不接受 $ref / $defs(必须内联)
      - additionalProperties=false
      - 所有字段 required
    为避免 $ref 展开的复杂性,这里直接手写与 SearchResultList 一致的 schema,
    并用断言保证与 Pydantic 模型字段同步。
    """
    # 字段一致性自检(防止 Pydantic 模型改了但这里漏改)
    _expected = {"url", "title", "snippet"}
    assert set(SearchResultItem.model_fields.keys()) == _expected, (
        "SearchResultItem 字段变更,请同步更新 _build_search_response_format"
    )
    assert set(SearchResultList.model_fields.keys()) == {"results"}, (
        "SearchResultList 字段变更,请同步更新 _build_search_response_format"
    )

    item_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "title": {"type": "string"},
            "snippet": {"type": "string"},
        },
        "required": ["url", "title", "snippet"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": item_schema,
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "search_results",
            "schema": schema,
            "strict": True,
        },
    }


class WebSearchTool(BaseTool):
    """联网搜索工具,按 LLM_PROVIDER 在 DashScope(enable_search)/ Ollama(纯知识)间切换。

    - dashscope:用 Qwen enable_search 真联网,返回 {url,title,snippet} 列表
    - ollama:本地模型不联网,基于模型内置知识返回(可能无 url,字段为空)
    """

    name: str = "web_search"
    description: str = (
        "联网搜索(DashScope)或知识检索(Ollama),返回包含 url/title/snippet 的结果列表。"
        "用于口罩政策、防护知识、公共卫生新闻等时效性信息查询。"
    )
    args_schema: type = WebSearchInput

    def _provider(self) -> str:
        return (getattr(config, "LLM_PROVIDER", "dashscope") or "dashscope").lower()

    def _client(self) -> AsyncOpenAI:
        """按 LLM_PROVIDER 创建 OpenAI 兼容客户端。"""
        if self._provider() == "ollama":
            return AsyncOpenAI(
                base_url=getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                api_key=getattr(config, "OLLAMA_API_KEY", "ollama") or "ollama",
            )
        return AsyncOpenAI(
            base_url=config.LLM_BASE_URL, api_key=config.DASHSCOPE_API_KEY
        )

    def _search_model(self) -> str:
        """当前 provider 用的搜索模型名。"""
        if self._provider() == "ollama":
            return getattr(config, "OLLAMA_SEARCH_MODEL", "") or getattr(config, "OLLAMA_MODEL", "") or "qwen2.5:7b"
        return getattr(config, "SEARCH_LLM_MODEL", "qwen-max")

    def _run(self, query: str, num_results: Optional[int] = None) -> List[dict]:
        return asyncio.run(self._arun(query, num_results))

    async def _arun(self, query: str, num_results: Optional[int] = None) -> List[dict]:
        n = num_results or getattr(config, "SEARCH_NUM_RESULTS", 5)
        client = self._client()
        provider = self._provider()

        user_msg = {
            "role": "user",
            "content": format_web_search_user_prompt(query=query, num_results=n),
        }

        # 构造请求参数:ollama 不支持 DashScope 专有的 enable_search / extra_body,
        # 也不一定支持 response_format json_schema strict,按 provider 分支处理
        if provider == "ollama":
            # Ollama:不传 enable_search(纯知识回答),response_format 用 json_object
            # (Ollama 多数模型支持 json_object,json_schema strict 支持有限)
            # 限制 num_ctx 防止模型用过大 context(如 256K)导致 prefill 爆慢
            num_ctx = int(getattr(config, "OLLAMA_NUM_CTX", 4096))
            resp = await client.chat.completions.create(
                model=self._search_model(),
                messages=[user_msg],
                temperature=0.1,
                response_format={"type": "json_object"},
                extra_body={"options": {"num_ctx": num_ctx}},
            )
        else:
            # DashScope:真联网搜索 + response_format json_schema strict
            resp = await client.chat.completions.create(
                model=self._search_model(),
                messages=[user_msg],
                temperature=0.1,
                # 用 response_format json_schema(strict)强约束 LLM 输出为
                # {"results": [{url, title, snippet}, ...]},替代纯正则解析
                response_format=_build_search_response_format(),
                extra_body={
                    "enable_search": True,
                    "search_options": {
                        "enable_source": True,
                        "search_result_count": max(n, getattr(config, "SEARCH_RESULT_COUNT", 6)),
                    },
                },
            )
        msg = resp.choices[0].message
        content = msg.content or ""

        # 优先用 response_format 产出的结构化 JSON
        results = self._parse_structured_results(content)
        if results:
            return results[: max(n, 3)]

        # 兜底1:从 DashScope 响应扩展字段中提取服务端原始搜索结果
        results = self._extract_search_info(resp)
        if results:
            return results[: max(n, 3)]

        # 兜底2:正则解析 LLM 文本
        return self._parse_results(content, n)

    def _parse_structured_results(self, content: str) -> List[dict]:
        """解析 response_format json_schema 产出的结构化 JSON。

        正常情况下 content 是 {"results": [{url,title,snippet}, ...]} 格式,
        直接 json.loads + Pydantic 校验。失败返回 [] 走后续兜底。
        """
        if not content:
            return []
        try:
            # response_format 下整段就是合法 JSON,但兼容首尾空白/包裹
            text = content.strip()
            # 容忍偶尔被 ```json 包裹的情况
            if text.startswith("```"):
                text = text.strip("`").lstrip("json").strip()
            data = json.loads(text)
            validated = SearchResultList.model_validate(data)
            return [
                {
                    "url": str(item.url),
                    "title": str(item.title),
                    "snippet": str(item.snippet),
                }
                for item in validated.results
                if item.url or item.title or item.snippet
            ]
        except Exception:
            return []

    def _extract_search_info(self, resp) -> List[dict]:
        """从 DashScope 响应扩展字段中提取结构化搜索结果。"""
        results: List[dict] = []
        candidates = []
        try:
            candidates.append(resp.model_extra)
        except Exception:
            pass
        try:
            candidates.append(resp.choices[0].model_extra)
        except Exception:
            pass
        try:
            candidates.append(resp.choices[0].message.model_extra)
        except Exception:
            pass

        for extra in candidates:
            if not extra or not isinstance(extra, dict):
                continue
            sr = extra.get("search_results") or extra.get("search_info")
            if isinstance(sr, list):
                for it in sr:
                    if isinstance(it, dict) and it.get("url"):
                        results.append({
                            "url": str(it["url"]),
                            "title": str(it.get("title", "")),
                            "snippet": str(it.get("snippet") or it.get("content", "")),
                        })
        return results

    def _parse_results(self, content: str, num_results: int) -> List[dict]:
        """从 Qwen 返回文本中解析出 {url,title,snippet} 列表。"""
        results: List[dict] = []

        m = re.search(r"\[[\s\S]*\]", content)
        if m:
            try:
                arr = json.loads(m.group(0))
                if isinstance(arr, list):
                    for it in arr:
                        if isinstance(it, dict) and it.get("url"):
                            results.append({
                                "url": str(it["url"]),
                                "title": str(it.get("title", "")),
                                "snippet": str(it.get("snippet", "")),
                            })
                        if len(results) >= num_results:
                            break
            except Exception:
                pass

        if len(results) < 3:
            urls = re.findall(r"https?://[^\s\)\]\）\"]+", content)
            seen = {r["url"] for r in results}
            for u in urls:
                if u in seen:
                    continue
                seen.add(u)
                results.append({"url": u, "title": "", "snippet": ""})

        if not results:
            results = [{"url": "", "title": "(无结构化结果)", "snippet": content[:200]}]
        return results[: max(num_results, 3)]


if __name__ == "__main__":
    q = "2024年口罩佩戴政策"
    print(f"查询: {q}")
    for i, r in enumerate(WebSearchTool().invoke({"query": q}), 1):
        print(f"\n[{i}] {r.get('title','')}\n    {r.get('url','')}\n    {r.get('snippet','')[:80]}")
