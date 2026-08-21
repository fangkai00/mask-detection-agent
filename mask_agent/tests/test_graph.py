# -*- coding: utf-8 -*-
"""
LangGraph 图结构测试

测试图的构建、编译、初始状态、路由逻辑(不依赖 LLM 调用)。

运行: python tests/test_graph.py
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import compat  # noqa: E402


def test_graph_build():
    """测试图构建与编译。"""
    print("=" * 50)
    print("[test] 图构建与编译")
    from agent.graph import build_graph
    graph = build_graph()
    assert graph is not None, "图编译失败"
    print(f"  ✓ 图编译成功: {type(graph).__name__}")
    print()


def test_initial_state():
    """测试初始状态构建。"""
    print("=" * 50)
    print("[test] 初始状态构建")
    from agent.graph import build_initial_state
    state = build_initial_state("测试问题", "test.jpg")
    assert state["user_query"] == "测试问题", "user_query 不正确"
    assert state["image_path"] == "test.jpg", "image_path 不正确"
    assert state["detection_result"] == {}, "detection_result 初始应为空 dict"
    assert state["search_results"] == [], "search_results 初始应为空 list"
    assert state["steps"] == 0, "steps 初始应为 0"
    print("  ✓ 初始状态正确")
    print()


def test_route_parsing():
    """测试路由 JSON 解析。"""
    print("=" * 50)
    print("[test] 路由 JSON 解析")
    from agent.graph import _parse_route, RouteDecision

    # 正常解析
    d = _parse_route('{"next": "mask_detect", "reason": "用户要求检测"}')
    assert d.next == "mask_detect", f"期望 mask_detect, 实际 {d.next}"
    print(f"  ✓ mask_detect 解析正确: {d.reason}")

    d = _parse_route('{"next": "search", "reason": "知识查询"}')
    assert d.next == "search", f"期望 search, 实际 {d.next}"
    print(f"  ✓ search 解析正确: {d.reason}")

    d = _parse_route('{"next": "finish", "reason": "信息充分"}')
    assert d.next == "finish", f"期望 finish, 实际 {d.next}"
    print(f"  ✓ finish 解析正确: {d.reason}")

    # 非法值回退
    d = _parse_route('{"next": "invalid", "reason": "test"}')
    assert d.next == "finish", f"非法值应回退 finish, 实际 {d.next}"
    print(f"  ✓ 非法路由回退 finish")

    # 解析失败回退
    d = _parse_route("not a json")
    assert d.next == "finish", f"解析失败应回退 finish, 实际 {d.next}"
    print(f"  ✓ 解析失败回退 finish")
    print()


def test_summarize_gathered():
    """测试信息汇总。"""
    print("=" * 50)
    print("[test] 信息汇总")
    from agent.graph import _summarize_gathered

    # 空状态
    state = {"messages": [], "detection_result": {}, "search_results": []}
    result = _summarize_gathered(state)
    assert result == "(暂无)", f"空状态应为 '(暂无)', 实际: {result}"
    print("  ✓ 空状态汇总正确")

    # 含检测结果
    state = {
        "messages": [],
        "detection_result": {
            "total_persons": 3,
            "mask_count": 2,
            "no_mask_count": 1,
            "compliance_rate": 0.667,
            "detections": [{"cls_name": "mask", "confidence": 0.95}],
            "annotated_image": "data/results/test.jpg",
        },
        "search_results": [],
    }
    result = _summarize_gathered(state)
    assert "[MaskDetect]" in result, "应含 [MaskDetect] 标记"
    assert "3" in result, "应含总人数"
    assert "标注图" in result, "应含标注图信息"
    print("  ✓ 检测结果汇总正确")

    # 含搜索结果
    state = {
        "messages": [],
        "detection_result": {},
        "search_results": [{"title": "口罩政策", "url": "https://example.com", "snippet": "最新规定"}],
    }
    result = _summarize_gathered(state)
    assert "[Search]" in result, "应含 [Search] 标记"
    assert "口罩政策" in result, "应含搜索标题"
    print("  ✓ 搜索结果汇总正确")
    print()


if __name__ == "__main__":
    test_graph_build()
    test_initial_state()
    test_route_parsing()
    test_summarize_gathered()
    print("=" * 50)
    print("图结构测试全部通过 ✓")
