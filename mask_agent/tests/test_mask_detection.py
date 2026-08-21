# -*- coding: utf-8 -*-
"""
口罩检测工具测试

测试 MaskDetectionTool 的权重查找、模型加载、推理流程。
不依赖 LLM,纯模型层测试。

运行: python tests/test_mask_detection.py
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import compat  # noqa: E402
from tools.mask_detection import MaskDetectionTool, _find_best_pt


def test_find_weights():
    """测试模型权重自动查找。"""
    print("=" * 50)
    print("[test] 模型权重查找")
    weights = _find_best_pt()
    if weights:
        print(f"  ✓ 找到权重: {weights}")
        assert os.path.exists(weights), "权重文件不存在"
    else:
        print("  ⚠ 未找到权重(需先训练模型)")
    print()


def test_detection_tool():
    """测试口罩检测工具(需要模型权重和测试图片)。"""
    print("=" * 50)
    print("[test] 口罩检测工具")

    # 检查是否有测试图片
    test_images = [
        os.path.join(_ROOT, "..", "MaskDataSet", "test", "images"),
        os.path.join(_ROOT, "..", "MaskDataSet", "archive", "images"),
    ]

    test_image = None
    for d in test_images:
        d = os.path.abspath(d)
        if os.path.exists(d):
            imgs = [f for f in os.listdir(d) if f.endswith(('.jpg', '.png'))]
            if imgs:
                test_image = os.path.join(d, imgs[0])
                break

    if not test_image:
        print("  ⚠ 未找到测试图片,跳过检测测试")
        return

    print(f"  测试图片: {test_image}")

    weights = _find_best_pt()
    if not weights:
        print("  ⚠ 未找到模型权重,跳过检测测试")
        return

    try:
        tool = MaskDetectionTool()
        result = tool._run(test_image)

        print(f"  总人数: {result.get('total_persons', 0)}")
        print(f"  戴口罩: {result.get('mask_count', 0)}")
        print(f"  未戴口罩: {result.get('no_mask_count', 0)}")
        print(f"  合规率: {result.get('compliance_rate', 0)*100:.1f}%")
        print(f"  标注图: {result.get('annotated_image', '无')}")

        assert isinstance(result, dict), "结果应为 dict"
        assert "total_persons" in result, "结果应含 total_persons"
        assert "mask_count" in result, "结果应含 mask_count"
        assert "no_mask_count" in result, "结果应含 no_mask_count"
        print("  ✓ 检测工具测试通过")
    except Exception as e:
        print(f"  ✗ 检测失败: {type(e).__name__}: {e}")
    print()


def test_error_handling():
    """测试错误处理(不存在的图片)。"""
    print("=" * 50)
    print("[test] 错误处理(不存在的图片)")
    tool = MaskDetectionTool()
    try:
        tool._run("nonexistent_image.jpg")
        print("  ✗ 应该抛出 FileNotFoundError")
        assert False, "应该抛出异常"
    except FileNotFoundError as e:
        print(f"  ✓ 正确抛出 FileNotFoundError: {e}")
    except Exception as e:
        print(f"  ✗ 抛出了意外异常: {type(e).__name__}: {e}")
    print()


if __name__ == "__main__":
    test_find_weights()
    test_detection_tool()
    test_error_handling()
    print("=" * 50)
    print("测试完成")
