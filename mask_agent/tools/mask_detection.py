# -*- coding: utf-8 -*-
"""
MaskDetectionTool —— YOLOv8 口罩检测工具(核心工具)

调用训练好的 YOLOv8 模型对输入图片进行口罩佩戴检测。
检测类别(与 MaskDataSet/data.yaml 一致):
  - 0: mask     (戴口罩)
  - 1: no-mask  (未戴口罩)

流程:
1. 加载模型权重(best.pt)——自动搜索最新训练目录,或用 config 指定路径
2. 对输入图片执行推理(model.predict)
3. 解析结果:逐人提取类别、置信度、边界框坐标
4. 统计:总人数、戴口罩人数、未戴口罩人数、合规率
5. 保存标注后的图片(可选)供用户查看

接口:input image_path(str) → output dict(结构化检测结果)
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# 项目根入 sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

import config

logger = logging.getLogger("mask_agent.detection")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)
logger.propagate = False


# ============================================================
# 模型权重自动查找
# ============================================================
def _find_best_pt() -> Optional[str]:
    """自动搜索最新训练目录下的 best.pt。

    搜索顺序:
    1. config.MASK_MODEL_WEIGHTS(若配置且文件存在)
    2. 项目根下 runs/mask_yolov8_v2/train*/weights/best.pt(从新到旧)
    3. 项目根下 runs/mask_yolov8/train*/weights/best.pt(旧版目录)
    """
    # 1. 配置指定的路径
    configured = getattr(config, "MASK_MODEL_WEIGHTS", "")
    if configured:
        # config 路径可能是相对项目根的,已由 config/__init__.py 拼接
        # 但 MASK_MODEL_WEIGHTS 不以 _DIR/_FILE 结尾,需手动处理
        p = Path(configured)
        if not p.is_absolute():
            p = Path(config.PROJECT_ROOT) / p
        if p.exists():
            return str(p)
        logger.info("[mask_detection] 配置的权重路径不存在: %s,开始自动搜索...", p)

    # 2. 自动搜索 runs/ 下的训练目录
    runs_dir = Path(config.PROJECT_ROOT).parent / "runs"
    if not runs_dir.exists():
        runs_dir = Path(config.PROJECT_ROOT) / "runs"

    search_dirs = []
    for subdir in ["mask_yolov8_v2", "mask_yolov8"]:
        d = runs_dir / subdir
        if d.exists():
            search_dirs.append(d)

    for base_dir in search_dirs:
        # glob 匹配 train, train-2, train-3 等(从新到旧)
        train_dirs = sorted(base_dir.glob("train*"), reverse=True)
        for td in train_dirs:
            best_pt = td / "weights" / "best.pt"
            if best_pt.exists():
                logger.info("[mask_detection] 自动找到权重: %s", best_pt)
                return str(best_pt)

    return None


# ============================================================
# 模型单例(懒加载,避免重复加载)
# ============================================================
_model = None


def get_model():
    """获取 YOLO 模型单例(懒加载)。

    若找不到权重文件,抛出 FileNotFoundError 让上层(safe_tool_call)捕获。
    """
    global _model
    if _model is not None:
        return _model

    weights = _find_best_pt()
    if weights is None:
        raise FileNotFoundError(
            "未找到 YOLOv8 口罩检测模型权重(best.pt)。"
            "请先训练模型(python yolov8_mask_v2/yolov8_mask.py),"
            "或在 config.yaml 中设置 MASK_MODEL_WEIGHTS 路径。"
        )

    from ultralytics import YOLO
    _model = YOLO(weights)
    logger.info("[mask_detection] 模型加载完成: %s", weights)
    return _model


# ============================================================
# 工具输入/输出模型
# ============================================================
class MaskDetectionInput(BaseModel):
    image_path: str = Field(description="待检测图片的路径(绝对或相对项目根)")


class DetectionItem(BaseModel):
    """单个检测结果。"""
    cls_name: str = Field(description="类别名: mask(戴口罩) / no-mask(未戴口罩)")
    confidence: float = Field(description="置信度(0~1)")
    box: List[float] = Field(description="边界框坐标 [x1, y1, x2, y2]")


# ============================================================
# MaskDetectionTool
# ============================================================
class MaskDetectionTool(BaseTool):
    """YOLOv8 口罩检测工具:对输入图片执行口罩佩戴检测。

    输入:图片路径
    输出:结构化检测结果(总人数、戴/未戴口罩人数、合规率、逐人详情、标注图路径)
    """

    name: str = "mask_detect"
    description: str = (
        "使用 YOLOv8 模型对输入图片进行口罩佩戴检测。"
        "返回戴口罩人数、未戴口罩人数、合规率及逐人检测结果。"
        "适用于用户上传图片后要求检测口罩佩戴情况的场景。"
    )
    args_schema: type = MaskDetectionInput

    def _run(self, image_path: str) -> dict:
        """同步执行口罩检测。

        Args:
            image_path: 待检测图片路径

        Returns:
            dict: 结构化检测结果
        """
        # 1. 校验图片路径
        img_path = Path(image_path)
        if not img_path.is_absolute():
            img_path = Path(config.PROJECT_ROOT) / image_path
        if not img_path.exists():
            raise FileNotFoundError(f"图片不存在: {image_path}(解析后: {img_path})")

        # 2. 加载模型
        model = get_model()

        # 3. 执行推理
        conf = getattr(config, "MASK_CONF_THRESHOLD", 0.25)
        iou = getattr(config, "MASK_IOU_THRESHOLD", 0.7)
        imgsz = getattr(config, "MASK_IMG_SIZE", 640)
        device = getattr(config, "MASK_DEVICE", "auto")
        save_annotated = getattr(config, "MASK_SAVE_ANNOTATED", True)
        class_names = getattr(config, "MASK_CLASS_NAMES", ["mask", "no-mask"])

        # ultralytics 8.4.x 不支持 device="auto",需手动解析
        if device == "auto":
            try:
                import torch
                device = "0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        results = model.predict(
            source=str(img_path),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            save=save_annotated,
            verbose=False,
        )

        # 4. 解析结果
        detections: List[dict] = []
        result = results[0]  # 单张图,取第一个结果

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf_val = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]
                cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                detections.append({
                    "cls_name": cls_name,
                    "confidence": round(conf_val, 4),
                    "box": [round(v, 1) for v in xyxy],
                })

        # 5. 统计
        mask_count = sum(1 for d in detections if d["cls_name"] == "mask")
        no_mask_count = sum(1 for d in detections if d["cls_name"] == "no-mask")
        total = len(detections)
        compliance_rate = round(mask_count / total, 4) if total > 0 else 0.0

        # 6. 标注图路径
        annotated_path = ""
        if save_annotated and result.save_dir:
            # ultralytics 保存到 save_dir/image_name.jpg
            save_dir = Path(result.save_dir)
            annotated_files = list(save_dir.glob("*.jpg")) + list(save_dir.glob("*.png"))
            if annotated_files:
                # 复制到 data/results/ 便于统一管理
                result_dir = Path(getattr(config, "RESULT_IMAGE_DIR", "data/results"))
                if not result_dir.is_absolute():
                    result_dir = Path(config.PROJECT_ROOT) / result_dir
                result_dir.mkdir(parents=True, exist_ok=True)
                import shutil
                dest = result_dir / f"detected_{img_path.name}"
                shutil.copy2(annotated_files[0], dest)
                annotated_path = str(dest)

        result_dict = {
            "image_path": str(img_path),
            "total_persons": total,
            "mask_count": mask_count,
            "no_mask_count": no_mask_count,
            "compliance_rate": compliance_rate,
            "detections": detections,
            "annotated_image": annotated_path,
        }

        logger.info(
            "[mask_detection] 检测完成: 总%d人, 戴口罩%d, 未戴%d, 合规率%.1f%%",
            total, mask_count, no_mask_count, compliance_rate * 100,
        )
        return result_dict


# ============================================================
# MaskDetectionAPITool —— 通过 HTTP API 调用检测服务(可选方案)
# ============================================================
# 适用场景: 检测服务独立部署(api_server.py),智能体通过 HTTP 调用
# 启用方式: config.yaml 中设 MASK_USE_API: true + MASK_API_BASE_URL
# ============================================================

class MaskDetectionAPITool(BaseTool):
    """通过 HTTP API 调用口罩检测服务。

    需先启动 api_server.py:
        uvicorn api_server:app --port 8000

    然后在 config.yaml 中设置:
        MASK_USE_API: true
        MASK_API_BASE_URL: "http://localhost:8000"
    """

    name: str = "mask_detect"
    description: str = (
        "通过 HTTP API 调用口罩检测服务,返回检测结果。"
        "需要 api_server.py 服务运行中。"
    )
    args_schema: type = MaskDetectionInput

    def _run(self, image_path: str) -> dict:
        import requests

        base_url = getattr(config, "MASK_API_BASE_URL", "http://localhost:8000")
        url = f"{base_url}/detect/path"

        resp = requests.post(url, data={"image_path": image_path}, timeout=30)

        if resp.status_code == 404:
            raise FileNotFoundError(f"图片不存在: {image_path}")
        if resp.status_code != 200:
            raise RuntimeError(
                f"API 调用失败(HTTP {resp.status_code}): {resp.text[:200]}"
            )

        return resp.json()

    async def _arun(self, image_path: str) -> dict:
        """异步调用版本。"""
        import aiohttp

        base_url = getattr(config, "MASK_API_BASE_URL", "http://localhost:8000")
        url = f"{base_url}/detect/path"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, data={"image_path": image_path}, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 404:
                    raise FileNotFoundError(f"图片不存在: {image_path}")
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"API 调用失败(HTTP {resp.status}): {text[:200]}")
                return await resp.json()


# ============================================================
# 工厂函数:根据配置选择进程内 / API 调用
# ============================================================
def get_mask_detection_tool() -> BaseTool:
    """根据 config.MASK_USE_API 决定用哪种检测工具。

    - MASK_USE_API=false(默认): 返回 MaskDetectionTool(进程内单例,最快)
    - MASK_USE_API=true:        返回 MaskDetectionAPITool(HTTP 调用,需独立服务)
    """
    use_api = getattr(config, "MASK_USE_API", False)
    if use_api:
        logger.info("[mask_detection] 使用 HTTP API 模式调用检测服务")
        return MaskDetectionAPITool()
    return MaskDetectionTool()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python mask_detection.py <图片路径>")
        sys.exit(1)
    tool = get_mask_detection_tool()
    result = tool._run(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False, indent=2))
