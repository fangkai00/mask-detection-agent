# -*- coding: utf-8 -*-
"""
FastAPI 口罩检测服务(可选方案)

将 YOLOv8 口罩检测模型封装为独立 HTTP 服务,供多客户端调用。
默认情况下智能体使用进程内 Function Call 直接调用(更快);
需要分布式/多客户端场景时,启动此服务并在 config.yaml 中设 MASK_USE_API: true。

启动方式:
    cd mask_agent
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

接口:
    POST /detect/file    上传图片文件检测 → 返回 JSON 结果
    POST /detect/path    传图片路径检测   → 返回 JSON 结果
    GET  /health         健康检查
    GET  /               服务信息
"""
import logging
import os
import sys
import shutil
import tempfile
from pathlib import Path

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config

logger = logging.getLogger("mask_agent.api")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)
logger.propagate = False

try:
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "FastAPI 未安装。请运行: pip install fastapi uvicorn python-multipart"
    )

from tools.mask_detection import MaskDetectionTool, get_model, _find_best_pt


# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(
    title="口罩检测 API",
    description="YOLOv8 口罩佩戴检测服务,支持上传图片或传路径进行检测",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    """服务启动时预加载模型(避免首次请求慢)。"""
    try:
        model = get_model()
        logger.info("[API] 模型预加载完成: %s", type(model).__name__)
    except Exception as e:
        logger.warning("[API] 模型预加载失败(首次请求时会重试): %s", e)


# ============================================================
# 响应模型
# ============================================================
class DetectionResponse(BaseModel):
    image_path: str
    total_persons: int
    mask_count: int
    no_mask_count: int
    compliance_rate: float
    detections: list
    annotated_image: str
    elapsed_s: float


# ============================================================
# 接口
# ============================================================
@app.get("/")
def root():
    """服务信息。"""
    weights = _find_best_pt()
    return {
        "service": "口罩检测 API",
        "model": "YOLOv8",
        "weights": weights or "(未找到)",
        "class_names": getattr(config, "MASK_CLASS_NAMES", ["mask", "no-mask"]),
        "endpoints": {
            "POST /detect/file": "上传图片文件检测(multipart/form-data)",
            "POST /detect/path": "传图片路径检测(application/json)",
            "GET /health": "健康检查",
        },
    }


@app.get("/health")
def health():
    """健康检查。"""
    weights = _find_best_pt()
    return {
        "status": "healthy" if weights else "no_weights",
        "weights_found": bool(weights),
    }


@app.post("/detect/file", response_model=DetectionResponse)
async def detect_file(file: UploadFile = File(...)):
    """上传图片文件进行口罩检测。

    请求体: multipart/form-data,字段 file=图片文件
    返回: DetectionResponse JSON
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="未提供文件")

    # 保存到临时文件
    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        tool = MaskDetectionTool()
        import time
        start = time.time()
        result = tool._run(tmp_path)
        elapsed = round(time.time() - start, 3)

        # DetectionResponse 已包含 image_path,这里替换为原始文件名
        result["image_path"] = file.filename
        result["elapsed_s"] = elapsed
        return DetectionResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {type(e).__name__}: {e}")
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/detect/path", response_model=DetectionResponse)
async def detect_path(image_path: str = Form(...)):
    """传入图片路径进行口罩检测。

    请求体: application/x-www-form-urlencoded,字段 image_path=路径
    返回: DetectionResponse JSON
    """
    try:
        tool = MaskDetectionTool()
        import time
        start = time.time()
        result = tool._run(image_path)
        elapsed = round(time.time() - start, 3)
        result["elapsed_s"] = elapsed
        return DetectionResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {type(e).__name__}: {e}")


if __name__ == "__main__":
    import uvicorn

    host = getattr(config, "MASK_API_HOST", "0.0.0.0")
    port = int(getattr(config, "MASK_API_PORT", 8000))
    logger.info("[API] 启动服务: http://%s:%s", host, port)
    uvicorn.run("api_server:app", host=host, port=port, reload=False)
