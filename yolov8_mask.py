# -*- coding: utf-8 -*-
"""
YOLOv8 口罩检测 - 训练 / 评估 / 可视化
数据集: MaskDataSet (2 类: mask / no-mask)
参考: reference/objectDetection.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无窗口环境也能保存图像
import matplotlib.pyplot as plt
import cv2
import torch
from ultralytics import YOLO

# ============ 配置区 ============
ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "MaskDataSet" / "data.yaml"
# 预训练权重：用本地已下载的文件，避免每次联网下载
# yolov8n.pt = nano(快、轻量)  yolov8m.pt = medium(精度更高、较慢)
WEIGHTS = str(ROOT / "yolov8n.pt")
EPOCHS = 300
BATCH = 16
IMG_SIZE = 640
# 自动按 CUDA 可用性选择；多卡时可用 "0,1,2" 指定具体卡
DEVICE = "0" if torch.cuda.is_available() else "cpu"
WORKERS = 0                     # Windows 下建议 0，避免多进程报错
PROJECT = ROOT / "runs" / "mask_yolov8"

# 学习率相关（ultralytics 默认值，可按需修改）
OPTIMIZER = "Adam"   # auto / SGD / Adam / AdamW；微调预训练权重用 SGD 或 AdamW 常见
LR0 = 0.001           # 初始学习率；微调时常用 0.001~0.01
LRF = 0.0001           # 最终学习率系数：final_lr = lr0 * lrf（余弦退火终点）
MOMENTUM = 0.937     # SGD 动量（Adam/AdamW 不生效）
WEIGHT_DECAY = 0.0005  # L2 正则
WARMUP_EPOCHS = 3.0   # 预热 epoch 数（前几轮学习率从极小线性升到 lr0）
WARMUP_MOMENTUM = 0.8  # 预热起始动量

print(f"[INFO] torch={torch.__version__} cuda_available={torch.cuda.is_available()} device={DEVICE}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")


# ============ 1. 训练 ============
def train():
    model = YOLO(WEIGHTS)
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMG_SIZE,
        device=DEVICE,
        workers=WORKERS,
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_momentum=WARMUP_MOMENTUM,
        project=str(PROJECT),
        name="train",
        plots=True,  # 训练曲线 / 混淆矩阵 / PR 曲线自动保存
    )
    return model


# ============ 2. 评估 + 可视化 ============
def evaluate(model):
    metrics = model.val(
        project=str(PROJECT),
        name="val",
        plots=True,  # 生成 PR_curve / P_curve / R_curve / F1_curve / confusion_matrix
    )
    print("\n========== 评估指标 ==========")
    print(f"Precision (mP) : {metrics.box.mp:.4f}")
    print(f"Recall    (mR) : {metrics.box.mr:.4f}")
    print(f"mAP@0.5       : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95  : {metrics.box.map:.4f}")
    return metrics


# ============ 3. 测试集预测可视化 ============
def predict():
    test_dir = ROOT / "MaskDataSet" / "test" / "images"
    # 自动找最新的 train* 子目录（ultralytics 遇到重名会自动加 -2/-3 后缀）
    train_dirs = sorted(PROJECT.glob("train*"))
    best_pt = None
    for d in reversed(train_dirs):  # 从最新开始找
        candidate = d / "weights" / "best.pt"
        if candidate.exists():
            best_pt = candidate
            break
    if best_pt is None:
        print(f"[WARN] 未找到 best.pt（训练可能未完成或被中断），跳过预测步骤")
        print(f"       已扫描: {[str(d) for d in train_dirs]}")
        return
    print(f"[INFO] 使用权重: {best_pt}")
    model = YOLO(str(best_pt))
    model.predict(
        source=str(test_dir),
        save=True,
        project=str(PROJECT),
        name="predict",
        conf=0.25,
    )


# ============ 4. 汇总评估可视化 ============
def make_summary():
    """把训练/评估产生的关键可视化图汇总成一张大图。"""
    targets = [
        ("训练曲线",         PROJECT / "train" / "results.png"),
        ("混淆矩阵 (训练)",   PROJECT / "train" / "confusion_matrix.png"),
        ("混淆矩阵 (归一化)", PROJECT / "train" / "confusion_matrix_normalized.png"),
        ("PR 曲线",          PROJECT / "val"   / "PR_curve.png"),
        ("F1 曲线",          PROJECT / "val"   / "F1_curve.png"),
        ("P 曲线",           PROJECT / "val"   / "P_curve.png"),
        ("R 曲线",           PROJECT / "val"   / "R_curve.png"),
        ("混淆矩阵 (评估)",   PROJECT / "val"   / "confusion_matrix.png"),
    ]
    items = [(t, p) for t, p in targets if p.exists()]
    if not items:
        print("[WARN] 未找到任何可视化图，跳过汇总。")
        return

    cols = 2
    rows = (len(items) + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, (title, p) in zip(axes, items):
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[len(items):]:
        ax.axis("off")
    plt.tight_layout()
    out = PROJECT / "evaluation_summary.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n[INFO] 评估可视化汇总图: {out}")


# ============ 主流程 ============
def main():
    PROJECT.mkdir(parents=True, exist_ok=True)
    model = train()
    evaluate(model)
    predict()
    make_summary()
    print(f"\n[INFO] 所有结果保存在: {PROJECT}")


if __name__ == "__main__":
    main()
