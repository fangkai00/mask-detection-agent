# -*- coding: utf-8 -*-
"""
YOLOv8 口罩检测 - 训练 / 评估 / 可视化 (v2)
数据集: MaskDataSet (2 类: mask / no-mask)
参考: reference/objectDetection.py

与根目录 yolov8_mask.py 的区别：
- PROJECT 输出到 runs/mask_yolov8_v2/，不覆盖上一次的 runs/mask_yolov8/
- 补全学习率退火策略参数 (warmup_lr / warmup_bias_lr)
- 新增 plot_lr_schedule() 可视化学习率退火曲线
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无窗口环境也能保存图像
import matplotlib.pyplot as plt
# 配置中文字体，避免中文显示为方块（Windows 用 SimHei / Microsoft YaHei）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示
import cv2
import torch
from ultralytics import YOLO

# ============ 配置区 ============
# 本脚本位于 yolov8_mask_v2/ 子文件夹，ROOT 上一级才是项目根目录
ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "MaskDataSet" / "data.yaml"
# 预训练权重：用本地已下载的文件，避免每次联网下载
# yolov8n.pt = nano(快、轻量)  yolov8m.pt = medium(精度更高、较慢)
WEIGHTS = str(ROOT / "yolov8m.pt")
EPOCHS = 200                 # 958 张小数据 + 预训练权重，100 epoch 足够；配 patience 早停
BATCH = 32                   # A6000 48GB 显存充裕；如爆显存改 16
IMG_SIZE = 640
PATIENCE = 50                # 20 epoch 无提升则早停，避免过拟合浪费时间
AMP = True                   # 自动混合精度（A6000 加速明显）；首次训练会下载 yolo26n.pt 做 AMP 检查
# 自动按 CUDA 可用性选择；多卡时可用 "0,1,2" 指定具体卡
DEVICE = "1" if torch.cuda.is_available() else "cpu"
WORKERS = 0                  # Windows 下建议 0，避免多进程报错
# 输出到独立的 v2 目录，不影响旧版 runs/mask_yolov8/ 的结果
PROJECT = ROOT / "runs" / "mask_yolov8_v2"
# 各步骤子目录命名（对比实验时改这里：train / train-2 / train-lr0001 等）
NAME_TRAIN = "train-5"        # 训练输出目录名
NAME_VAL = "val-7"            # 评估输出目录名
NAME_PREDICT = "predict-7"    # 预测输出目录名

# ============ 学习率退火策略 ============
# 学习率曲线（warmup 线性升 → 余弦退火降）：
#   lr
#   lr0 ─────╮                    ╭── lr0*lrf (退火终点)
#            │                   /
#   warmup_lr╲                  /
#             ╲________________/
#           0 ├─warmup─┤├── 余弦退火 ──┤
#                    WARMUP_EPOCHS    EPOCHS
OPTIMIZER = "AdamW"          # auto / SGD / Adam / AdamW；微调预训练权重用 SGD 或 AdamW 常见
LR0 = 0.0005                  # 初始学习率（warmup 终点 & 退火起点）；AdamW 微调常用 0.001~0.003
LRF = 0.01                   # 退火终点比例：final_lr = lr0 * lrf = 1e-5（太小则后期不更新）
MOMENTUM = 0.937             # SGD 动量（Adam/AdamW 不生效）
WEIGHT_DECAY = 0.0005        # L2 正则
WARMUP_EPOCHS = 3.0          # 预热 epoch 数（前几轮学习率从 warmup_lr 线性升到 lr0）
WARMUP_MOMENTUM = 0.8        # 预热起始动量
WARMUP_BIAS_LR = 0.1         # warmup 阶段 bias 参数的学习率（bias 收敛快，可用大值）
# 注：ultralytics 8.4.90 不支持 warmup_lr 参数（起点 lr 由内部自动决定），仅新版可用

# 数据增强（小数据集尤其重要，提升泛化）
HSV_H = 0.015                # 色调增强
HSV_S = 0.7                  # 饱和度增强
HSV_V = 0.4                  # 明度增强
DEGREES = 0.0               # 旋转 ±10°
TRANSLATE = 0.1              # 平移 ±10%
SCALE = 0.5                  # 缩放 ±50%
FLIPLR = 0.5                 # 水平翻转概率
MOSAIC = 1.0                 # Mosaic 增强（4 图拼接，小数据集很有效）
MIXUP = 0.0                  # MixUp 增强（轻度混入，提升泛化）

print(f"[INFO] torch={torch.__version__} cuda_available={torch.cuda.is_available()} device={DEVICE}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
print(f"[INFO] 输出目录: {PROJECT}（不影响旧版 runs/mask_yolov8/）")


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
        patience=PATIENCE,
        amp=AMP,
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_momentum=WARMUP_MOMENTUM,
        warmup_bias_lr=WARMUP_BIAS_LR,
        hsv_h=HSV_H, hsv_s=HSV_S, hsv_v=HSV_V,
        degrees=DEGREES, translate=TRANSLATE, scale=SCALE,
        fliplr=FLIPLR, mosaic=MOSAIC, mixup=MIXUP,
        project=str(PROJECT),
        name=NAME_TRAIN,
        plots=True,  # 训练曲线 / 混淆矩阵 / PR 曲线自动保存
        multi_scale=True,
    )
    return model


# ============ 2. 评估 + 可视化 ============
def evaluate(model):
    metrics = model.val(
        project=str(PROJECT),
        name=NAME_VAL,
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
        name=NAME_PREDICT,
        conf=0.25,                 # 置信度阈值：只保留 conf>0.25 的框（提高=更少假阳性，降低=更多召回）
        iou=0.7,                   # NMS IOU 阈值：两框重叠>Iou 算同一目标去重（密集人群可降到 0.5）
        # augment=True,
    )


# ============ 4. 学习率退火曲线可视化 ============
def plot_lr_schedule():
    """从 results.csv 读取每 epoch 实际学习率，画出 warmup + 余弦退火曲线。
    用于验证 LR0/LRF/WARMUP 配置是否符合预期。"""
    csv_path = PROJECT / "train" / "results.csv"
    if not csv_path.exists():
        print(f"[WARN] 未找到 {csv_path}，跳过学习率曲线绘制")
        return

    import csv
    epochs, lrs = [], []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # ultralytics 不同版本字段名可能为 'lr' 或 '                  lr'
            lr_val = None
            for k, v in row.items():
                if k.strip() == "lr":
                    lr_val = v
                    break
            if lr_val is None:
                continue
            try:
                epochs.append(int(row.get("epoch", len(epochs) + 1)))
                lrs.append(float(lr_val))
            except (ValueError, TypeError):
                continue
    if not lrs:
        print("[WARN] results.csv 中未找到 lr 列，跳过")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, lrs, color="#1f77b4", linewidth=2, marker="o", markersize=4)

    # 标注关键点
    max_lr_idx = lrs.index(max(lrs))
    ax.axhline(LR0, color="green", linestyle="--", alpha=0.6, label=f"lr0={LR0}")
    ax.axhline(LR0 * LRF, color="red", linestyle="--", alpha=0.6, label=f"final lr=lr0*lrf={LR0*LRF}")
    ax.axvline(WARMUP_EPOCHS, color="orange", linestyle=":", alpha=0.6, label=f"warmup end @ep{WARMUP_EPOCHS}")

    ax.scatter([epochs[max_lr_idx]], [lrs[max_lr_idx]], color="green", s=100, zorder=5,
               label=f"peak lr={lrs[max_lr_idx]:.2e}")
    ax.scatter([epochs[-1]], [lrs[-1]], color="red", s=100, zorder=5,
               label=f"last lr={lrs[-1]:.2e}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title(f"Learning Rate Schedule (warmup → cosine annealing)\n"
                 f"LR0={LR0}, LRF={LRF}, warmup={WARMUP_EPOCHS}ep, total={EPOCHS}ep")
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = PROJECT / "lr_schedule.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[INFO] 学习率退火曲线: {out}")
    print(f"       peak lr={max(lrs):.2e} (应为 ~{LR0})")
    print(f"       last lr={lrs[-1]:.2e} (应为 ~{LR0*LRF})")


# ============ 5. 汇总评估可视化 ============
def make_summary():
    """把训练/评估产生的关键可视化图汇总成一张大图。"""
    targets = [
        ("训练曲线",         PROJECT / "train" / "results.png"),
        ("学习率退火曲线",    PROJECT / "lr_schedule.png"),
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
    import argparse
    parser = argparse.ArgumentParser(description="YOLOv8 口罩检测 - 训练/评估/预测/可视化")
    parser.add_argument(
        "--only",
        choices=["train", "val", "predict", "summary", "lr"],
        help="只跑指定步骤（不指定则跑全流程）："
             "train=训练, val=评估, predict=预测, summary=汇总图, lr=学习率曲线",
    )
    args = parser.parse_args()

    PROJECT.mkdir(parents=True, exist_ok=True)

    if args.only is None:
        # 默认全流程
        model = train()
        plot_lr_schedule()
        evaluate(model)
        predict()
        make_summary()
    else:
        # 单步执行
        if args.only == "train":
            model = train()
            plot_lr_schedule()
        elif args.only == "val":
            # 评估需要从最新 train* 加载 best.pt
            from ultralytics import YOLO as _YOLO
            train_dirs = sorted(PROJECT.glob("train*"))
            best_pt = None
            for d in reversed(train_dirs):
                cand = d / "weights" / "best.pt"
                if cand.exists():
                    best_pt = cand
                    break
            if best_pt is None:
                print("[ERROR] 未找到 best.pt，请先跑训练：python yolov8_mask.py")
                return
            print(f"[INFO] 加载权重: {best_pt}")
            evaluate(_YOLO(str(best_pt)))
        elif args.only == "predict":
            predict()
        elif args.only == "summary":
            make_summary()
        elif args.only == "lr":
            plot_lr_schedule()

    print(f"\n[INFO] 所有结果保存在: {PROJECT}")


if __name__ == "__main__":
    main()
