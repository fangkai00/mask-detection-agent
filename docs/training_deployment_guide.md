# 训练部署指南

本指南详细说明 YOLOv8 口罩检测模型的训练、评估、推理与智能体部署流程。

---

## 目录

- [1. 环境准备](#1-环境准备)
- [2. 数据集说明](#2-数据集说明)
- [3. 模型训练](#3-模型训练)
- [4. 模型评估](#4-模型评估)
- [5. 推理与预测](#5-推理与预测)
- [6. 智能体部署](#6-智能体部署)
- [7. 常见问题](#7-常见问题)

---

## 1. 环境准备

### 1.1 硬件要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| GPU | 任意 CUDA 11.7+ 显卡（4GB+） | RTX 3090 / A6000（24GB+） |
| CPU | 4 核 | 8 核+ |
| 内存 | 8 GB | 16 GB+ |
| 硬盘 | 5 GB（含数据集） | 20 GB（含多版本权重） |

> CPU 也可训练，但速度极慢（单 epoch 约 10–30 分钟）。

### 1.2 软件依赖

```bash
# 1. 创建 conda 环境
conda create -n mask-agent python=3.10 -y
conda activate mask-agent

# 2. 安装 PyTorch（GPU 版，按 CUDA 版本选择）
# 参考 https://pytorch.org/get-started/locally/
# 例：CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 安装项目依赖（国内推荐清华源）
cd mask-detection-agent
pip install -r mask_agent/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 1.3 验证 CUDA

```bash
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

预期输出：
```
torch=2.x.x, cuda=True, gpu=NVIDIA GeForce RTX ...
```

---

## 2. 数据集说明

### 2.1 数据集结构

```
MaskDataSet/
├── data.yaml                 # YOLOv8 训练配置
├── archive/                  # 原始数据（Pascal VOC XML 标注）
│   ├── images/               #   853 张 PNG 图片
│   └── annotations/          #   853 个 XML 标注文件
├── train/                    # 训练集（YOLO 格式，约 80%）
│   ├── images/
│   └── labels/
├── valid/                    # 验证集（约 10%）
│   ├── images/
│   └── labels/
└── test/                     # 测试集（约 10%）
    ├── images/
    └── labels/
```

### 2.2 类别定义

| ID | 类别名 | 含义 |
|----|--------|------|
| 0 | `mask` | 戴口罩 |
| 1 | `no-mask` | 未戴口罩 |

`data.yaml` 内容：
```yaml
train: train/images
val: valid/images
test: test/images
nc: 2
names: ['mask', 'no-mask']
```

### 2.3 重新切分数据集（可选）

若需调整 train/valid/test 比例：

```bash
python resplit_dataset.py
```

修改 `resplit_dataset.py` 中的比例参数后执行。

---

## 3. 模型训练

### 3.1 准备预训练权重

YOLOv8 微调需要预训练权重（如 `yolov8n.pt`、`yolov8m.pt`），放在项目根目录：

```bash
# 方式1：从 ultralytics 自动下载（首次训练会自动下载到 cwd）
# 方式2：手动下载
# https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt
```

权重规模选择：
- `yolov8n.pt`（nano，6MB）：速度快，精度低，适合 CPU 或边缘设备
- `yolov8m.pt`（medium，50MB）：精度高，推荐 GPU 训练
- `yolov8l.pt`（large，84MB）：精度更高，训练慢，需大显存

### 3.2 训练脚本（推荐 v2）

项目提供两个训练脚本：
- `yolov8_mask.py`（v1，旧版本）
- `yolov8_mask_v2/yolov8_mask_v2.py`（v2，**推荐**）

v2 改进点：
- 输出到独立的 `runs/mask_yolov8_v2/`，不覆盖旧版结果
- 补全学习率退火策略参数
- 新增 `plot_lr_schedule()` 可视化学习率曲线

### 3.3 启动训练

```bash
# 全流程：训练 + 学习率曲线 + 评估 + 预测 + 汇总图
python yolov8_mask_v2/yolov8_mask_v2.py

# 仅训练
python yolov8_mask_v2/yolov8_mask_v2.py --only train

# 仅评估（需已训练）
python yolov8_mask_v2/yolov8_mask_v2.py --only val

# 仅预测
python yolov8_mask_v2/yolov8_mask_v2.py --only predict

# 仅生成学习率曲线
python yolov8_mask_v2/yolov8_mask_v2.py --only lr

# 仅生成评估汇总图
python yolov8_mask_v2/yolov8_mask_v2.py --only summary
```

### 3.4 关键训练参数

`yolov8_mask_v2/yolov8_mask_v2.py` 中的配置区：

```python
EPOCHS = 200          # 总训练轮数
BATCH = 32            # 批大小（爆显存改 16）
IMG_SIZE = 640        # 输入图片尺寸
PATIENCE = 50         # 早停：50 epoch 无提升则停止
AMP = True            # 自动混合精度（加速 + 省显存）
DEVICE = "1"          # GPU 编号，CPU 改 "cpu"

# 学习率退火策略
OPTIMIZER = "AdamW"   # 优化器
LR0 = 0.0005          # 初始学习率
LRF = 0.01            # 退火终点比例（final_lr = lr0 * lrf = 5e-6）
WARMUP_EPOCHS = 3.0   # 预热 epoch 数

# 数据增强
MOSAIC = 1.0          # Mosaic 4 图拼接
FLIPLR = 0.5          # 水平翻转概率
HSV_H = 0.015         # 色调增强
HSV_S = 0.7           # 饱和度增强
HSV_V = 0.4           # 明度增强
```

### 3.5 训练输出

训练完成后，权重和可视化结果保存在 `runs/mask_yolov8_v2/`：

```
runs/mask_yolov8_v2/
├── train-N/                       # 第 N 次训练（自动递增）
│   ├── weights/
│   │   ├── best.pt                # 最优权重（用于部署）
│   │   └── last.pt                # 最后一轮权重
│   ├── results.csv                # 每 epoch 指标
│   ├── results.png                # 训练曲线（loss/mAP）
│   ├── confusion_matrix.png      # 混淆矩阵
│   ├── confusion_matrix_normalized.png
│   └── ...
├── val-N/                          # 评估结果
│   ├── PR_curve.png               # PR 曲线
│   ├── F1_curve.png               # F1 曲线
│   ├── P_curve.png                # Precision 曲线
│   ├── R_curve.png                # Recall 曲线
│   └── confusion_matrix.png
├── predict-N/                      # 测试集预测可视化
│   └── *.jpg                      # 每张测试图的预测结果
├── lr_schedule.png                 # 学习率退火曲线
└── evaluation_summary.png          # 所有可视化汇总图
```

---

## 4. 模型评估

### 4.1 评估指标说明

训练完成后查看 `runs/mask_yolov8_v2/val-N/` 下的可视化图：

| 指标 | 含义 | 目标值 |
|------|------|--------|
| **mAP@0.5** | IoU=0.5 时的平均精度 | ≥ 0.85 |
| **mAP@0.5:0.95** | IoU 从 0.5 到 0.95 的平均精度 | ≥ 0.60 |
| **Precision** | 精确率（检测出的有多少是对的） | ≥ 0.85 |
| **Recall** | 召回率（实际目标有多少被检出） | ≥ 0.80 |

### 4.2 混淆矩阵解读

`confusion_matrix.png` 展示各类别的识别情况：

- 对角线越亮 = 该类别识别越准
- `mask` → `no-mask` 的格 = 戴口罩被误判为未戴（假阴性）
- `no-mask` → `mask` 的格 = 未戴口罩被误判为戴（假阳性，**风险更高**）

### 4.3 调优建议

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| mAP 低 | 数据量不足 | 启用更强数据增强，或增加数据 |
| Recall 低（漏检多） | 置信度阈值过高 | 降低 `MASK_CONF_THRESHOLD` 到 0.2 |
| Precision 低（误检多） | 模型过拟合 | 提高 `MASK_CONF_THRESHOLD` 到 0.35 |
| 训练 loss 不下降 | 学习率不当 | 调整 `LR0`（AdamW 建议 0.001–0.0001） |
| 显存不足 | batch 过大 | 减小 `BATCH` 或 `IMG_SIZE` |

---

## 5. 推理与预测

### 5.1 在智能体中配置模型权重

训练完成后，将 `best.pt` 路径填入配置：

```bash
cp mask_agent/config/config.yaml.example mask_agent/config/config.yaml
```

编辑 `mask_agent/config/config.yaml`：

```yaml
# 方式1：指定绝对路径
MASK_MODEL_WEIGHTS: "D:/project_k/mask-detection-agent/runs/mask_yolov8_v2/train/weights/best.pt"

# 方式2：留空，自动搜索最新训练目录下的 best.pt
MASK_MODEL_WEIGHTS: ""

# 检测参数
MASK_CONF_THRESHOLD: 0.25    # 置信度阈值
MASK_IOU_THRESHOLD: 0.7      # NMS IOU 阈值
MASK_DEVICE: "auto"          # auto / cpu / 0 / 1
```

### 5.2 检测调用模式

智能体支持两种检测调用模式：

**模式 1：进程内调用（默认，推荐）**

```yaml
MASK_USE_API: false
```

- 模型常驻内存，启动后首次检测加载模型（约 2–5 秒）
- 后续检测无加载开销，速度最快
- 缺点：Streamlit 进程占用显存

**模式 2：HTTP API 调用**

```yaml
MASK_USE_API: true
MASK_API_BASE_URL: "http://localhost:8000"
```

启动独立 API 服务：

```bash
# 终端 1
python api_server.py
# 终端 2
cd mask_agent
python agent_cli.py --gui
```

- 检测服务与 GUI 解耦，可独立扩缩容
- 适合生产部署或多客户端共享

### 5.3 单张图片快速验证

```bash
cd mask_agent
python agent_cli.py --test "检测这张图" data/images/test.jpg
```

---

## 6. 智能体部署

### 6.1 配置文件说明

完整配置项见 [config.yaml.example](../mask_agent/config/config.yaml.example)，关键配置分组：

| 配置组 | 说明 |
|--------|------|
| **LLM 配置** | 提供方切换（DashScope/Ollama）、模型名、温度 |
| **YOLOv8 配置** | 权重路径、置信度阈值、设备 |
| **联网搜索** | 结果数量、搜索模型 |
| **RAG 配置** | PDF 目录、向量索引目录、top-k |
| **编排配置** | Planner 最大步数（防死循环） |
| **错误反馈** | 工具重试次数、错误案例库 |

### 6.2 LLM 提供方切换

**模式 A：DashScope（云端，推荐）**

```yaml
LLM_PROVIDER: "dashscope"
DASHSCOPE_API_KEY: "sk-your-key-here"
MAIN_LLM_MODEL: "qwen-max"
```

申请入口：[阿里云百炼控制台](https://bailian.console.aliyun.com/)

**模式 B：Ollama（本地部署）**

```bash
# 1. 安装 Ollama: https://ollama.com/
# 2. 拉取模型
ollama pull qwen2.5:7b
# 3. 启动服务（默认监听 11434）
ollama serve
```

配置：
```yaml
LLM_PROVIDER: "ollama"
OLLAMA_BASE_URL: "http://localhost:11434/v1"
OLLAMA_MODEL: "qwen2.5:7b"
OLLAMA_NUM_CTX: 4096    # 上下文窗口
```

> 注意：Ollama 不支持 `enable_search`，web_search 工具在 Ollama 模式下退化为纯 LLM 知识回答。

### 6.3 构建 RAG 知识库（可选）

将 PDF 文档放入 `mask_agent/rag_data/`：

```
mask_agent/rag_data/
└── 你的文档.pdf
```

首次启动智能体时，系统会自动：
1. 调用 MinerU API（若配置了 `MINERU_API_KEY`）或 pypdf 将 PDF 转为 Markdown
2. 用 DashScope Embedding 模型向量化
3. 持久化到 `rag_data/storage/`

后续启动直接复用索引。配置：

```yaml
MINERU_API_KEY: "sk-your-mineru-key"  # 留空则用 pypdf 兜底
RAG_EMBED_MODEL: "text-embedding-v4"
RAG_TOP_K: 4
```

### 6.4 启动智能体

```bash
cd mask_agent

# GUI 模式（推荐）
python agent_cli.py --gui
# 浏览器自动打开 http://localhost:8501

# TUI 终端模式
python agent_cli.py --tui

# 单条测试
python agent_cli.py --test "N95 口罩和医用外科口罩有什么区别?"
```

### 6.5 生产部署建议

| 场景 | 建议 |
|------|------|
| 单机 Demo | GUI 模式 + 进程内检测 |
| 多用户共享 | API 模式（`api_server.py`）+ 反向代理 |
| 高并发 | API 模式 + 多实例 + 负载均衡 |
| 边缘设备 | Ollama 本地 LLM + yolov8n 权重 + CPU |

---

## 7. 常见问题

### Q1: 训练时报错 `CUDA out of memory`

减小 `BATCH`（如 32 → 16 → 8）或 `IMG_SIZE`（640 → 512）。

### Q2: 训练 loss 下降后又上升

过拟合。启用 `PATIENCE` 早停，或增加数据增强强度。

### Q3: 检测结果全是 `no-mask`（假阳性高）

- 检查 `MASK_CONF_THRESHOLD` 是否过低
- 检查训练数据是否类别不平衡（mask 样本过少）
- 提高 `MASK_CONF_THRESHOLD` 到 0.35–0.5

### Q4: LLM 调用 401 错误

`DASHSCOPE_API_KEY` 配置错误或已失效，到[百炼控制台](https://bailian.console.aliyun.com/)重新获取。

### Q5: Streamlit GUI 启动报 `ModuleNotFoundError`

未安装依赖：
```bash
pip install -r mask_agent/requirements.txt
```

### Q6: RAG 知识库检索无结果

首次使用需构建索引，检查：
1. `mask_agent/rag_data/` 下是否有 PDF
2. `DASHSCOPE_API_KEY` 是否有效（Embedding 也用 DashScope）
3. `rag_data/storage/` 是否有 `index_store.json`

### Q7: 模型权重路径找不到

`config.yaml` 中 `MASK_MODEL_WEIGHTS` 留空，系统会自动搜索 `runs/mask_yolov8_v2/train*/weights/best.pt`。

---

## 附录：训练流程速查

```bash
# 1. 环境准备
conda create -n mask-agent python=3.10 -y
conda activate mask-agent
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r mask_agent/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 训练（全流程）
cd mask-detection-agent
python yolov8_mask_v2/yolov8_mask_v2.py

# 3. 配置
cp mask_agent/config/config.yaml.example mask_agent/config/config.yaml
# 编辑 config.yaml 填入 API Key 和权重路径

# 4. 启动智能体
cd mask_agent
python agent_cli.py --gui
```
