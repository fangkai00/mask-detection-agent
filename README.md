# 口罩检测智能体 (Mask Detection Agent)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.6+-ee4c2c.svg)](https://pytorch.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-00acc1.svg)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-purple.svg)](https://python.langchain.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-8A2BE2.svg)](https://docs.ultralytics.com/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.13+-097988.svg)](https://docs.llamaindex.ai/)
[![DashScope](https://img.shields.io/badge/DashScope-Qwen-FF6F00.svg)](https://bailian.console.aliyun.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local-000000.svg)](https://ollama.com/)
[![Tavily](https://img.shields.io/badge/Tavily-Search-01B0CD.svg)](https://tavily.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

基于 **LangGraph 编排** 的单 LLM 智能体。用户通过对话上传图片，由单一 LLM 自主决策，调度 **YOLOv8**(视觉检测)、**RAG 知识库**、**Tavily 联网搜索** 等工具，精准回答口罩标准、选型、佩戴规范等专业问题。

- **单 LLM 决策 + 多工具协同**:决策 LLM 仅负责文本理解与任务规划,图像检测由独立 YOLOv8 模型承担(LLM 本身不处理图像),实现"视觉检测 + 知识检索 + 实时信息"一站式闭环。
- **云端 / 本地双部署**:决策 LLM 可一键切换——云端用阿里云百炼(开箱即用,质量高),本地用 **Ollama** 部署(无需 API Key、数据不出域、离线可用),满足从快速体验 到 私有化落地的不同场景。
- **柔性重决策 + 终答诚实声明**:planner 重复调用已产出工具时,不直接强制收尾,而是注入 `[NoRepeat]` 提示让 LLM 重新决策一次(限 `MAX_REPLAN` 次),给自纠机会;重决策耗尽或工具失败超限时强制 finish,并把被放弃工具写入 `state.abandoned_tools`,由 finalizer 在终答中声明"XX 功能不可用",避免编造结果。配合 prompt 层"禁止重复"软约束 + 代码层 `_enforce_no_repeat` 硬判定,构成三级防死循环保障。

---

## 效果展示

### 智能体交互示例

<table>
  <tr>
    <td width="50%" align="center"><b>示例 1：口罩检测 + 合规分析</b></td>
    <td width="50%" align="center"><b>示例 2：知识问答 + 检测</b></td>
  </tr>
  <tr>
    <td><img src="docs/images/demo1.png" alt="示例1"/></td>
    <td><img src="docs/images/demo2.png" alt="示例2"/></td>
  </tr>
</table>

### YOLOv8 训练结果

<table>
  <tr>
    <td width="50%" align="center"><b>训练曲线（loss / mAP）</b></td>
    <td width="50%" align="center"><b>归一化混淆矩阵</b></td>
  </tr>
  <tr>
    <td><img src="docs/images/training_curve.png" alt="训练曲线"/></td>
    <td><img src="docs/images/confusion_matrix.png" alt="混淆矩阵"/></td>
  </tr>
  <tr>
    <td width="50%" align="center"><b>PR 曲线</b></td>
    <td width="50%" align="center"><b>F1 曲线</b></td>
  </tr>
  <tr>
    <td><img src="docs/images/pr_curve.png" alt="PR曲线"/></td>
    <td><img src="docs/images/f1_curve.png" alt="F1曲线"/></td>
  </tr>
</table>

**预测可视化示例**：

<table>
  <tr>
    <td width="50%" align="center"><b>预测示例 1</b></td>
    <td width="50%" align="center"><b>预测示例 2</b></td>
  </tr>
  <tr>
    <td><img src="docs/images/predict_demo1.jpg" alt="预测示例1"/></td>
    <td><img src="docs/images/predict_demo2.jpg" alt="预测示例2"/></td>
  </tr>
</table>

**训练指标**（实验 1：YOLOv8m, imgsz=640, 100 epochs, lr0=5e-4, lrf=0.01）：

| 指标 | 值 |
|------|-----|
| Precision | 0.8821 |
| Recall | 0.7926 |
| mAP@0.5 | 0.8839 |
| mAP@0.5:0.95 | **0.6106** |

> 实验 1 为 4 组对比实验中的综合最优（mAP@0.5:0.95 最高、Precision 最高、训练时间最短 1h41min）。完整对比详见 [experiments_comparison.html](experiments_comparison.html)。

---

## 架构

### 系统架构图

```mermaid
graph TB
    subgraph CLI["启动入口"]
        A1[agent_cli.py]
    end

    subgraph Frontend["前端交互层"]
        B1[Streamlit GUI<br/>app_streamlit.py]
        B2[TUI 终端<br/>agent/tui.py]
        B3[Test 模式<br/>agent/cli.py]
    end

    subgraph Core["编排核心（单 LLM 决策）"]
        C1[Planner 节点<br/>主 LLM 路由决策]
        C2[Finalizer 节点<br/>主 LLM 汇总回答]
    end

    subgraph Tools["工具节点（被动调用）"]
        D1[mask_detect<br/>YOLOv8 检测]
        D2[web_search<br/>Tavily 搜索]
        D3[rag_search<br/>LlamaIndex 检索]
    end

    subgraph Models["模型与服务"]
        E1[Qwen LLM<br/>qwen-max]
        E2[YOLOv8 权重<br/>best.pt]
        E3[向量索引<br/>rag_data/storage]
    end

    A1 --> B1 & B2 & B3
    B1 & B2 & B3 --> C1

    C1 -->|"路由: mask_detect"| D1
    C1 -->|"路由: search"| D2
    C1 -->|"路由: rag_search"| D3
    C1 -->|"路由: finish"| C2

    D1 & D2 & D3 -->|"结果回传"| C1

    C1 -.->|"调用"| E1
    C2 -.->|"调用"| E1
    D1 -.->|"加载"| E2
    D3 -.->|"检索"| E3
```

### 记忆分层

系统对话记忆分四层,避免短期/长期/RAG 工具概念混淆:

| 层 | 载体 | 实现 | 作用范围 |
|---|---|---|---|
| 短期记忆 | LLM context | `agent/cli.py:_build_state` 把本会话最近 N 轮(`SHORT_TERM_TURNS`,默认 8)直接拼到 `messages` | 本会话近邻,精确但受窗口限制 |
| 长期记忆 | RAG 向量库 | `tools/memory_rag.py` 把每轮对话向量化写入 `data/memory_index/`,下轮用 `user_query` 语义召回相关早期对话 | 跨会话、跨重启,突破窗口与时间衰减 |
| 文档知识库 | RAG 工具 | `tools/rag_knowledge.py` 检索 `rag_data/` 下 PDF/MD(口罩标准、生物安全规范) | 专业领域文档,与对话记忆完全独立 |
| JSON 审计 | JSON 文件 | `data/memory/session_*.json` 由 `session_store.py` 落盘 | `/export`、侧边栏会话列表、跨重启查看 |

> 关键:短期记忆只承载本会话近邻对话;长期记忆由独立向量库 `memory_rag.py` 承担,而非复用文档知识库索引——避免"对话历史"与"专业文档"在召回时串味。

### 执行流程图

```mermaid
flowchart TD
    Start([用户提问 + 可选图片]) --> Planner[Planner 决策路由]
    Planner --> CheckStep{步数 ≤ 上限?}
    CheckStep -->|否| Final
    CheckStep -->|是| Route{路由决策}

    Route -->|mask_detect| MaskDetect[YOLOv8 检测]
    Route -->|search| WebSearch[Tavily 搜索]
    Route -->|rag_search| RagSearch[LlamaIndex 检索]
    Route -->|finish| Final

    MaskDetect & WebSearch & RagSearch --> ErrorCheck{成功?}
    ErrorCheck -->|成功| BackToPlanner[结果回传]
    ErrorCheck -->|失败| RetryCheck{重试 < 上限?}
    RetryCheck -->|是| BackToPlanner
    RetryCheck -->|否| AbandonTool[记 abandoned_tools<br/>强制 finish]

    BackToPlanner --> NoRepeat{防重复检查<br/>_enforce_no_repeat}
    NoRepeat -->|未重复 / 失败重试| Planner
    NoRepeat -->|纯重复 且 重决策 < 上限| Replan[注入 NoRepeat 提示<br/>柔性重决策 1 次]
    Replan --> Planner
    NoRepeat -->|重决策耗尽 / 失败超限| AbandonTool
    AbandonTool --> Final

    Final[Finalizer 汇总回答] --> Disclaimer{触发免责声明?}
    Disclaimer -->|是| Append[追加声明] --> End([返回回答])
    Disclaimer -->|否| End
```

---

## 快速开始

> 完整训练与部署流程详见 [docs/training_deployment_guide.md](docs/training_deployment_guide.md)。

### 1. 安装

```bash
git clone https://github.com/fangkai00/mask-detection-agent.git
cd mask-detection-agent

conda create -n mask-agent python=3.10 -y
conda activate mask-agent

# PyTorch（按 CUDA 版本选择，参考 https://pytorch.org/）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 项目依赖（国内推荐清华源）
pip install -r mask_agent/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 配置

```bash
cp mask_agent/config/config.yaml.example mask_agent/config/config.yaml
```

编辑 `config.yaml`，填入 API Key：

| 配置项 | 说明 | 必填 |
|--------|------|------|
| `LLM_PROVIDER` | `dashscope`（云端）/ `ollama`（本地部署） | 是 |
| `DASHSCOPE_API_KEY` | 阿里云百炼 API Key（`dashscope` 模式必填） | `dashscope` 模式必填 |
| `OLLAMA_MODEL` | 本地 Ollama 模型名，如 `qwen2.5:7b`（`ollama` 模式必填，需先 `ollama pull`） | `ollama` 模式必填 |
| `TAVILY_API_KEY` | Tavily 联网搜索 API Key | 是 |
| `MASK_MODEL_WEIGHTS` | YOLOv8 权重路径（留空自动搜索） | 否 |
| `MAX_REPLAN` | 柔性重决策上限（纯重复时注入提示让 planner 重选次数，默认 `1`） | 否 |
| `MAX_TOOL_ERRORS` | 单工具最大失败次数（超限则放弃该工具，默认 `3`） | 否 |

### 3. 训练 YOLOv8（可选）

```bash
python yolov8_mask_v2/yolov8_mask_v2.py
```

训练完成后权重位于 `runs/mask_yolov8_v2/train*/weights/best.pt`。

### 4. 启动

```bash
cd mask_agent
python agent_cli.py --gui       # Streamlit Web GUI（推荐）
python agent_cli.py --tui       # 终端交互模式
python agent_cli.py --test "N95和医用外科口罩有什么区别?"  # 单条测试
```

浏览器自动打开 `http://localhost:8501`，支持图片上传 / 粘贴 / 截图。

> **本地部署（无外网 / 数据不出域）**：安装 [Ollama](https://ollama.com/) 后 `ollama pull 模型名称(modelscope/huggingface查找)`，在 `config.yaml` 设 `LLM_PROVIDER: "ollama"` 即可完全离线运行（决策 LLM 走本地，不调用任何云端 API）。联网搜索仍走 Tavily，与 LLM 部署方式解耦。

---

## 项目结构

```
mask-detection-agent/
├── docs/
│   ├── training_deployment_guide.md  # 训练部署指南
│   └── images/                       # README 展示图片
├── mask_agent/                       # 智能体核心
│   ├── agent/                        #   LangGraph 编排（planner / 工具节点 / finalizer）
│   ├── config/                       #   配置（config.yaml.example 模板）
│   ├── prompts/                      #   提示词
│   ├── tools/                        #   工具(口罩检测 / 联网搜索 / RAG 文档检索 / 对话长期记忆 / PDF转MD)
│   ├── data/                         #   用户数据(图片/结果/对话JSON审计/长期记忆向量索引,均不入库)
│   └── rag_data/                     #   RAG 知识库（PDF/索引不入库）
├── MaskDataSet/                      # 口罩检测数据集（mask / no-mask）
├── yolov8_mask_v2/                   # YOLOv8 训练脚本（v2，推荐）
├── agent_cli.py                      # 统一启动入口
├── app_streamlit.py                  # Streamlit GUI
├── api_server.py                     # 独立 FastAPI 检测服务
├── experiments_report.md             # 实验报告
└── LICENSE
```

---

## 注意事项

1. **API Key 安全**：`config.yaml` 已 gitignore，首次配置从 `.example` 复制
2. **YOLOv8 权重**：`*.pt` 不入库，请自行训练或下载
3. **文档知识库(RAG 工具)**:将 PDF 放入 `mask_agent/rag_data/`,首次启动自动构建索引,与对话长期记忆完全独立
4. **对话长期记忆**:`data/memory_index/` 首次对话自动构建,每轮自动向量化写入,跨会话语义召回历史对话(不入库,可由对话重新累积)
5. **第三方库**:`libs/` 和 `vendor/` 不入库,通过 `requirements.txt` 安装

---

## License

[MIT](LICENSE)
