# Hardware-Aftersales-AI

[中文](README.md) | [English](README_EN.md)

## 版本状态

**v1.0.0 — Graph-driven RAG Knowledge Base**

当前为 v1 稳定基线（`v1.0.0` tag / `release/v1.0` 分支）。v2（Agentic After-sales Platform）路线图见 `docs/V2_ROADMAP.md`，v1 架构事实见 `docs/V1_ARCHITECTURE.md`。

当前真实状态：

- ✅ RAG 核心回归测试通过（`eval/cluster_test.py` 16/16，纯内存指标测试）
- ✅ rc1 API 测试通过（`pytest tests/` 14 passed，含 `delete_message`、`/health`、SSE/task utils）
- ⚠️ **真实 Docker / Milvus / MongoDB / MinIO / LLM 的端到端 E2E 尚未在当前开发环境执行**。检索链路的回归以纯内存 `cluster_test.py` + 隔离 `pytest` 为准；完整的真实基础设施 E2E（`eval/run_eval.py` 及 docker-compose 全链路）需要在具备 Milvus/MongoDB/MinIO/LLM 的环境补跑后才算完整验证。

> 本项目基于 MIT 许可证开源

---

> **工业级、多路召回与重排融合的 Graph-driven RAG 智能知识库引擎**
> 
> 适用于 **AI应用工程师 / LLM应用开发 / RAG开发工程师** 学习与生产落地

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://github.com/langchain-ai/langgraph)
[![Milvus](https://img.shields.io/badge/Milvus-2.4+-orange.svg)](https://milvus.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**关键词**: `LangGraph工作流` `BGE-M3混合检索` `RRF融合重排` `HyDE假设文档` `断崖截断` `MinerU PDF解析` `企业级知识库` `智能问答系统` `流式对话SSE`

## ✨ 核心亮点

- 🔄 **LangGraph 流程编排**：基于 DAG 状态图实现高度可扩展的导入/问答 Agent 工作流
- 🎯 **实体识别预对齐**：检索前自动识别与确认商品/产品名称，大幅提升垂直领域检索精准度
- 🔍 **三路并行召回**：
  - **BGE-M3 混合检索**：同时利用 Dense（稠密）与 Sparse（稀疏）向量
  - **HyDE 假设文档**：LLM 生成假设回答后再进行语义匹配
  - **MCP WebSearch**：本地知识库无匹配时自动补充网络搜索
- 📊 **高级重排与断崖截断**：RRF (Reciprocal Rank Fusion) 融合 + Qwen3-Rerank 交叉编码，结合 score gap 智能截断噪音上下文
- 📄 **深度 PDF 解析**：集成 MinerU，精准提取表格、公式与 Markdown 图片
- 🚀 **SSE 流式响应**：实时进度推送与答案流式输出，用户体验极佳

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         前端界面                                 │
│  ┌─────────────────────┐       ┌─────────────────────┐          │
│  │  import.html        │       │  chat.html          │          │
│  │  (文档导入)          │       │  (对话问答)          │          │
│  └─────────────────────┘       └─────────────────────┘          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────┐
│  ┌─────────────────────────┐     ┌─────────────────────────┐    │
│  │   import_service        │     │   query_service         │    │
│  │   (port 8000)           │     │   (port 8001)           │    │
│  │   文档导入服务            │     │   问答查询服务            │    │
│  └─────────────────────────┘     └─────────────────────────┘    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────┐
│                      处理层 (LangGraph)                          │
│                                                                 │
│  ┌────────────────────────┐     ┌────────────────────────┐      │
│  │      导入流程           │     │      查询流程           │      │
│  │                        │     │                        │      │
│  │  1. 文件类型检测        │     │  1. 产品名称确认        │      │
│  │  2. PDF→MD转换         │     │     ↓                  │      │
│  │  3. 图片处理           │     │  ┌────┬────┬────┐      │      │
│  │  4. 文档分块           │     │  │向量│HyDE│网络│      │      │
│  │  5. 产品名识别         │     │  └──┬─┴──┬─┴──┬─┘      │      │
│  │  6. BGE-M3向量化       │     │     └──┬─┘   │        │      │
│  │  7. Milvus存储        │     │     RRF融合    │        │      │
│  │                        │     │     Rerank重排  │        │      │
│  │                        │     │     LLM生成答案  │        │      │
│  └────────────────────────┘     └────────────────────────┘      │
│                                                                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────┐
│                        基础设施层                                │
│                                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │  Milvus  │ │ MongoDB  │ │  MinIO   │ │  BGE-M3  │           │
│  │  (向量库) │ │(对话历史) │ │(对象存储) │ │ (向量化)  │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
│                                                                 │
│  ┌──────────────┐ ┌──────────────┐                               │
│  │Qwen3-Rerank  │ │  DashScope   │                               │
│  │   (重排)      │ │   (LLM)      │                               │
│  └──────────────┘ └──────────────┘                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| Web框架 | FastAPI + Uvicorn | 双服务架构 (导入/查询) |
| 前端 | HTML/CSS/JS | SSE流式对话界面 |
| 工作流 | LangGraph | DAG流程编排 |
| LLM | DashScope (Qwen) | 问答生成、文档理解 |
| 向量化 | BGE-M3 (本地) | Dense + Sparse向量 |
| 向量库 | Milvus | 文档切片检索 |
| 数据库 | MongoDB | 对话历史存储 |
| 对象存储 | MinIO | PDF/图片存储 |
| PDF解析 | MinerU | PDF→Markdown |
| 包管理 | UV | 依赖管理 + CUDA支持 |

## 🚀 快速开始

### 方式一：Docker Compose（推荐）

```bash
# 1. 克隆项目
git clone https://github.com/CW5201/Hardware-Aftersales-AI.git
cd Hardware-Aftersales-AI

# 2. 一键启动所有服务
docker-compose up -d

# 3. 访问界面
# 文档导入: http://localhost:8000/import.html
# 对话问答: http://localhost:8001/chat.html
```

### 方式二：手动安装

```bash
# 1. 安装依赖
uv sync

# 2. 下载模型
uv run python tool/download_bgem3.py
uv run python tool/download_bge_reranker_large.py

# 3. 启动服务
# 导入服务 (port 8000)
uv run python -m web.api.import_service

# 查询服务 (port 8001)
uv run python -m web.api.query_service
```

### 环境变量配置

复制 `.env.example` 为 `.env`，并填写以下配置：

```bash
# DashScope API
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

# 模型路径
BGE_M3_PATH=/path/to/bge-m3

# 数据库
MILVUS_URL=http://localhost:19530
MONGO_URL=mongodb://localhost:27017
MINIO_ENDPOINT=localhost:9000
```

## 📊 查询流程详解

### 1. 产品名称确认 (node_item_name_confirm)

- 从MongoDB获取对话历史
- LLM提取产品名并改写查询
- BGE-M3向量化后在Milvus中检索
- 三种结果分支:
  - **A. 确认匹配** → 继续多路检索
  - **B. 候选匹配** → 要求用户确认
  - **C. 无匹配** → 返回未找到提示

### 2. 多路并行检索

| 路径 | 说明 |
|------|------|
| 向量检索 | BGE-M3混合检索 (Dense+Sparse) |
| HyDE检索 | LLM生成假设文档后向量检索 |
| 网络搜索 | MCP调用阿里百炼WebSearch |

### 3. 结果融合与重排

- **RRF融合**: k=60平滑分数，合并多路结果
- **Rerank重排**: Qwen3-Rerank交叉编码器打分
- **断崖检测**: 绝对差值0.3 / 相对差值0.25截断
- **动态Top-K**: 根据分数分布选取2-5篇文档

### 4. 答案生成 (node_answer_output)

- 拼接重排文档作为上下文
- LLM生成概述式答案
- 提取文档中的图片URL
- 提取参考资料来源
- 支持SSE流式输出

## 📁 目录结构

```
Hardware-Aftersales-AI/
├── .env                    # 环境变量配置
├── pyproject.toml          # 项目依赖
├── docker-compose.yml      # Docker编排
├── Dockerfile              # 容器镜像
│
├── config/                 # 配置模块
│   ├── lm_config.py        # LLM API配置
│   ├── embedding_config.py # BGE-M3配置
│   ├── milvus_config.py    # Milvus配置
│   ├── minio_config.py     # MinIO配置
│   ├── reranker_config.py  # Rerank配置
│   └── bailian_mcp_config.py # Web搜索MCP配置
│
├── processor/              # 核心处理逻辑
│   ├── import_processor/   # 文档导入流程
│   │   ├── main_graph.py   # 导入LangGraph
│   │   └── nodes/          # 导入节点
│   │
│   └── query_processor/    # 查询问答流程
│       ├── main_graph.py   # 查询LangGraph
│       └── nodes/          # 查询节点
│
├── utils/                  # 工具模块
│   ├── embedding_utils.py  # BGE-M3向量化
│   ├── milvus_utils.py     # Milvus操作
│   ├── minio_utils.py      # MinIO操作
│   ├── llm_utils.py        # LLM客户端
│   ├── reranker_http_utils.py # Rerank API
│   ├── mongo_history_utils.py # MongoDB历史
│   ├── sse_utils.py        # SSE流式工具
│   └── task_utils.py       # 任务进度追踪
│
├── web/                    # Web层
│   ├── api/
│   │   ├── import_service.py  # 导入服务 (port 8000)
│   │   └── query_service.py   # 查询服务 (port 8001)
│   └── page/
│       ├── import.html    # 文档导入界面
│       └── chat.html      # 对话界面
│
├── eval/                   # 检索质量评测
│   ├── dataset/            # 测试数据集
│   ├── results/            # 评测结果
│   ├── run_eval.py         # 评测主入口
│   ├── retriever.py        # 检索策略封装
│   └── compute_chunk_metrics.py # 指标计算
│
└── tool/                   # 开发工具
    └── logger.py           # 日志配置
```

## 🔧 API接口

### 导入服务 (port 8000)

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /upload | 上传文档 |
| GET | /status/{task_id} | 查询任务状态 |
| GET | /health | 健康检查 |

### 查询服务 (port 8001)

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /query | 查询问答 |
| GET | /stream/{session_id} | SSE流式输出 |
| GET | /history/{session_id} | 获取对话历史 |
| DELETE | /history/{session_id} | 清空对话历史 |
| GET | /sessions | 获取所有会话 |
| DELETE | /sessions/{session_id} | 删除会话 |
| GET | /health | 健康检查 |

## 🎯 设计模式

- **LangGraph状态图**: 导入/查询均为DAG流程，TypedDict状态定义
- **BaseNode抽象基类**: 统一入口、日志、进度上报
- **懒加载单例**: Milvus/BGE-M3/MinIO/MongoDB全局单例
- **内存任务追踪**: 字典存储节点运行状态，前端轮询展示
- **幂等写入**: 按file_title去重，重复导入不产生重复数据
- **多源融合**: 向量+HyDE+网络三路检索，RRF融合+Rerank重排
- **SSE流式**: 内存Queue + 异步生成器，实时推送进度和答案

## 📊 检索质量评测

项目内置完整的 RAG 检索质量评测框架，支持 4 种检索策略 × 4 个 Top-K 的消融实验。

### 评测方案

| 编号 | 方案 | 说明 |
|------|------|------|
| A | hybrid | Dense + Sparse 混合检索，无融合/重排 |
| B | hybrid_rrf | 混合检索 + RRF 融合 |
| C | hybrid_rrf_rerank | 混合检索 + RRF + Cross-Encoder 重排 + 断崖截断 |
| D | hyde_hybrid_rrf_rerank | HyDE + 混合检索 + RRF + Rerank + 断崖截断 |

### 评测结果（34 条 Chunk-level GT）

| 方案 | Hit@5 | MRR | Recall@5 | 平均延迟 |
|------|-------|-----|----------|----------|
| A. hybrid | 94.1% | 0.875 | 83.7% | 549ms |
| B. hybrid + RRF | 94.1% | 0.875 | 83.7% | 1071ms |
| C. hybrid + RRF + Rerank | 94.1% | 0.843 | 82.0% | 2044ms |
| D. HyDE + hybrid + RRF + Rerank | 94.1% | 0.843 | 82.0% | 4976ms |

**结论**: 纯 hybrid（方案 A）最优，各项准确率最高且延迟最低。RRF/Rerank/HyDE 均无提升。

### 运行评测

```bash
# 1. 跑检索评测（需 Milvus 运行中）
python eval/run_eval.py

# 2. 计算 chunk-level 指标
python eval/compute_chunk_metrics.py
```

### 评测文件

```
eval/
├── dataset/
│   ├── rag_eval.jsonl               # 356 条原始问题集
│   └── rag_eval_chunk_level.jsonl   # 34 条 Chunk-level GT
├── results/
│   ├── summary.csv                  # 16 组汇总指标
│   ├── detailed_results.csv         # 逐题详细指标
│   ├── retrieval_results.csv        # 544 条原始检索结果
│   ├── report.md                    # 完整评测报告
│   └── figures/                     # 5 张对比图
├── run_eval.py                      # 主入口
├── retriever.py                     # RetrieverEvaluator
├── compute_chunk_metrics.py         # 指标计算
├── build_chunk_gt.py                # 构建 chunk 级 GT
└── report.py / metrics.py           # 辅助模块
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request

## 📄 License

本项目基于 MIT 许可证开源 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- [LangGraph](https://github.com/langchain-ai/langgraph) - 工作流编排框架
- [Milvus](https://milvus.io) - 向量数据库
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) - 混合检索模型
- [MinerU](https://github.com/opendatalab/MinerU) - PDF解析工具
- [DashScope](https://dashscope.aliyuncs.com) - LLM API服务

---

## 🏷️ GitHub Topics（添加到仓库右侧 About -> Topics）

```
ai-engineer llm-application agentic-rag advanced-rag knowledge-base
langgraph milvus bge-m3 qwen fastapi python
rag langchain document-parsing sse-streaming
```

如果这个项目对你有帮助，请给个 ⭐ Star 支持一下！
