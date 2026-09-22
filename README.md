# Hardware-Aftersales-AI

[中文](README.md) | [English](README_EN.md)

## 版本状态

**v1.0.0 — Graph-driven RAG Knowledge Base（FROZEN，`v1.0.0` tag / `release/v1.0` 分支）**

```
v1.0  Graph-driven RAG Knowledge Base
  ↓
v2.0  Agentic After-sales Platform
```

v1 是**稳定 RAG 底座**（BGE-M3 混合检索 + HyDE + RRF + Rerank + 断崖截断 + MinerU + SSE），
v2 在其之上构建**智能体售后平台**（Triage → Memory → Retrieval → Diagnosis →
MCP 业务 Tool → Policy/HITL → Ticket → Trace/Eval）。v2 全部改动在 `main` 上进行，
v1 检索链路与 v1 API 保持兼容、回归测试通过。

当前真实状态：

- ✅ v1 检索回归 16/16（`eval/cluster_test.py`，纯内存）
- ✅ 全量测试 175/175（`uv run pytest tests -q`，v1 + v2 + 前端）
- ⚠️ **真实 Docker / Milvus / MongoDB / MinIO / LLM 的端到端 E2E 尚未在当前开发环境执行**
  （开发机无 docker daemon，Milvus/MongoDB/MinIO 均 DOWN，无 DashScope key）。
  v2 全链路测试基于**本地模拟数据**（InMemory business/memory/trace store）+ mock LLM/检索；
  真实基础设施 E2E（`eval/run_eval.py` + docker-compose 全链路）需在具备依赖的环境补跑。

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

### v2 新增（Agentic After-sales Platform）

- 🧭 **Triage Agent**：把报修/提问结构化为 `TriageResult`（意图 / 型号 / 错误码 / 紧急度 / 检索策略），不凭空猜测
- 🔀 **Adaptive Retrieval**：简单 → `hybrid`，复杂 → `hybrid_rrf_rerank`，低置信 / 知识不足 → `hyde_hybrid_rrf_rerank`（+ MCP WebSearch 兜底信号），按需选路、不跑全链路
- 🩺 **Diagnosis Agent**：基于 Triage + 证据做结构化诊断（`hypotheses` / `evidence` / `confidence` / `recommended_actions` / `need_more_information` / `need_human_review`）；**无证据不编答案**（`insufficient_evidence`），检索故障不做正常诊断（`retrieval_failed` + 转人工）
- 🔌 **MCP Business Tools**：`get_device_info` / `get_device_status` / `get_device_warranty` / `get_repair_history` / `get_spare_parts`（READ）+ `create_service_ticket` / `reserve_spare_part` / `update_service_ticket`（WRITE），Agent → Tool → BusinessService → 模拟 PostgreSQL，**禁止 Agent 直接 SQL**，全链路幂等
- 🧠 **Memory**：短期（thread 维度，TTL 清理）+ 长期（customer/device 维度，**严格数据隔离**），不引入复杂向量记忆
- 🎫 **Ticket Workflow**：`PENDING → IN_PROGRESS → WAITING_APPROVAL → COMPLETED → CLOSED` 状态机 + `ticket_events` 事件流回放
- 🙋 **Human-in-the-loop**：高风险写操作经 Policy + LangGraph `interrupt()` 挂起 → 人工 approve/reject → `resume` 幂等执行；审批页展示 设备/客户/问题/Diagnosis/Evidence/Tool/参数
- 📈 **Agent Trace**：记录 run 级 + 节点级（input/output/tool/retrieved_docs/latency/token/error），`GET /api/v2/threads/{id}/trace` + `GET /api/v2/runs/{id}` 可回放全链路
- 🧪 **Agent Evaluation**：Task Success / Tool Selection / Tool Argument / Diagnosis / Groundedness / Citation / Abstention / Human Escalation Accuracy + 9 类 failure dataset；**无真实标注 → 明确 "Not evaluated"，不编数字**
- 🖥️ **Vue3 工作台**：Vue3 + TypeScript + Vite + Element Plus + ECharts + SSE，8 个页面（Dashboard / Chat / Devices / Tickets / Approvals / Trace / Evaluation / Knowledge Base），重点展示 Agent 生命周期而非聊天框

## 🏗️ 系统架构

### v1 → v2 演进

```
v1.0  Graph-driven RAG Knowledge Base
   BGE-M3 混合检索 + HyDE + RRF + Rerank + 断崖截断 + MinerU + SSE
   └─ 稳定检索底座（FROZEN）

v2.0  Agentic After-sales Platform（在 v1 检索底座之上）
   User
    ↓
   Triage Agent          意图识别 + 信息抽取 + 紧急度 + 策略推荐
    ↓
   Memory                短期(thread) + 长期(customer/device，数据隔离)
    ↓
   Adaptive Retrieval    简单→hybrid · 复杂→hybrid_rrf_rerank · 低置信/不足→HyDE(+WebSearch)
    ↓
   Diagnosis Agent       结构化诊断（hypotheses/evidence/confidence/actions/need_human_review）
    ↓
   MCP Business Tools    READ / WRITE（幂等，禁止直接 SQL）
    ↓
   Policy                高风险写 → 需人工审批；READ/低风险 → 自动放行
    ↓
   Human-in-the-loop     interrupt() → 审批页(approve/reject) → resume
    ↓
   Ticket Workflow       PENDING→IN_PROGRESS→WAITING_APPROVAL→COMPLETED→CLOSED + 事件流
    ↓
   Final Answer
   （全程）Trace + Evaluation
```

### v2 主图（LangGraph）

```
START
  ↓
Triage            结构化意图识别
  ↓
Memory Retrieve   注入 thread 短期 + customer/device 长期记忆
  ↓
Retrieval         Adaptive 选策略 → search_knowledge_base Tool → v1 检索 → 证据
  ↓
Diagnosis         基于证据的结构化诊断（无证据不编）
  ↓
Memory Persist    写回短期；diagnosed → 记长期故障
  ↓
END

（高风险写操作旁路）
  Policy → interrupt() → Approval(pending) → approve/reject → resume → 执行(幂等)
```

### v1 架构（保留不变）

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
│  导入流程: 文件检测 → PDF→MD → 图片 → 分块 → 产品名识别 → BGE-M3 → Milvus
│  查询流程: 产品名确认 → 多路检索(向量/HyDE/网络) → RRF → Rerank → LLM 答案
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────┐
│                        基础设施层                                │
│  Milvus(向量) · MongoDB(历史) · MinIO(对象) · BGE-M3(向量化)
│  Qwen3-Rerank(重排) · DashScope(LLM)
└───────────────────────────────────────────────────────────────────┘
```

## 🛠️ 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| Web框架 | FastAPI + Uvicorn | 双服务架构 (导入/查询) + v2 Agent API |
| 前端 | HTML/CSS/JS（v1） · Vue3 + TS + Vite + Element Plus + ECharts + SSE（v2 工作台） | 界面 |
| 工作流 | LangGraph | DAG 流程编排（v1 导入/查询 + v2 Agent 主图 + HITL） |
| LLM | DashScope (Qwen) | 问答生成、Triage / Diagnosis / HyDE |
| 向量化 | BGE-M3 (本地) | Dense + Sparse 向量 |
| 向量库 | Milvus | 文档切片检索 |
| 数据库 | MongoDB | 对话历史存储 |
| 对象存储 | MinIO | PDF/图片存储 |
| PDF解析 | MinerU | PDF→Markdown |
| Agent | Triage / Diagnosis / Policy / Memory | v2 智能体售后平台 |
| 业务数据 | InMemory（模拟 PostgreSQL） | 客户 / 设备 / 备件 / 工单 / 审批（本地模拟，可换真实 DB） |
| 包管理 | UV | 依赖管理 + CUDA 支持 |
| 前端构建 | Vite + vue-tsc | v2 工作台 |

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
├── core/                   # v1/v2 共享内核
│   └── retrieval/          # RetrievalService（v1 检索门面）+ adaptive.py（Adaptive 策略）
│
├── processor/              # 核心处理逻辑
│   ├── import_processor/   # 文档导入流程（v1）
│   ├── query_processor/    # 查询问答流程（v1 + v1.1 multi-agent）
│   └── agent_processor/    # v2 Agent 主图（Triage/Retrieval/Diagnosis/Memory/Policy）
│       ├── main_graph.py   # Triage→Memory→Retrieval→Diagnosis→Memory 主图
│       ├── hitl_graph.py   # HITL Graph（interrupt/resume + checkpointer）
│       ├── traced_graph.py # 带 Trace 记录的主图
│       ├── state.py        # AgentState + TriageResult + DiagnosisResult
│       ├── nodes/          # triage / retrieval / diagnosis / memory / policy
│       └── tools/          # search_knowledge_base + business_tools（MCP 业务工具）
│
├── services/               # 业务服务层
│   ├── business/           # BusinessService（模拟 PostgreSQL）+ TicketWorkflow + ApprovalService
│   ├── memory/             # 短期 + 长期 Memory（严格 customer 数据隔离）
│   ├── trace/              # Agent Trace（run 级 + 节点级事件）
│   └── evaluation_service.py # Agent 评测（无标注 → Not evaluated）
│
├── utils/                  # 工具模块（v1 检索 / SSE / Mongo / 重排等）
├── web/                    # Web层
│   ├── api/                # import_service / query_service / v2_service（v2 Agent API）
│   ├── page/               # v1 静态页面（import.html / chat.html）
│   └── workbench/          # v2 Vue3 工作台（Vue3 + TS + Vite + Element Plus + ECharts）
│
├── eval/                   # 检索质量评测（v1）+ agent_eval.py（v2 Agent 评测）+ failure dataset
└── tool/                   # 开发工具
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

### v2 Agent API（挂载于查询服务，`/api/v2` 命名空间，与 v1 路由隔离）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/v2/agent/triage | Triage 结构化意图识别 |
| POST | /api/v2/agent/run | 完整主图（Triage→Memory→Retrieval→Diagnosis→Memory） |
| POST | /api/v2/tickets | 创建工单（幂等键 + 自然键去重） |
| GET | /api/v2/tickets | 工单列表（customer / status 过滤，数据隔离） |
| GET | /api/v2/tickets/{id} | 工单详情 + 事件流 |
| POST | /api/v2/tickets/{id}/transition | 状态迁移（校验合法迁移） |
| GET | /api/v2/approvals/{id} | 审批详情（展示 设备/客户/问题/Diagnosis/Evidence/Tool/参数） |
| POST | /api/v2/approvals/{id}/approve | 批准 |
| POST | /api/v2/approvals/{id}/reject | 驳回 |
| GET | /api/v2/threads/{thread_id}/trace | 该 thread 全量 run + 事件 |
| GET | /api/v2/runs/{run_id} | 单次 run 全量事件 |
| GET | /api/v2/devices | 设备台账（模拟数据，按 customer 隔离） |
| GET | /api/v2/eval/agent | Agent 级评测（无标注 → Not evaluated） |

## 🎯 设计模式

- **LangGraph状态图**: 导入/查询均为DAG流程，TypedDict状态定义
- **BaseNode抽象基类**: 统一入口、日志、进度上报
- **懒加载单例**: Milvus/BGE-M3/MinIO/MongoDB全局单例
- **内存任务追踪**: 字典存储节点运行状态，前端轮询展示
- **幂等写入**: 按file_title去重，重复导入不产生重复数据
- **多源融合**: 向量+HyDE+网络三路检索，RRF融合+Rerank重排
- **SSE流式**: 内存Queue + 异步生成器，实时推送进度和答案

### v2 设计模式

- **结构化契约**: Triage / Diagnosis 均为 Pydantic 模型，缺失字段 None/[]，不凭空猜测
- **证据优先**: Diagnosis 无证据不编答案（insufficient_evidence），检索故障不做正常诊断（retrieval_failed + need_human_review）
- **READ / WRITE 分类**: 业务 Tool 区分读/写；写操作强制幂等键（idempotency_key + 自然键去重），防 retry/resume 重复建单
- **数据隔离**: Memory 与业务 Tool 严格按 customer_id 作用域，跨客户访问返回空（不泄露、不报错）
- **HITL interrupt/resume**: LangGraph checkpointer 持久化 pending；审批页展示完整上下文；execute 幂等
- **Adaptive Retrieval 按需选路**: 不跑全链路，按问题复杂度/证据质量选 strategy；知识不足才触发 WebSearch 兜底信号
- **Trace 全链路**: run 级 + 节点级事件，可回放 Triage→Memory→Retrieval→Diagnosis→Tool→Approval→Ticket
- **评测不伪造**: 无 ground truth 标注的指标一律 "Not evaluated"

## 🧪 测试与验证

### 全量测试

```bash
uv run pytest tests -q          # v1 + v2 + 前端（175 通过）
uv run python eval/cluster_test.py   # v1 检索回归（16/16）
```

测试覆盖（按模块）：

| 模块 | 测试文件 | 说明 |
|------|---------|------|
| v1 基础 | test_basic / test_delete_message / test_rc1_service | import / query / SSE / 历史 |
| v1 检索回归 | test_retrieval_service_regression | RetrievalService 各策略 |
| v2 Triage | test_triage_agent | 结构化意图识别 |
| v2 RAG Tool | test_search_knowledge_base | RetrievalService → Tool 包装 |
| v2 Agent Flow | test_agent_retrieval_flow | Triage→Memory→Retrieval→Diagnosis 全链路 |
| v2 Diagnosis | test_diagnosis_agent | 有/无证据 / 故障 / 降级 |
| v2 MCP Tools | test_business_tools | 8 个业务 Tool（schema/正常/无数据/非法/异常/幂等） |
| v2 Memory | test_memory_service | 短期/长期 + customer 数据隔离 |
| v2 Ticket | test_ticket_workflow | 状态机 + 事件流 + 隔离 |
| v2 HITL | test_hitl_flow | interrupt/resume + 幂等去重 |
| v2 Trace | test_agent_trace | run/节点事件 + API |
| v2 Eval | test_agent_eval | Agent 指标 + failure dataset（不伪造） |
| v2 Adaptive | test_adaptive_retrieval | 策略判定 + 节点集成 |
| v2 前端 | test_vue_workbench | 文件/路由/生命周期/构建 |
| 集成 | test_full_integration | 主图 + HITL + Adaptive + API 协同 |

### 真实基础设施 E2E（需具备依赖的环境）

```
当前开发环境：Not verified（无 docker daemon；Milvus/MongoDB/MinIO DOWN；无 DashScope key）
```

在具备 Docker / Milvus / MongoDB / MinIO / DashScope API 的环境补跑：

```bash
docker-compose up -d                       # 起 Milvus/Mongo/MinIO
uv run python tool/download_bgem3.py       # 下模型
uv run python eval/run_eval.py             # 真实检索评测
uv run python -m web.api.query_service      # 起服务，走 /query /sse /api/v2/agent/run
```

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

### v2 Agent 级评测

在 v1 检索指标（Hit@K / MRR / Recall@K / Latency，`eval/metrics.py`）之上，v2 新增 Agent 级指标
（`eval/agent_eval.py`）：Task Success / Tool Selection / Tool Argument / Diagnosis /
Answer Groundedness / Citation / Abstention / Human Escalation Accuracy，外加 9 类
failure dataset（wrong_tool / wrong_argument / wrong_device / missing_evidence /
hallucination / memory_contamination / unsafe_write / wrong_escalation / duplicate_ticket）。

**原则：没有真实 ground truth 标注的指标一律 "Not evaluated"，不编造数字。**

### 运行评测

```bash
# 1. 跑检索评测（需 Milvus 运行中）
python eval/run_eval.py

# 2. 计算 chunk-level 指标
python eval/compute_chunk_metrics.py

# 3. v2 Agent 评测（本地无标注 → 指标 Not evaluated；failure dataset 可跑判定器）
python -m eval.agent_eval
```

### 评测文件

```
eval/
├── dataset/
│   ├── rag_eval.jsonl               # 356 条原始问题集
│   ├── rag_eval_chunk_level.jsonl   # 34 条 Chunk-level GT
│   └── agent_failures.json          # v2 failure dataset（9 类失败样本）
├── results/
│   ├── summary.csv                  # 16 组汇总指标
│   ├── detailed_results.csv         # 逐题详细指标
│   ├── retrieval_results.csv        # 544 条原始检索结果
│   ├── report.md                    # 完整评测报告
│   └── figures/                     # 5 张对比图
├── run_eval.py                      # 主入口
├── retriever.py                     # RetrieverEvaluator
├── compute_chunk_metrics.py         # 指标计算
├── agent_eval.py                    # v2 Agent 级指标 + failure 判定器
├── build_chunk_gt.py                # 构建 chunk 级 GT
└── report.py / metrics.py           # 辅助模块
```

## 🖥️ v2 Vue3 工作台

```bash
cd web/workbench
npm install
npm run dev        # http://localhost:5173（vite 代理 /api → :8000）
npm run build      # 产物 dist/
```

8 个页面：Dashboard（生命周期 + 延迟分布）/ Chat（全链路展示）/ Devices /
Tickets（状态机 + 事件流）/ Approvals（HITL 审批）/ Trace（全链路回放）/
Evaluation（指标 + failure）/ Knowledge Base。重点展示 Agent 生命周期
Triage→Memory→Retrieval→Diagnosis→Tool→Approval→Ticket，而非聊天框。

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
