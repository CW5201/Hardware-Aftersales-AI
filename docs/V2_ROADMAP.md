# V2 开发路线图（V2_ROADMAP.md）

> 本文档记录 `Hardware-Aftersales-AI` 从 v1.0（Graph-driven RAG Knowledge Base）
> 迭代到 v2.0（Agentic After-sales Platform）的开发顺序与每阶段的验收标准。
> 前置文档：`docs/V1_ARCHITECTURE.md`（v1 事实现状审查，Step 0 产物）。

---

## 0. 目标架构

```text
v1.0  用户问题 → RAG → 答案

v2.0  用户报修 → Triage → Memory → Retrieval → Diagnosis → Tool → Human Approval → Ticket → Final Answer
```

项目定位：**面向工业硬件售后场景的 Agentic RAG 平台**，支持多轮故障诊断、
知识检索、MCP 业务工具调用、长期记忆、人工审批、工单流转以及 Agent Trace / Evaluation。

核心架构原则（本次迭代全程遵守）：

```text
v1 RAG  →  RetrievalService  →  Agent Tool  →  v2 Agent
Agent 不允许直接操作 Milvus，必须经过 Tool → Service → Infrastructure
```

- 不重写 v1 RAG，只做"门面封装 + 策略自适应"
- 检索策略由 Agent 决策（Adaptive Retrieval），不是每次全量执行 Dense/Sparse/HyDE/WebSearch
- 所有写操作（创建工单、预留备件）必须幂等
- 高风险写操作必须 Human-in-the-loop

---

## 1. 开发顺序（Step 0 ~ Step 13）

严格按顺序推进，每个阶段结束必须完成"阶段验收清单"才能进入下一阶段。

### Step 0 — 审查当前项目 ✅（本次已完成）

产物：`docs/V1_ARCHITECTURE.md`
- 已确认 v1 调用链、稳定模块、Retrieval Pipeline、API 清单、数据库/存储、测试/评测机制
- 已确认首选抽取目标：`RetrievalService`（封装向量检索+HyDE+RRF+Rerank+断崖截断）
- 已确认 v2 起点：`processor/query_processor/main_graph_v2.py` 中已有的
  Router/Knowledge/WebSearch/Synthesizer 四 Agent 雏形

### Step 1 — 冻结 v1.0

- 基于当前 main 分支（提交 `309d817`）打 tag `v1.0.0` + 分支 `release/v1.0`
- **冻结范围（不允许在后续任何阶段破坏）**：
  - `web/api/import_service.py` / `web/api/query_service.py` 现有全部路由
  - `processor/import_processor/`、`processor/query_processor/`（含 `main_graph_v2.py`）全部现有节点行为
  - `utils/` 下现有全部工具函数签名与行为
  - `eval/` 现有检索评测脚本与指标
  - `docker-compose.yml` / `Dockerfile`
  - MongoDB `chat_message` 集合读写逻辑
  - Milvus `kb_chunks` / `kb_item_names` 两个集合 schema
- 冻结前逐项核对 `docs/V1_ARCHITECTURE.md` 第 12 节的"既有问题清单"，
  对确认要保留的问题（如 `DELETE /message/{message_id}` 未实现、`import_service` 缺 `/health`）
  明确记录"已知限制，不在本次修复范围"，避免后续被误当作 v2 引入的回归
- 验收：`git tag -l` 能看到 `v1.0.0`；`release/v1.0` 分支存在；跑一遍
  `python eval/cluster_test.py`（纯内存单测）确认基线可复现

### Step 2 — 抽取 `RetrievalService`

```text
services/retrieval_service.py
core/retrieval/            （可选：若后续 Strategy 类需要独立目录，此时再建，不强制）
```

- 实现 `RetrievalService.search(query, product_model, strategy, top_k)`
  （接口形态已在 `docs/V1_ARCHITECTURE.md` 第 11 节定死）
- **本阶段不改任何现有节点内部代码**，只是把 `node_search_embedding` /
  `node_search_embedding_hyde` / `node_rrf` / `node_rerank` 的调用序列包一层
  纯 Python 门面类，内部仍然复用现有 `NodeRrf._rrf_merge` / `NodeRerank._step_3_cliff_cutoff` 等
- 新增单元测试（不依赖外部服务）：
  - `tests/test_retrieval_service.py`：mock 掉 `generate_embeddings`/`hybrid_search`/`rerank_documents`，
    验证 4 种 strategy 的调用顺序、字段传递正确
- 新增回归测试基线：
  - 跑一次 `eval/retriever.py` 现有 4 种策略，记录输出到
    `eval/results/baseline_v1_0_0.csv`（作为 Step 2 的"改动前后对比"锚点，
    后续任何改到检索链路的步骤都必须和这份基线对比）
- 验收：`pytest tests/test_retrieval_service.py` 通过；`eval/retriever.py` 跑出的指标
  与 `baseline_v1_0_0.csv` 完全一致（说明门面类没有改变行为）

### Step 3 — Triage Agent

```text
processor/agent_processor/
├── main_graph.py
├── state.py
└── nodes/
    └── triage.py
```

- 新增 `TriageResult`（Pydantic 模型）：
  ```python
  class TriageResult(BaseModel):
      intent: str                    # fault_diagnosis / general_query / ...
      product_model: Optional[str]
      device_id: Optional[str]
      error_code: Optional[str]
      symptom: Optional[str]
      urgency: Literal["low", "medium", "high"]
      missing_information: List[str]
      retrieval_strategy: Literal["hybrid", "hybrid_rrf", "hybrid_rrf_rerank", "hyde_hybrid_rrf_rerank"]
  ```
- 策略取值**直接复用 Step 2 `RetrievalService` 的 4 种 strategy 名**，不新造词汇
- 本阶段先只做"结构化意图识别"，不接 Memory/Tool/Diagnosis（那是 Step 4~6）
- 验收：`tests/test_triage_agent.py` 覆盖"能提取字段"和"缺字段时 missing_information 正确"两类用例

### Step 4 — RAG Tool（把 `RetrievalService` 包成 Agent 可调用的 Tool）

- 新增 `processor/agent_processor/tools/search_knowledge_base.py`
- 用 LangGraph 的 tool 抽象封装 `RetrievalService.search()`，
  入参/出参都是 Step 3 已定义的 Pydantic 类型，不允许绕过 Service 直接 import `utils/milvus_utils.py`
- 写一个最小可用的 `processor/agent_processor/main_graph.py`：
  `Triage → Retrieval(Tool) → (占位)FinalAnswer`，验证 Tool 在 LangGraph 里的实际调用链路通了
- 验收：能跑通一次"提问 → Triage 结构化 → 调 search_knowledge_base Tool → 拿到证据列表"的端到端测试（可 mock 掉 LLM 部分，单独验证 Tool 协议）

### Step 5 — MCP Business Tools

```text
mcp_server/
├── server.py
└── tools/
    ├── device.py        get_device_info / get_device_status
    ├── repair.py        get_repair_history
    ├── warranty.py      get_device_warranty
    ├── inventory.py     get_spare_parts / reserve_spare_part
    └── ticket.py        create_service_ticket / update_service_ticket
```

- 数据源：本地 PostgreSQL（`docker-compose.yml` 新增 `postgres` 服务 + 初始化脚本
  `mcp_server/db/schema.sql`，建表结构见 `docs/V1_ARCHITECTURE.md` 需要补充的
  `customers`/`devices`/`repair_records`/`tickets`/`ticket_events`/`approvals` 模型，
  具体字段定义在 Step 8 前细化，本阶段先建 `devices`/`repair_records` 两张最小表用于联调）
- 工具权限分层（本阶段定义，Step 9 前不做强制拦截，Step 9 才真正接 Policy）：
  - READ：`get_device_info`/`get_device_warranty`/`get_repair_history`/`get_spare_parts`/`search_knowledge_base`
  - WRITE：`create_service_ticket`/`reserve_spare_part`/`update_service_ticket`
- 验收：`tests/test_mcp_business_tools.py` 覆盖"每个 tool 能独立调通"（mock 或直连本地 PostgreSQL）

### Step 6 — Diagnosis Agent

```text
processor/agent_processor/nodes/diagnosis.py
```

- 输入：`TriageResult` + `RetrievedChunk` 列表（Step 4）+ 可选的工具结果（Step 5）
- 输出结构化诊断结果（Pydantic）：
  ```python
  class DiagnosisResult(BaseModel):
      hypothesis: str                 # 最可能的故障原因
      confidence: float               # 0~1
      evidence_refs: List[str]        # 引用了哪些 chunk_id / tool 结果
      recommended_actions: List[str]
      need_more_information: bool
      need_human_review: bool
      need_ticket: bool
      ticket_priority: Optional[Literal["low","medium","high"]]
  ```
- 第一版只做"单轮诊断"（不做多轮追问循环，多轮留给后续版本），
  支持 `need_more_information=True` 时直接返回追问话术，不强行给结论
- 验收：`tests/test_diagnosis_agent.py` 覆盖"证据充分→给结论"、"证据不足→追问"两条主路径

### Step 7 — Memory

```text
memory/
├── short_term.py
├── long_term.py
├── memory_store.py
└── memory_retriever.py
```

- Short-term：当前 thread 内的诊断状态/工具调用记录（内存即可，参考 `utils/task_utils.py` 现有内存字典模式，但独立一套，不复用，避免语义混淆）
- Long-term：跨会话的客户/设备/历史故障记录，落在 PostgreSQL（`memory` 相关表，
  与 Step 5 的业务表分开，避免业务表和"记忆"表混在一起）
- 不同 `customer_id` / `device_id` 维度隔离
- 验收：`tests/test_memory.py` 覆盖"同一设备第二次报修能取到第一次的维修记录"

### Step 8 — Ticket Workflow

```text
services/ticket_service.py
```

- 状态机：`PENDING → IN_PROGRESS → WAITING_APPROVAL → COMPLETED / CLOSED`
- `create_service_ticket()` / `update_service_ticket()` / `get_ticket(ticket_id)`
- 幂等性：`idempotency_key` 字段 + PostgreSQL `UNIQUE` 约束（重复调用同一 key 直接返回已创建结果，不报错）
- 验收：`tests/test_ticket_service.py` 覆盖"同一 idempotency_key 调两次只创建一条记录"

### Step 9 — Human-in-the-loop

- 基于 LangGraph `interrupt()` + checkpointer（本阶段引入 `langgraph.checkpoint.postgres`，
  与 Step 5/7/8 共用同一个 PostgreSQL 实例）
- 支持 `approve` / `reject` / `resume` / `cancel` 四个动作
- 验收：`tests/test_hitl.py` 覆盖"写工具触发 interrupt → 前端 approve → resume 后继续执行"
  的完整链路（可以用脚本模拟前端，不需要真正的前端页面）

### Step 10 — Agent Trace

```text
trace/
├── models.py
├── recorder.py
├── repository.py
└── service.py
```

- 记录粒度：每个 Agent 节点执行前后各一条记录
  （`run_id`/`thread_id`/`agent`/`node`/`input`/`output`/`tool_name`/`tool_args`/`tool_result`/`retrieved_docs`/`latency`/`token_usage`/`error`/`status`）
- 落在 PostgreSQL（`agent_runs`/`agent_steps`/`tool_calls` 三张表，与 Step 5 的
  `approvals` 表同库不同表，不混用）
- 验收：`tests/test_trace.py` 验证一次完整的 Agent 执行能落出 Triage→Memory→Retrieval→Diagnosis→Tool→HITL 全链路记录

### Step 11 — Evaluation（v2 增量部分）

```text
eval/
├── retrieval/    （v1 现有内容迁移整理到这里，不改动逻辑）
├── agent/
└── datasets/
```

- 新增指标：Task Success Rate / Tool Selection Accuracy / Tool Argument Accuracy /
  Diagnosis Accuracy / Answer Groundedness / Citation Accuracy / Abstention Accuracy /
  Human Escalation Accuracy
- 新增 Failure Cases 数据集（`wrong_tool`/`wrong_argument`/`wrong_device`/`hallucination`/
  `memory_contamination`/`unsafe_write`/`wrong_escalation` 七类，每类至少 5 条用例，
  手工构造，不伪造"真实业务数据"，README 里要说明这些是构造用例）
- 验收：`pytest tests/eval/` 能跑通；每次改 Agent 逻辑必须跑一遍
  `eval/agent/` 的 regression 脚本（脚本入口：`python eval/agent/run_agent_eval.py`，本阶段实现）

### Step 12 — Vue3 Workspace

```text
web/frontend/
```

- 技术栈：Vue3 + TypeScript + Vite + Element Plus + ECharts
- 页面：Dashboard / Chat（含 Triage 状态、证据、Tool Call、Diagnosis、Final Answer 分区展示）
  / Devices / Tickets / Trace / Knowledge Base / Approval
- Chat 页面对接 Step 4/6 的 `/api/v2/agent/stream` SSE 接口
- 验收：`npm run build` 通过；本地起服务能完整走通一次"报修→诊断→审批→工单"交互

### Step 13 — README / 架构图 / Demo 脚本

- 更新 `README.md`（保留 v1 内容，追加 v2 章节，不删除原有说明）
- 新增 `docs/architecture_v2.md`（v2 架构图，参照 `docs/V1_ARCHITECTURE.md` 的"事实描述"风格，不画理想化蓝图）
- 新增 `demos/after_sales_demo.md`：一个可执行的 demo 脚本说明（输入什么报修问题 → 走哪些节点 → 最终产生什么工单），供面试演示用
- 打 tag `v2.0.0`

---

## 2. v2 新增 API 规划（与 v1 API 完全隔离，Step 12 前逐步实现）

```text
POST /api/v2/agent/run
POST /api/v2/agent/stream
GET  /api/v2/threads/{thread_id}
GET  /api/v2/threads/{thread_id}/trace
GET  /api/v2/devices/{device_id}
GET  /api/v2/tickets/{ticket_id}
POST /api/v2/approvals/{approval_id}/approve
POST /api/v2/approvals/{approval_id}/reject
```

- 全部挂在一个新的 FastAPI 子应用（`web/api/v2_service.py`，独立端口或同进程挂载
  `APIRouter(prefix="/api/v2")`，具体挂载方式在 Step 4 落地时定，本阶段不做决定）
- **不改**任何现有 v1 路由

## 3. 目录演进（局部修改 + 新增，不做大规模搬迁）

```text
Hardware-Aftersales-AI/
├── core/                      # 新增（Step 2 起逐步填充，先只放 retrieval 相关）
│   └── retrieval/
├── processor/
│   ├── import_processor/      # v1，不动
│   ├── query_processor/       # v1 + main_graph_v2.py，不动
│   └── agent_processor/       # 新增（Step 3 起）
├── mcp_server/                # 新增（Step 5）
├── memory/                    # 新增（Step 7）
├── trace/                     # 新增（Step 10）
├── services/                  # 新增（Step 2 起：retrieval_service.py；Step 8：ticket_service.py）
├── web/
│   ├── api/                   # v1，不动；Step 12 前新增 v2_service.py
│   ├── page/                  # v1，不动
│   └── frontend/             # 新增（Step 12）
├── eval/                      # v1 现有内容保留，Step 11 前新增 agent/ 子目录
└── tests/                     # 现有 test_basic.py 保留，各 Step 追加对应测试文件
```

**不做**：不移动 `utils/`、`config/` 现有文件到新目录（避免"为了目录整洁"而引入的大规模
import 路径改动风险），v2 新代码按需新增即可。

## 4. 每阶段固定验收清单（复制给每个 Step 结尾使用）

```text
[ ] 1. git diff 已人工检查，无意外改动
[ ] 2. 本阶段相关 pytest 通过
[ ] 3. v1 regression：
       - python eval/cluster_test.py 通过（纯内存）
       - 若本阶段触及检索链路：eval/retriever.py 指标 vs baseline_v1_0_0.csv 无回退
       - docker-compose 能正常 build + 启动（至少 8000/8001 两个 /health 通）
[ ] 4. 新增/改动的 API 手动 curl 验证一次
[ ] 5. 确认没有破坏 v1 任何现有路由/节点行为
[ ] 6. 按第 22 节格式向用户汇报，等待确认再进入下一阶段
```

## 5. 明确的"不做"清单（面试项目原则落地）

- 不引入 Kafka / Kubernetes / 微服务拆分
- 不做 10+ Agent，第一版就 4 个角色（Triage / Retrieval / Diagnosis / Tool）
- 不伪造企业级数据，PostgreSQL 里放的是构造的示例设备/工单数据，README 里明确标注
- 所有性能/评测数字必须来自真实跑出的 `eval/` 结果，不允许手工填入报告
- v1 现有"每个节点文件自带 `__main__` demo"的写法，v2 新代码不再延续，统一放 `tests/`
