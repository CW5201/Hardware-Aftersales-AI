# v2 Architecture — Agentic After-sales Platform

> v2 在 **v1 稳定 RAG 底座**之上构建。v1 检索链路（BGE-M3 混合检索 + HyDE +
> RRF + Rerank + 断崖截断 + MinerU + SSE）**保持不动、回归通过**；v2 全部改动在
> `main` 上进行。

## 1. 定位

```
v1.0  Graph-driven RAG Knowledge Base   （FROZEN：v1.0.0 tag / release/v1.0）
v2.0  Agentic After-sales Platform       （Agent 化售后平台）
```

v1 解决"检索准不准"；v2 解决"多智能体协同 + 业务闭环 + 可审计 + 可评测"。

## 2. 核心链路

```
User
 ↓
Triage Agent            意图识别 + 信息抽取 + 紧急度 + 检索策略推荐
 ↓
Memory                  短期(thread) + 长期(customer/device，数据隔离)
 ↓
Adaptive Retrieval      简单→hybrid · 复杂→hybrid_rrf_rerank · 低置信/不足→HyDE(+WebSearch)
 ↓
Diagnosis Agent         结构化诊断（hypotheses/evidence/confidence/actions/need_human_review）
 ↓
MCP Business Tools      READ / WRITE（幂等，禁止直接 SQL）
 ↓
Policy                  高风险写 → 需人工审批；READ/低风险 → 自动放行
 ↓
Human-in-the-loop       interrupt() → 审批页(approve/reject) → resume
 ↓
Ticket Workflow         PENDING→IN_PROGRESS→WAITING_APPROVAL→COMPLETED→CLOSED + 事件流
 ↓
Final Answer
（全程）Trace + Evaluation
```

## 3. 模块与目录

| 层 | 路径 | 职责 |
|----|------|------|
| 共享检索内核 | `core/retrieval/retrieval_service.py` | v1/v2 共享 RetrievalService（Step 1） |
| Adaptive 策略 | `core/retrieval/adaptive.py` | 按 triage 复杂度/证据质量选检索策略（Step 13） |
| Agent 主图 | `processor/agent_processor/main_graph.py` | Triage→Memory→Retrieval→Diagnosis→Memory |
| HITL Graph | `processor/agent_processor/hitl_graph.py` | Policy 节点 + interrupt/resume + checkpointer |
| Traced 主图 | `processor/agent_processor/traced_graph.py` | 同主图 + 节点级 trace 记录 |
| 状态 | `processor/agent_processor/state.py` | AgentState + TriageResult + DiagnosisResult |
| 节点 | `processor/agent_processor/nodes/` | triage / retrieval / diagnosis / memory / policy |
| Tool | `processor/agent_processor/tools/` | search_knowledge_base（RAG）+ business_tools（MCP 业务） |
| 业务服务 | `services/business/` | BusinessService（模拟 PG）+ TicketWorkflow + ApprovalService |
| 记忆 | `services/memory/` | 短期 + 长期（严格 customer 隔离） |
| 追踪 | `services/trace/` | run 级 + 节点级事件 |
| 评测 | `services/evaluation_service.py` | 从 trace 出 Agent 指标（无标注 → Not evaluated） |
| v2 API | `web/api/v2_service.py` | `/api/v2/*`（agent / tickets / approvals / trace / devices / eval） |
| 前端 | `web/workbench/` | Vue3 + TS + Vite + Element Plus + ECharts + SSE |
| Agent 评测 | `eval/agent_eval.py` + `eval/dataset/agent_failures.json` | 8 项 Agent 指标 + 9 类 failure |

## 4. 关键设计契约

- **结构化输出**：Triage / Diagnosis 均为 Pydantic 模型；缺失字段 `None` / `[]`，不凭空猜测型号/错误码。
- **证据优先**：Diagnosis 无证据 → `insufficient_evidence`（不编答案）；检索故障 → `retrieval_failed` + `need_human_review=True`（不做正常诊断）。
- **两种"无结果"必须区分**：`documents=[] + error=None`（正常无匹配） vs `documents=[] + error!=None`（检索系统故障，不能当"没有答案"）。
- **禁止 Agent 直接 SQL**：所有业务读写经 `BusinessService`（模拟 PostgreSQL），Tool 分 READ / WRITE。
- **写操作幂等**：`idempotency_key` + 自然键去重（同 customer+device 的 OPEN 工单不重复建），防 retry / resume 重复落库。
- **数据隔离**：Memory 与业务 Tool 严格按 `customer_id` 作用域，跨客户访问返回空（不泄露、不报错）。
- **HITL**：高风险写工具（create_service_ticket / reserve_spare_part / update_service_ticket）经 Policy → `interrupt()` 挂起 → 人工 approve/reject → `resume` 幂等执行；审批页展示 设备/客户/问题/Diagnosis/Evidence/Tool/参数。
- **Adaptive Retrieval 按需选路**：不跑全链路；知识不足才发 WebSearch 兜底信号（本层只给信号，不直接接外部网络）。
- **Trace 全链路**：`run_id / thread_id / node / agent / input / output / tool_* / retrieved_docs / latency / token_usage / error / status`，可回放 7 步生命周期。
- **评测不伪造**：无 ground truth 标注的 Agent 指标一律 `Not evaluated`；failure dataset 是可跑的判定器，不靠 LLM 自评。

## 5. API 一览

```
POST /api/v2/agent/triage          结构化意图识别
POST /api/v2/agent/run             完整主图（Triage→Memory→Retrieval→Diagnosis→Memory）
POST /api/v2/tickets               创建工单（幂等键 + 自然键去重）
GET  /api/v2/tickets?customer_id=&status=   工单列表（数据隔离）
GET  /api/v2/tickets/{id}          工单详情 + 事件流
POST /api/v2/tickets/{id}/transition        状态迁移（校验合法）
GET  /api/v2/approvals/{id}        审批详情
POST /api/v2/approvals/{id}/approve / reject
GET  /api/v2/threads/{thread_id}/trace      thread 全量 run + 事件
GET  /api/v2/runs/{run_id}          单次 run 全量事件
GET  /api/v2/devices?customer_id=  设备台账（模拟数据）
GET  /api/v2/eval/agent             Agent 级评测（无标注 → Not evaluated）
```

v1 API（import_service / query_service）保持原样、回归通过。

## 6. 验证状态

- 全量测试 `uv run pytest tests -q` → 175 通过（v1 + v2 + 前端）
- v1 检索回归 `eval/cluster_test.py` → 16/16
- 真实基础设施 E2E（Docker / Milvus / MongoDB / MinIO / LLM）→ **Not verified**（当前开发环境无 docker daemon，各服务 DOWN，无 DashScope key）。v2 全链路测试基于本地模拟数据 + mock LLM/检索。
