# V1 架构审查报告（V1_ARCHITECTURE.md）

> 本文档仅记录当前 `Hardware-Aftersales-AI` 仓库 v1.0 的事实现状，不修改任何业务代码。
> 审查时间：2026-09-21。基于 main 分支当前提交 `309d817`。
> 仓库为独立 git 仓库（嵌套于 `knowledge_base` 目录下，有自己的 `.git`），本文档中"仓库"均指 `Hardware-Aftersales-AI` 这个独立仓库。

---

## 0. 审查范围说明

- 仓库根目录：`Hardware-Aftersales-AI/`
- 当前 main 分支最新提交：`309d817 feat: 添加检索质量评测框架与消融实验结果`
- 工作区（working tree）干净，仅有 3 个 untracked 辅助文件（`.commit_list.txt`、`.gen_plan.py`、`.newdates_plan.txt`，均为生成 git 历史时间戳用的辅助脚本，与业务代码无关，不属于本次 v1 冻结范围）
- 外层 `knowledge_base` 仓库另有若干 untracked 文件（`Hardware-Aftersales-AI` 目录本身、`cv_content.txt` 等），属于外层仓库的历史遗留，与本项目代码隔离，不影响本仓库内部运行

## 1. 当前 v1.0 完整调用链

### 1.1 导入链路（`import_service`，port 8000）

```
web/api/import_service.py:upload_files()
  └─ 逐文件：保存到本地 temp-files/ → MinIO 持久化 → background_tasks.add_task(run_graph_task)
run_graph_task()
  └─ processor/import_processor/main_graph.py:KBImportWorkflow
       node_entry            (node_enrty.py)        判断文件类型 → is_pdf_read_enabled / is_md_read_enabled
         ├─(PDF) node_pdf_to_md   (node_pdf_to_md.py)   MinerU API 上传PDF → 轮询 → 下载ZIP解压 → md_content
         └─(MD)  直接进入下游
       node_md_img             (node_md_img.py)      解析MD中图片引用，补充图片说明
       node_document_split     (node_document_split.py)  按MD标题切分 + 长短文调节 → chunks
       node_item_name_recognition (node_item_name_recognition.py)  LLM识别商品名 → 生成向量 → 写入 kb_item_names 集合
       node_bge_embedding      (node_bge_embedding.py)  对chunks逐条生成 dense+sparse 向量
       node_import_milvus      (node_import_milvus.py)  自动建表(若不存在) → 幂等清理同file_title旧数据 → 批量插入 → 回填chunk_id
```

- 各节点继承 `BaseNode`（`processor/import_processor/base.py`），通过 `__call__` 统一做任务追踪（`task_utils.add_running_task/add_done_task`，key 为 `task_id`）和日志
- 前端通过 `GET /status/{task_id}` 轮询 `task_utils` 内存字典获取节点级进度

### 1.2 查询链路（`query_service`，port 8001）

```
web/api/query_service.py:POST /query 或 POST /query/v2
  └─ is_stream ? background_tasks.add_task(run_query_graph)  : 同步 run_query_graph()
run_query_graph()  → processor/query_processor/main_graph.py:KBQueryWorkflow (v1)
     或 run_query_v2_graph() → processor/query_processor/main_graph_v2.py:KBQueryWorkflowV2 (v2 multi-agent 雏形，见下文 1.3)

v1 主流程（KBQueryWorkflow）:
  node_item_name_confirm  (node_item_name_confirm.py)
      1. MongoDB 取历史 get_recent_messages()
      2. save_chat_message() 先落库用户消息
      3. LLM(ITEM_MODEL, json_mode) 提取 item_names + 改写 rewritten_query
      4. 逐条 item_names 向量化 → Milvus kb_item_names 混合检索(10条)
      5. 阈值对齐（>0.85确认 / >=0.65候选top3 / 其余拒绝）
      分支A：确认 → 继续检索；分支B：候选 → 直接写 answer（反问）；分支C：无匹配 → 直接写 answer（拒识）
  node_multi_search (虚拟分叉节点)
      ├─ node_search_embedding      (node_search_embedding.py)  用 rewritten_query 混合检索 kb_chunks（带 item_name 过滤）
      ├─ node_search_embedding_hyde (node_search_embedding_hyde.py)  LLM生成假设文档 + 混合检索
      └─ node_web_search_mcp        (node_web_search_mcp.py)  调用阿里百炼 MCP WebSearch（openai-agents 的 MCPServerStreamableHttp）
  node_join → node_rrf (node_rrf.py, 两路结果 RRF 融合, max_results=5)
             → node_rerank (node_rerank.py, 合并 local+web 文档, DashScope Qwen3-Rerank 打分, 断崖截断2~5条)
             → node_answer_output (node_answer_output.py, LLM生成答案, 提取图片URL/参考资料, 写Mongo历史, SSE final事件)
```

- 各节点继承 `NodeBase`（`processor/query_processor/base.py`），通过 `__call__` 统一做任务追踪（key 为 `session_id`）和日志
- `add_done_task(session_id, name, is_stream)` 若 `is_stream=True` 会同时向 SSE 队列推送 `progress` 事件（`task_utils.task_push_queue`）

### 1.3 已存在的 v2 雏形（重要发现）

`processor/query_processor/main_graph_v2.py` 已实现一套 `KBQueryWorkflowV2`（Router Agent → Knowledge Agent / WebSearch Agent → Synthesizer Agent），且已被 `query_service.py` 的 `POST /query/v2` 接口挂载。这是本次 Step 0 需要特别注意的现状：**v2 多 Agent 骨架其实已经存在一部分，只是尚未按用户规划的 `processor/agent_processor/` 目录结构组织，且角色划分（Router/Knowledge/WebSearch/Synthesizer）与用户最终想要的（Triage/Retrieval/Diagnosis/Tool）不完全一致**，v2.0 应该基于此继续演进而不是重新发明。

## 2. v1 核心稳定模块

| 模块 | 路径 | 职责 | 稳定性判断 |
|---|---|---|---|
| 导入工作流 | `processor/import_processor/main_graph.py` | 7节点LangGraph DAG | 稳定，被 `import_service.py` 直接调用 |
| 导入节点 | `processor/import_processor/nodes/*` | PDF解析/切块/识别/向量化/入库 | 稳定 |
| 查询工作流 | `processor/query_processor/main_graph.py` | 9节点LangGraph DAG | 稳定，被 `query_service.py` 调用 |
| 查询节点 | `processor/query_processor/nodes/*` | 实体确认/三路检索/RRF/重排/答案 | 稳定，被 `main_graph_v2.py` 也复用（直接 import 这些 Node 类，说明节点本身是"无状态可复用"的设计，这是后续抽 `RetrievalService` 的关键前提） |
| 向量检索 | `utils/milvus_utils.py` | MilvusClient 单例、混合检索请求构造、`hybrid_search`、幂等批插 | 稳定，`eval/retriever.py` 也直接复用这里的函数 |
| 向量化 | `utils/embedding_utils.py` | BGE-M3 单例，`generate_embeddings()` 返回 dense+sparse | 稳定 |
| Rerank | `utils/reranker_http_utils.py` | DashScope Qwen3-Rerank API 封装 | 稳定 |
| LLM 客户端 | `utils/llm_utils.py` | `get_llm_client(model, json_mode)`，带缓存 | 稳定 |
| 历史存储 | `utils/mongo_history_utils.py` | MongoDB chat_message 集合读写 | 稳定 |
| SSE | `utils/sse_utils.py` | 内存Queue + 异步生成器 | 稳定 |
| 任务追踪 | `utils/task_utils.py` | 内存字典追踪节点running/done状态 | 稳定 |
| 对象存储 | `utils/minio_utils.py` | MinIO 上传 | 稳定 |

## 3. 适合抽成 Service 的代码（Step 2 的候选）

**首选抽取对象：检索 Pipeline（`node_search_embedding.py` + `node_search_embedding_hyde.py` + `node_rrf.py` + `node_rerank.py` 的组合能力）**

理由：
1. 这四个节点已经是"纯计算"的：输入是 state 里的字段（`rewritten_query`/`item_names`/`embedding_chunks`/`hyde_embedding_chunks`/`web_search_docs`），输出也是明确的字段（`embedding_chunks`/`hyde_embedding_chunks`/`rrf_chunks`/`reranked_docs`），没有隐藏副作用（除了 `task_utils.add_done_task` 的进度上报，这个可以做成可选参数或抽走）
2. `eval/retriever.py` 里 `RetrieverEvaluator` 已经在**没有改业务代码**的前提下，直接 import 了 `NodeRrf._rrf_merge` 和 `NodeRerank._step_3_cliff_cutoff` 来复用检索能力——这说明"把检索能力抽成独立可调用的函数/Service"这个模式在这个代码库里已经被验证过，不是新事物
3. `main_graph_v2.py` 里的 `AgentKnowledge` 已经把 `NodeSearchEmbedding`/`NodeSearchEmbeddingHyde`/`NodeRrf` 包装成"Agent 内调用"的形式（`_search_embedding`/`_search_hyde`/`_rrf_merge` 方法），这是现成的"Agent 直接调 Service"雏形，只差一层 `RetrievalService.search(query, product_model, strategy, top_k)` 的正式封装

**次要候选（后续 Step 再做）：**
- `NodeWebSearchMcp` → 抽成 `WebSearchService`（目前 MCP 调用直接硬编码在节点里，`agents.mcp.MCPServerStreamableHttp` 的实例化逻辑可以下沉为 Service）
- `node_item_name_confirm` 里的"实体提取+对齐"逻辑（`_step_4_extract_info`/`_step_5_vectorize_and_query`/`_step_6_align_item_names`）→ 可抽成 `EntityResolutionService`，供 Triage Agent 复用，不必重写

## 4. 当前 Retrieval Pipeline 实现细节

- 两路本地检索（`node_search_embedding` 直接检索、`node_search_embedding_hyde` 假设文档+query拼接后检索）都调用同一个底层函数链：
  `generate_embeddings([text])` → `create_hybrid_search_requests(dense, sparse, expr, limit=10)` → `hybrid_search(client, collection, reqs, ranker_weights=(0.8,0.2), limit, output_fields)`
- 过滤表达式：`expr = f'item_name in {item_names}'`（直接 f-string 拼接，`item_names` 是 `node_item_name_confirm` 对齐后的标准化商品名列表；`eval/retriever.py` 里有一版更安全的写法用 `escape_milvus_string` + 加引号，说明这是已知的一个小隐患：`node_search_embedding.py`/`node_search_embedding_hyde.py` 当前实现没有做字符串转义，若商品名本身含引号/特殊字符会导致 Milvus 表达式解析失败，这是 v2 重构时应该顺手修掉的小问题，但**属于现有行为，v1 冻结阶段不动它**）
- 权重固定 `(0.8, 0.2)`（dense 权重更高）
- RRF：`k=60`，`max_results=5`，两路等权重 `1.0`，输入输出字段是 `chunk_id`/`content`/`item_name`
- Rerank：把 `rrf_chunks`（local）和 `web_search_docs`（web）合并成统一结构 `[{chunk_id,title,content,url,source}]`，调用 `rerank_documents()`（DashScope，`top_n=len(documents)` 全量打分，不截断），按分数降序，再做"断崖截断"：`RERANK_MIN_TOPK=2`、`RERANK_MAX_TOPK=5`、绝对差 `0.3`、相对差 `0.25`
- `node_answer_output` 拿到 `reranked_docs` 后：按 `MAX_CONTEXT_CHARS=12000` 预算控制，逐条拼 `[idx][source=][chunk_id=][url=][title=][score=]` 元数据头 + 正文喂给 LLM

## 5. 当前 v1 API 清单

### import_service（port 8000）
| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/import.html` | 导入前端页面（静态文件） |
| POST | `/upload` | 多文件上传，返回 `task_ids`，触发后台 LangGraph 流程 |
| GET | `/status/{task_id}` | 轮询任务进度（`status`/`done_list`/`running_list`） |
| GET | `/health` | 健康检查 |

（注意：v1.0.0 冻结前，`import_service.py` 原本**没有**独立的 `/health` 路由，而 CI 里 `curl http://localhost:8000/health` 实际打不到 8000 的端点（该端点只存在于 8001 的 `query_service`）——CI 大概率一直在这里 fail 但从未被真正验证。此问题已在 **v1.0.0-rc1 修复**：`import_service` 已新增 `GET /health`（返回 `{"ok": true}`），CI 的 health 检查改为轮询两个服务的真实端点。下表已按修复后的现状列出。）

### query_service（port 8001）
| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/chat.html` | 查询前端页面 |
| POST | `/query` | v1 查询（`KBQueryWorkflow`） |
| POST | `/query/v2` | v2 multi-agent 雏形（`KBQueryWorkflowV2`） |
| GET | `/stream/{session_id}` | SSE 流式 |
| GET | `/sessions` | 所有会话列表 |
| DELETE | `/sessions/{session_id}` | 删除整个会话 |
| DELETE | `/history/{session_id}` | 清空历史 |
| DELETE | `/message/{message_id}` | 删除单条消息（v1.0.0-rc1 已实现 `delete_message`，不再 500） |
| GET | `/history/{session_id}` | 查询会话历史 |
| GET | `/health` | 健康检查 |

## 6. 需要兼容的数据库/数据结构

### Milvus（`utils/milvus_utils.py` + `node_import_milvus.py` 自动建表）
- 集合 `kb_chunks`（`CHUNKS_COLLECTION`），schema（自动建表逻辑在 `node_import_milvus._create_chunks_collection`）：
  ```
  chunk_id   INT64        主键, auto_id
  content    VARCHAR(65535)
  title      VARCHAR(200)
  parent_title VARCHAR(200)
  part       INT8
  file_title VARCHAR(200)
  item_name  VARCHAR(200)   ← 幂等清理依据 + 检索过滤字段
  sparse_vector SPARSE_FLOAT_VECTOR  (IP, DAAT_MAXSCORE, normalize)
  dense_vector  FLOAT_VECTOR(dim=1024)  (COSINE, AUTOINDEX)
  ```
- 集合 `kb_item_names`（`ITEM_NAME_COLLECTION`），字段：`item_name`、`dense_vector`、`sparse_vector`（由 `node_item_name_recognition.py` 写入，schema 定义在该节点内，非 chunks 集合）
- **v2 新增 PostgreSQL 业务库不能影响这两个集合**，二者是"文档级检索"的核心资产

### MongoDB（`utils/mongo_history_utils.py`）
- 库 `kb001`（`MONGO_DB_NAME`），集合 `chat_message`，索引 `(session_id, ts_desc)`
- 文档字段：`session_id`/`role`/`text`/`rewritten_query`/`item_names`/`image_urls`/`references`/`ts`
- **v1 全部历史读写逻辑（`save_chat_message`/`get_recent_messages`/`get_all_sessions`/`clear_history`/`update_message_item_names`）必须原样保留**，v2 的 Memory 层是"新增"，不是替换

### MinIO
- Bucket `knowledge-base`（`MINIO_BUCKET_NAME`），仅用于导入链路持久化原始文件，查询链路不直接依赖

## 7. 当前测试运行方式

- `tests/test_basic.py`：基础结构/导入类测试（`pytest`），大部分用 `pytest.skip` 兜底，实际覆盖面很浅（主要验证目录结构存在、模块可 import）
- `eval/cluster_test.py`：`TestRRF` 等**纯内存**单测（不依赖外部服务），可以直接 `python eval/cluster_test.py` 或 `pytest` 运行
- `eval/run_eval.py`：**检索评测主入口**，需要 Milvus 已启动且已导入数据，会调用真实 `generate_embeddings`/`hybrid_search`/`rerank_documents`（真实 LLM API 调用），非 CI 可跑的东西，是本地/手动跑的"重"评测
- 运行方式：`uv run pytest tests/ -v`（单元测试）；`python eval/run_eval.py`（检索评测，需 Milvus）
- **没有针对 v1 查询链路（`KBQueryWorkflow` 整体）的集成测试**，也没有 regression test 基线

## 8. Docker 启动方式

- `docker-compose.yml`：单一 `app` 服务（`import_service`+`query_service` 双进程，`CMD` 里用 `&` 并发启动两个 uvicorn）+ `milvus`(standalone) + `etcd` + `mongo` + `minio` 四个基础设施服务，全部本地可运行，无云依赖
- 启动：`docker-compose up -d`，前端 `http://localhost:8000/import.html`、`http://localhost:8001/chat.html`
- 非 Docker 方式：`uv run python -m web.api.import_service`（8000）+ `uv run python -m web.api.query_service`（8001），依赖先 `docker-compose up -d milvus etcd mongo minio` 或本地已装这些服务
- `.env` 需要真实配置（BGE-M3 本地模型路径、Milvus/Mongo/MinIO 连接串、DashScope API Key、MinerU token）

## 9. 当前评测运行方式（Step 11 扩展的基线）

- 检索指标已实现且**可离线计算**：`eval/metrics.py` 的 `hit_at_k`/`mrr`/`recall_at_k`/`ndcg_at_k`，纯函数，有独立单测（`eval/cluster_test.py` 里的部分用例）
- 但**当前测试集缺 chunk 级 ground truth**（`eval/retriever.py` 注释明确说明"由于当前测试集没有chunk级别的ground truth，我们用file_title级别的近似评估"），`eval/build_chunk_gt.py` 是补这块的脚本，但需要跑 Milvus 全量 chunk 加载 + 人工/规则匹配，产出 `rag_eval_chunk_level.jsonl`
- `RetrieverEvaluator` 4 种策略（`hybrid`/`hybrid_rrf`/`hybrid_rrf_rerank`/`hyde_hybrid_rrf_rerank`）直接复用业务代码里的 `NodeRrf`/`NodeRerank`，这是**很好的回归测试载体**：v2 任何对检索链路的改动，都应该让 `eval/retriever.py` 这套评测在改动前后指标可对比，这是"不破坏 v1"最直接的验证手段
- RAGAS 语义评估（`utils/evaluator.py` 的 `RAGEvaluator`，Faithfulness/Answer Relevance/Context Recall/Context Precision）已有代码，但目前依赖真实 LLM 调用，且测试集是**硬编码**在 `scripts/run_evaluation.py`（外层 knowledge_base 目录，非本仓库）里的静态数据，**不是**基于真实检索结果的，这块在 v2 Evaluation 阶段（Step 11）需要重做数据源，不能继续沿用那个"拍脑袋"版本

## 10. v2 最佳切入点

1. **`processor/query_processor/main_graph_v2.py` 已存在的 Router/Knowledge/WebSearch/Synthesizer 四 Agent 雏形**——v2 的 `agent_processor` 应该在此基础上重命名/扩展成 `Triage`/`Retrieval`/`Diagnosis`/`Tool` 四个角色，而不是从零写；这是当前代码里**离"Agentic"最近的一块资产**，复用成本最低
2. **`eval/retriever.py` 已经验证过的"无侵入复用业务节点"模式**（直接 import `NodeRrf._rrf_merge`）——这是抽 `RetrievalService` 时最安全的落地方式：先做一个"门面"（facade）类把四个节点的调用序列封装成 `RetrievalService.search()`，**先不动节点内部代码**，跑通 `eval/retriever.py` 同样的用例作为回归基线，再在下一小步（Step 2 的后半程）让节点内部反过来调用这个 Service（避免双向改动），这是典型的"strangler fig"渐进重构路径，风险最低
3. **`main_graph_v2.py` 里的 `AgentKnowledge` 已经在用 `ThreadPoolExecutor` 并行调 `NodeSearchEmbedding`/`NodeSearchEmbeddingHyde`**——这个模式（LangGraph 节点内部再做一次线程池并发）值得保留，v2 `Retrieval Agent` 沿用即可，不需要重新设计并发模型

## 11. 首先抽取哪个 Service，为什么

**结论：先抽 `RetrievalService`，封装对象是"向量检索 + HyDE + RRF + Rerank + 断崖截断"这条完整链路（不含 LLM 答案生成，不含 WebSearch）。**

理由：
- 这是"最容易被 Agent 复用"的一块：无论后续是 Triage Agent 决定"要不要补充检索一次"，还是 Diagnosis Agent 决定"再查一次维修手册"，调用的都是同一个"给我 query + 商品名，返回 top 证据列表"的能力，抽象成本最小、复用面最大
- `main_graph_v2.py` 里 `AgentKnowledge` 已经证明了这个组合是"可独立调用"的（它不依赖答案生成节点，只产出 `rrf_chunks`），说明抽出来不影响现有 v1 的 `node_answer_output` 消费方式
- 有现成的"消费者"：`eval/retriever.py`（评测）、`main_graph_v2.py`（Agent 雏形）、未来的 `search_knowledge_base` Tool 三个调用方都在等着这个接口，先建好它，Step 4 的 RAG Tool 直接就是"把 `RetrievalService.search()` 包一层 LangGraph tool 定义"，几乎零额外设计成本
- WebSearch（MCP）暂时**不**并进 `RetrievalService`，保持 `WebSearchService` 独立，原因是它依赖外部 MCP 网络调用、有独立的失败/重试语义，混进同一个 Service 会让"纯本地检索"的回归测试变脏（本地检索应该是可离线可复现的，WebSearch 不应该污染这个特性）

### 建议的接口形态（Step 2 具体实现前先定死，避免反复）

```python
class RetrievalService:
    def search(
        self,
        query: str,
        product_model: Optional[str] = None,   # 对应现在 item_names 里的标准化名称
        strategy: Literal["hybrid", "hybrid_rrf", "hybrid_rrf_rerank", "hyde_hybrid_rrf_rerank"] = "hybrid_rrf_rerank",
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        """
        统一检索入口。strategy 直接复用 eval/retriever.py 里已经验证过的4种策略命名，
        保证"eval 用什么策略名，v2 Agent 调 Tool 就用什么策略名"，不需要维护两套词汇。
        """
```

`RetrievedChunk` 是一个纯 dataclass/pydantic 模型（`chunk_id`/`content`/`item_name`/`file_title`/`score`/`source`/`url`），不直接返回 MilvusClient 的原始 hit 对象，隔离底层实现。

## 12. 本次审查发现的既有问题清单（记录在案，Step 1 冻结前逐项确认，不擅自修改）

> **v1.0.0 最终状态说明**：下表第 1、2 项已在 **v1.0.0-rc1** 修复（`delete_message` 已实现、`import_service` 已加 `/health`、CI health 检查已修正），仅作为审计记录保留。第 3、4 项**未修**，留待 v2 Step 2/4 处理。

1. ✅ **已修（rc1）** `web/api/query_service.py` 的 `DELETE /message/{message_id}` 原本调用了未定义的 `delete_message()`（会 500）。rc1 在 `utils/mongo_history_utils.py` 实现了 `delete_message(message_id)`，行为与现有 `save_chat_message`/`update_message_item_names` 一致（字符串 message_id → ObjectId，异常吞掉返回 0）
2. ✅ **已修（rc1）** `web/api/import_service.py` 原本没有 `/health`，CI 里 `curl http://localhost:8000/health` 打不到对应路由。rc1 已为 import_service 新增 `GET /health`，并把 CI 的 docker health 检查改为轮询等待两个服务
3. `node_search_embedding.py`/`node_search_embedding_hyde.py` 里 `expr = f'item_name in {item_names}'` 未做字符串转义（与 `eval/retriever.py` 里已有的 `escape_milvus_string` 写法不一致），是已知隐患但**不在 rc1/v1.0.0 范围**，记录给 v2 Step 2/4 处理
4. 仓库里大量节点/工具文件末尾都有 `if __name__ == "__main__"` 的手动测试代码（每个节点文件都自带一份 mock 数据 + 打印逻辑），这是开发习惯不是 bug，但 v2 新增文件时应避免再延续"每个节点文件都带一遍可运行 demo"的写法，改为统一放到 `tests/` 目录

> **配置类修复（rc1 同批，属"修 key 错配"，不碰 Rerank/MCP 检索逻辑）**：`config/reranker_config.py` 改为优先读 `RERANK_API_KEY`（DashScope key），`config/bailian_mcp_config.py` 改为优先读 `MCP_API_KEY`（DashScope key），两者均回退 `OPENAI_API_KEY`（智谱 key，供 LLM）。此前两个 config 都读 `OPENAI_API_KEY`，导致 Rerank/MCP 拿智谱 key 打 DashScope 端点 401。

---

**Step 0 结论**：v1 调用链、模块边界、数据模型、评测/测试机制已全部摸清；`RetrievalService` 是首选抽取目标，`main_graph_v2.py` 的现有 Agent 雏形是 v2 的起点，两者都具备"最小改动、最大复用"的条件。等待用户确认进入 Step 1。
