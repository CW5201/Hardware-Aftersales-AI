"""
processor/agent_processor/traced_graph.py

TracedAgentWorkflow（Step 10）：带 Agent Trace 记录的完整主图。

在 main_graph.AgentWorkflow（Triage → Memory → Retrieval → Diagnosis → Memory）
的同一套节点上，加 trace 记录：
  - run 级：run_id / thread_id / 总耗时 / token 估算 / 终态
  - 节点级：Triage / MemoryRetrieve / Retrieval / Diagnosis / MemoryPersist，
    每个节点记录 input / output / latency / token（估算或 LLM usage）/ error

Trace API（web/api/v2_service.py）：
    GET /api/v2/threads/{thread_id}/trace   → 该 thread 的所有 run 摘要 + 事件
    GET /api/v2/runs/{run_id}              → 单次 run 全量事件

Trace 记录走 InMemoryTraceStore（进程内，可替换为 PostgreSQL / MongoDB）。
"""

import time
import uuid

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from processor.agent_processor.nodes.diagnosis import DiagnosisNode
from processor.agent_processor.nodes.memory import MemoryPersistNode, MemoryRetrieveNode
from processor.agent_processor.nodes.triage import TriageNode
from processor.agent_processor.state import AgentState
from services.memory.memory_service import MemoryService, get_memory_service
from services.trace.trace_store import (
    InMemoryTraceStore,
    TokenUsage,
    estimate_tokens,
    get_trace_store,
)
from tool.logger import logger


class TracedAgentWorkflow:
    """带 trace 记录的 v2 Agent 工作流（Step 10 完整主图）。"""

    def __init__(
        self,
        llm=None,
        retrieval_tool=None,
        retrieval_service=None,
        diagnosis_llm=None,
        memory: MemoryService = None,
        trace_store: InMemoryTraceStore = None,
    ):
        # 延迟 import（C 扩展保护，同 main_graph）
        from processor.agent_processor.nodes.retrieval import RetrievalNode

        self._triage = TriageNode(llm=llm)
        self._retrieval = RetrievalNode(tool=retrieval_tool, service=retrieval_service)
        self._diagnosis = DiagnosisNode(llm=diagnosis_llm if diagnosis_llm is not None else llm)
        mem = memory if memory is not None else get_memory_service()
        self._memory_retrieve = MemoryRetrieveNode(mem)
        self._memory_persist = MemoryPersistNode(mem)
        self._trace = trace_store if trace_store is not None else get_trace_store()
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("triage", self._wrap_node("triage", self._triage, "triage_agent"))
        graph.add_node("memory_retrieve", self._wrap_node("memory_retrieve", self._memory_retrieve, "memory"))
        graph.add_node("retrieval", self._wrap_node("retrieval", self._retrieval, "retrieval_agent"))
        graph.add_node("diagnosis", self._wrap_node("diagnosis", self._diagnosis, "diagnosis_agent"))
        graph.add_node("memory_persist", self._wrap_node("memory_persist", self._memory_persist, "memory"))
        graph.add_edge(START, "triage")
        graph.add_edge("triage", "memory_retrieve")
        graph.add_edge("memory_retrieve", "retrieval")
        graph.add_edge("retrieval", "diagnosis")
        graph.add_edge("diagnosis", "memory_persist")
        graph.add_edge("memory_persist", END)
        return graph.compile()

    # ---------------- 节点 trace 包装 ----------------

    def _wrap_node(self, name: str, node, node_type: str = "agent") -> callable:
        """
        把节点包成"跑完记一条 trace"的节点。
        记录 input/output/latency/token（估算）/error/status。
        """

        def _run(state: AgentState) -> AgentState:
            t0 = time.time()
            try:
                out = node(state)
                latency = (time.time() - t0) * 1000
                error = None
                status = "success"
            except Exception as e:  # noqa: BLE001 —— 记 error 后重抛，让 Graph 感知
                latency = (time.time() - t0) * 1000
                out = state
                error = f"{type(e).__name__}: {e}"
                status = "error"
                logger.exception(f"TracedWorkflow: 节点 {name} 异常: {e}")
                raise

            run_id = state.get("run_id")
            if run_id and self._trace.get_run(run_id):
                retrieval = out.get("retrieval") or {}
                retrieved = len(retrieval.get("documents") or []) if name == "retrieval" else None
                prompt_tokens = estimate_tokens(state.get("user_query", ""))
                out_tokens = estimate_tokens(str(out.get("diagnosis") or out.get("triage") or ""))
                self._trace.add_event(
                    run_id,
                    _make_event(
                        self._trace.get_run(run_id).run_id,
                        0,
                        node=name,
                        agent=node_type,
                        kind="node",
                        input=state.get("user_query"),
                        output=_node_output_summary(out, name),
                        retrieved_docs_count=retrieved,
                        latency_ms=round(latency, 2),
                        token_usage=TokenUsage(prompt_tokens, out_tokens),
                        error=error,
                        status=status,
                    ),
                )
            return out

        return _run

    # ---------------- run ----------------

    def run(
        self,
        user_query: str,
        thread_id: str = "default",
        customer_id: str | None = None,
        device_id: str | None = None,
    ) -> AgentState:
        """执行完整 traced 图，记录 TraceRun + 节点事件，返回 state（含 trace_run_id）。"""
        run_id = f"run-{uuid.uuid4().hex[:10]}"
        trace_run = self._trace.start_run(run_id, thread_id, user_query)

        initial_state: AgentState = {
            "user_query": user_query,
            "thread_id": thread_id,
            "run_id": run_id,
        }
        if customer_id:
            initial_state["customer_id"] = customer_id
        if device_id:
            initial_state["device_id"] = device_id

        status = "success"
        error = None
        try:
            result = self._graph.invoke(initial_state)
        except Exception as e:  # noqa: BLE001
            status = "error"
            error = f"{type(e).__name__}: {e}"
            logger.exception(f"TracedWorkflow: run 异常: {e}")
            result = initial_state
            # 兜底：把 error 写进 retrieval 标记，让下游知道是故障
            result["retrieval"] = {
                "documents": [], "scores": [], "strategy": None,
                "metadata": {}, "latency": 0.0, "error": f"agent_run_failed: {error}",
            }

        # 汇总 token：triage + diagnosis 两次 LLM 调用的 prompt/output 估算
        total_in = estimate_tokens(user_query) * 2
        total_out = estimate_tokens(str(result.get("diagnosis") or "")) + estimate_tokens(
            str(result.get("triage") or "")
        )
        trace_run.token_usage = TokenUsage(total_in, total_out)
        trace_run.finish(status=status, error=error)

        result["trace_run_id"] = run_id
        result.setdefault("retrieval", {})
        logger.info(
            f"TracedWorkflow 完成: run={run_id}, status={status}, "
            f"latency={trace_run.total_latency_ms}ms"
        )
        return result


def _make_event(run_id, seq, **kw):
    """延迟 import TraceEvent 避免循环；直接构造。"""
    from services.trace.trace_store import TraceEvent

    return TraceEvent(run_id, seq, **kw)


def _node_output_summary(state: AgentState, node_name: str) -> dict:
    """节点输出的 trace 友好摘要（避免把整个 state 塞进 trace）。"""
    if node_name == "triage":
        t = state.get("triage")
        return {"intent": getattr(t, "intent", None), "strategy": getattr(t, "retrieval_strategy", None)}
    if node_name == "retrieval":
        r = state.get("retrieval") or {}
        return {"docs": len(r.get("documents") or []), "error": r.get("error")}
    if node_name == "diagnosis":
        d = state.get("diagnosis")
        return {"status": getattr(d, "diagnosis_status", None), "need_human_review": getattr(d, "need_human_review", None)}
    if node_name in ("memory_retrieve", "memory_persist"):
        m = state.get("memory") or {}
        lt = m.get("long_term") or {}
        return {"devices": len(lt.get("devices") or []), "faults": len(lt.get("faults") or [])}
    return {}


if __name__ == "__main__":
    wf = TracedAgentWorkflow()
    s = wf.run("我的X200出现ERR-203，重启之后还是报错", thread_id="trace-demo", customer_id="CUST-0001")
    print("trace_run_id:", s.get("trace_run_id"))
    run = get_trace_store().get_run(s["trace_run_id"])
    print("events:", [e.node for e in run.events])
