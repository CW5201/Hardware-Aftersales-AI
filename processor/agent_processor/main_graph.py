"""
processor/agent_processor/main_graph.py

v2 Agent 工作流（Step 7）。

    START → triage → memory_retrieve → retrieval → diagnosis → memory_persist → END

- triage:         结构化意图识别（Step 2）
- memory_retrieve: 注入 thread 短期 + customer/device 长期记忆（Step 7）
- retrieval:      调 search_knowledge_base Tool → RetrievalService → v1 检索（Step 4）
- diagnosis:      基于 triage + evidence 做结构化诊断（Step 5）
- memory_persist:  把本轮中间态写回短期 / 关键事件写长期（Step 7）

本阶段不接 Ticket / HITL（Step 8/9）。
"""

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from processor.agent_processor.nodes.triage import TriageNode
from processor.agent_processor.state import AgentState
from tool.logger import logger


class AgentWorkflow:
    """v2 Agent 工作流（当前 = Triage → Memory → Retrieval → Diagnosis → Memory）。"""

    def __init__(
        self,
        llm=None,
        retrieval_tool=None,
        retrieval_service=None,
        diagnosis_llm=None,
        memory=None,
    ):
        self._triage = TriageNode(llm=llm)
        # 延迟 import：RetrievalNode 会经 search_knowledge_base → core.retrieval →
        # utils.embedding_utils 间接拉入 FlagEmbedding 的 C 扩展。
        # 保持 main_graph 顶层不触碰该 C 扩展，避免 pytest assertion-rewriting
        # 下同一 C 扩展二次 exec 导致 segfault（Python 3.14 限制）。
        from processor.agent_processor.nodes.retrieval import RetrievalNode
        from processor.agent_processor.nodes.diagnosis import DiagnosisNode
        from processor.agent_processor.nodes.memory import MemoryPersistNode, MemoryRetrieveNode
        from services.memory.memory_service import get_memory_service

        # 默认用全局共享的 search_knowledge_base Tool（底层 RetrievalService 单例）
        self._retrieval = RetrievalNode(tool=retrieval_tool, service=retrieval_service)
        self._diagnosis = DiagnosisNode(llm=diagnosis_llm if diagnosis_llm is not None else llm)
        mem = memory if memory is not None else get_memory_service()
        self._memory_retrieve = MemoryRetrieveNode(mem)
        self._memory_persist = MemoryPersistNode(mem)
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("triage", self._triage)
        graph.add_node("memory_retrieve", self._memory_retrieve)
        graph.add_node("retrieval", self._retrieval)
        graph.add_node("diagnosis", self._diagnosis)
        graph.add_node("memory_persist", self._memory_persist)
        graph.add_edge(START, "triage")
        graph.add_edge("triage", "memory_retrieve")
        graph.add_edge("memory_retrieve", "retrieval")
        graph.add_edge("retrieval", "diagnosis")
        graph.add_edge("diagnosis", "memory_persist")
        graph.add_edge("memory_persist", END)
        return graph.compile()

    def run(
        self,
        user_query: str,
        thread_id: str = "default",
        customer_id: str | None = None,
        device_id: str | None = None,
    ) -> AgentState:
        """
        执行完整 Graph（Triage → Memory → Retrieval → Diagnosis → Memory）。

        :param thread_id: 会话线程 ID（短期记忆锚点）
        :param customer_id / device_id: 数据隔离锚点（长期记忆 + 业务 Tool 作用域）
        """
        initial_state: AgentState = {
            "user_query": user_query,
            "thread_id": thread_id,
        }
        if customer_id:
            initial_state["customer_id"] = customer_id
        if device_id:
            initial_state["device_id"] = device_id

        result = self._graph.invoke(initial_state)
        diagnosis = result.get("diagnosis")
        retrieval = result.get("retrieval") or {}
        logger.info(
            f"AgentWorkflow(→Diagnosis→Memory) 完成: "
            f"diagnosis_status={getattr(diagnosis, 'diagnosis_status', None)}, "
            f"retrieval_docs={len(retrieval.get('documents', []))}, "
            f"retrieval_error={retrieval.get('error')}"
        )
        return result


if __name__ == "__main__":
    wf = AgentWorkflow()
    state = wf.run("我的X200出现ERR-203，重启之后还是报错", thread_id="t-demo", customer_id="CUST-0001")
    print(state["triage"])
    print(state["retrieval"])
    print(state["diagnosis"])
    print(state.get("memory", {}).get("long_term", {}).get("devices"))
