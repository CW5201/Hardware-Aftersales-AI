"""
processor/agent_processor/main_graph.py

v2 Agent 工作流（Step 5）。

    START → triage → retrieval → diagnosis → END

- triage:     结构化意图识别（Step 2）
- retrieval:  调 search_knowledge_base Tool → RetrievalService → v1 检索（Step 4）
- diagnosis:  基于 triage + evidence 做结构化诊断（Step 5）

本阶段不接 Memory / Ticket / HITL。
"""

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from processor.agent_processor.nodes.triage import TriageNode
from processor.agent_processor.state import AgentState
from tool.logger import logger


class AgentWorkflow:
    """v2 Agent 工作流（当前 = Triage → Retrieval → Diagnosis）。"""

    def __init__(self, llm=None, retrieval_tool=None, retrieval_service=None, diagnosis_llm=None):
        self._triage = TriageNode(llm=llm)
        # 延迟 import：RetrievalNode 会经 search_knowledge_base → core.retrieval →
        # utils.embedding_utils 间接拉入 FlagEmbedding 的 C 扩展。
        # 保持 main_graph 顶层不触碰该 C 扩展，避免 pytest assertion-rewriting
        # 下同一 C 扩展二次 exec 导致 segfault（Python 3.14 限制）。
        from processor.agent_processor.nodes.retrieval import RetrievalNode
        from processor.agent_processor.nodes.diagnosis import DiagnosisNode

        # 默认用全局共享的 search_knowledge_base Tool（底层 RetrievalService 单例）
        self._retrieval = RetrievalNode(tool=retrieval_tool, service=retrieval_service)
        self._diagnosis = DiagnosisNode(llm=diagnosis_llm if diagnosis_llm is not None else llm)
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("triage", self._triage)
        graph.add_node("retrieval", self._retrieval)
        graph.add_node("diagnosis", self._diagnosis)
        graph.add_edge(START, "triage")
        graph.add_edge("triage", "retrieval")
        graph.add_edge("retrieval", "diagnosis")
        graph.add_edge("diagnosis", END)
        return graph.compile()

    def run(self, user_query: str) -> AgentState:
        """执行 Triage + Retrieval + Diagnosis，返回含 triage / retrieval / diagnosis 的状态。"""
        initial_state: AgentState = {"user_query": user_query}
        result = self._graph.invoke(initial_state)
        diagnosis = result.get("diagnosis")
        retrieval = result.get("retrieval") or {}
        logger.info(
            f"AgentWorkflow(Triage→Retrieval→Diagnosis) 完成: "
            f"diagnosis_status={getattr(diagnosis, 'diagnosis_status', None)}, "
            f"retrieval_docs={len(retrieval.get('documents', []))}, "
            f"retrieval_error={retrieval.get('error')}"
        )
        return result


if __name__ == "__main__":
    wf = AgentWorkflow()
    state = wf.run("我的X200出现ERR-203，重启之后还是报错")
    print(state["triage"])
    print(state["retrieval"])
    print(state["diagnosis"])
