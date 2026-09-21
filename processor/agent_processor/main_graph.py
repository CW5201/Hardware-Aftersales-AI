"""
processor/agent_processor/main_graph.py

v2 Agent 工作流（Step 4）。

    START → triage → retrieval → END

- triage:    结构化意图识别（Step 2）
- retrieval: 调 search_knowledge_base Tool → RetrievalService → v1 检索（Step 4）

本阶段只接 Triage + Retrieval，不接 Diagnosis / Memory / Ticket / HITL。
"""

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from processor.agent_processor.nodes.triage import TriageNode
from processor.agent_processor.state import AgentState
from tool.logger import logger


class AgentWorkflow:
    """v2 Agent 工作流（当前 = Triage → Retrieval）。"""

    def __init__(self, llm=None, retrieval_tool=None, retrieval_service=None):
        self._triage = TriageNode(llm=llm)
        # 延迟 import：RetrievalNode 会经 search_knowledge_base → core.retrieval →
        # utils.embedding_utils 间接拉入 FlagEmbedding 的 C 扩展。
        # 保持 main_graph 顶层不触碰该 C 扩展，避免 pytest assertion-rewriting
        # 下同一 C 扩展二次 exec 导致 segfault（Python 3.14 限制）。
        from processor.agent_processor.nodes.retrieval import RetrievalNode

        # 默认用全局共享的 search_knowledge_base Tool（底层 RetrievalService 单例）
        self._retrieval = RetrievalNode(tool=retrieval_tool, service=retrieval_service)
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("triage", self._triage)
        graph.add_node("retrieval", self._retrieval)
        graph.add_edge(START, "triage")
        graph.add_edge("triage", "retrieval")
        graph.add_edge("retrieval", END)
        return graph.compile()

    def run(self, user_query: str) -> AgentState:
        """执行 Triage + Retrieval，返回含 triage 与 retrieval 的状态。"""
        initial_state: AgentState = {"user_query": user_query}
        result = self._graph.invoke(initial_state)
        logger.info(
            f"AgentWorkflow(Triage→Retrieval) 完成: "
            f"triage={result.get('triage')}, "
            f"retrieval_docs={len((result.get('retrieval') or {}).get('documents', []))}, "
            f"retrieval_error={(result.get('retrieval') or {}).get('error')}"
        )
        return result


if __name__ == "__main__":
    wf = AgentWorkflow()
    state = wf.run("我的X200出现ERR-203，重启之后还是报错")
    print(state["triage"])
    print(state["retrieval"])
