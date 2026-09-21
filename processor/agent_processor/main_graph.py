"""
processor/agent_processor/main_graph.py

v2 Agent 工作流（最小 Graph，Step 2）。

    START → triage → END

本阶段只挂 Triage 一个节点，不接 Retrieval / Tool / Diagnosis / Ticket。
后续阶段在同一张图上增量加节点，不重写本图。
"""

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from processor.agent_processor.nodes.triage import TriageNode
from processor.agent_processor.state import AgentState
from tool.logger import logger


class AgentWorkflow:
    """v2 Agent 工作流（当前 = Triage only）。"""

    def __init__(self, llm=None):
        self._triage = TriageNode(llm=llm)
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("triage", self._triage)
        graph.add_edge(START, "triage")
        graph.add_edge("triage", END)
        return graph.compile()

    def run(self, user_query: str) -> AgentState:
        """执行 Triage，返回含 TriageResult 的状态。"""
        initial_state: AgentState = {"user_query": user_query}
        result = self._graph.invoke(initial_state)
        logger.info(f"AgentWorkflow(Triage) 完成: triage={result.get('triage')}")
        return result


if __name__ == "__main__":
    wf = AgentWorkflow()
    state = wf.run("我的X200出现ERR-203，重启之后还是报错")
    print(state["triage"])
