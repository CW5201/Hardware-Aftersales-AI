"""
processor/agent_processor/hitl_graph.py

Human-in-the-loop Graph（Step 9）：在 Policy 节点用 LangGraph interrupt()
挂起，等人工 approve/reject 后 resume。

与主图（main_graph.AgentWorkflow）的区别：
  - 必须配 checkpointer（InMemorySaver）才能持久化 pending 状态
  - 末尾加 policy 节点：高风险 WRITE Tool → interrupt → 人工决策 → 执行
  - 提供 run_hitl() 便捷方法：invoke 到 interrupt 或 END；
    若挂起则暴露 approval_id + thread，调用方人工决策后调 resume()

用法（最小）：
    wf = HitlWorkflow()
    out = wf.run_hitl("X200 ERR-203", customer_id="CUST-0001",
                      tool_name="create_service_ticket",
                      tool_args={"customer_id": "CUST-0001", "problem": "ERR-203",
                                  "idempotency_key": "k1"})
    if out["interrupted"]:
        wf.approve(out["approval_id"])   # 人工决策（走 ApprovalService）
        out2 = wf.resume(out["thread_id"], "approved")
"""

import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Command

from processor.agent_processor.nodes.policy import PolicyNode
from processor.agent_processor.state import AgentState
from services.business.approval_service import ApprovalService
from services.business.business_service import BusinessService
from tool.logger import logger


class HitlWorkflow:
    """Triage → ... → Policy(interrupt) 的 HITL Graph（最小可用版）。

    本阶段 policy 作为末端节点：前面可复用主图节点，但为聚焦 Step 9 契约，
    这里只接 policy（输入 state 已由上游填好 tool_name/tool_args）。
    生产完整链路在 Step 13 集成（Adaptive Retrieval → Tool 决策 → policy）。
    """

    def __init__(
        self,
        approval_service: ApprovalService = None,
        business_service: BusinessService = None,
    ):
        biz = business_service or BusinessService()
        self._approval = approval_service or ApprovalService(biz)
        self._policy = PolicyNode(self._approval, biz)
        graph = StateGraph(AgentState)
        graph.add_node("policy", self._policy)
        graph.add_edge(START, "policy")
        graph.add_edge("policy", END)
        self._saver = InMemorySaver()
        self._graph = graph.compile(checkpointer=self._saver)

    # ---------------- 相位 1：invoke 到 interrupt / END ----------------

    def _cfg(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def start(
        self,
        user_query: str,
        tool_name: str,
        tool_args: dict,
        customer_id: str = "",
        device_id: str = "",
        thread_id: str = None,
        run_id: str = None,
        diagnosis=None,
        problem: str = "",
    ) -> dict:
        """
        首次 invoke：生成 run_id（稳定，resume 复用），跑图到 interrupt 或 END。

        返回 {interrupted, thread_id, run_id, approval_id?}
        """
        thread_id = thread_id or f"hitl-{uuid.uuid4().hex[:8]}"
        run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
        state: AgentState = {
            "user_query": user_query,
            "thread_id": thread_id,
            "run_id": run_id,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "problem": problem,
        }
        if customer_id:
            state["customer_id"] = customer_id
        if device_id:
            state["device_id"] = device_id
        if diagnosis is not None:
            state["diagnosis"] = diagnosis

        self._graph.invoke(state, config=self._cfg(thread_id))
        snap = self._graph.get_state(self._cfg(thread_id))
        interrupted = bool(snap.tasks and any(getattr(t, "interrupts", None) for t in snap.tasks))
        approval_id = None
        if interrupted:
            # 取 policy 刚创建的 pending 审批（同 run_id + tool_name + idem_key）
            idem = tool_args.get("idempotency_key")
            for a in reversed(self._approval._svc.store.approvals):
                if a.status == "pending" and a.requester == run_id:
                    approval_id = a.approval_id
                    break
        return {
            "interrupted": interrupted,
            "thread_id": thread_id,
            "run_id": run_id,
            "approval_id": approval_id,
        }

    # 兼容旧调用名
    def run_hitl(self, *args, **kwargs) -> dict:
        return self.start(*args, **kwargs)

    # ---------------- 人工决策 + resume（相位 2） ----------------

    def approve(self, approval_id: str, decided_by: str = "human") -> dict:
        return self._approval.approve(approval_id, decided_by=decided_by)

    def reject(self, approval_id: str, decided_by: str = "human") -> dict:
        return self._approval.reject(approval_id, decided_by=decided_by)

    def resume(self, thread_id: str, run_id: str, decision: str) -> dict:
        """
        人工决策后恢复图执行。

        :param decision: "approved" | "rejected"
        :param run_id: 必须与 start() 返回的 run_id 一致（幂等复用同一条审批记录）
        返回 {thread_id, run_id, state, decision}。
        """
        result = self._graph.invoke(
            Command(resume=decision),
            config=self._cfg(thread_id),
        )
        return {
            "thread_id": thread_id,
            "run_id": run_id,
            "state": result,
            "decision": decision,
        }


if __name__ == "__main__":
    wf = HitlWorkflow()
    out = wf.run_hitl(
        "X200 ERR-203",
        tool_name="create_service_ticket",
        tool_args={"customer_id": "CUST-0001", "problem": "ERR-203", "idempotency_key": "demo1"},
        customer_id="CUST-0001",
        device_id="DEV-X200-001",
        problem="ERR-203 供电电压异常",
    )
    print(out)
    if out["interrupted"]:
        wf.approve(out["approval_id"])
        print(wf.resume(out["thread_id"], "approved"))
