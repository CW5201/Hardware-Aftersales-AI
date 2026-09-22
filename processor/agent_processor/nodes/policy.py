"""
processor/agent_processor/nodes/policy.py

Human-in-the-loop（Step 9）：Policy 节点 + LangGraph interrupt/resume。

流程（高风险 WRITE Tool）：
    Agent 决策调 WRITE Tool
      ↓
    PolicyNode：Policy.requires_approval(tool) == True
      ↓
    生成 ApprovalRecord（pending）→ 写 state["approval"]
      ↓
    LangGraph interrupt(payload) → 图挂起（saver 持久化到 pending）
      ↓
    人工在审批页看到 {设备, 客户, 问题, Diagnosis, Evidence, Tool, 参数}
      ↓
    approve / reject（POST /api/v2/approvals/{id}/approve|reject）
      ↓
    Command(resume="approved"|"rejected") → 图恢复
      ↓
    approved → ApprovalService.execute()（幂等）→ 写 state["tool_result"]
    rejected  → 不执行，写 state["tool_result"]["rejected"]=True

READ Tool 不过 Policy（直接执行）。

本节点设计为"可独立注入审批服务"：不持有全局单例，便于测试隔离。
"""

from langgraph.types import interrupt

from processor.agent_processor.state import AgentState
from services.business.approval_service import ApprovalService
from tool.logger import logger


class PolicyNode:
    """
    LangGraph 节点：policy 判定 + interrupt + resume + 执行。

    用法（Graph 里）：
        graph.add_node("policy", PolicyNode(approval_service, business_service))

    输入 state 需含：
      - tool_name / tool_args（本轮要执行的 Tool）
      - customer_id / device_id / problem / diagnosis / evidence（审批展示上下文）
      - thread_id / run_id（审批记录关联）

    输出 state 追加：
      - approval: {approval_id, status, ...}
      - tool_result: {executed, deduplicated, result, error} 或 {rejected: True}

    幂等说明：create_approval 是"记录型"操作（不产生业务副作用，不幂等保护）。
    interrupt 挂起期间图持久化到 checkpointer；resume 时 policy 节点**重放**，
    重放前必须清理本节点上次遗留的 pending 审批记录，否则 execute() 会因
    "approval 仍是 pending" 而拒绝执行。
    """

    def __init__(self, approval_service: ApprovalService, business_service=None):
        self._approval = approval_service
        self._biz = business_service or approval_service._svc

    def __call__(self, state: AgentState) -> AgentState:
        tool_name = state.get("tool_name")
        tool_args = state.get("tool_args") or {}

        # ---------- 不需要审批（READ 或无 tool）→ 直接透传 ----------
        if not tool_name or not self._approval.requires_approval(tool_name, tool_args):
            logger.info(f"PolicyNode: {tool_name!r} 不需要人工审批，直接放行")
            return {
                **state,
                "approval": {
                    "approval_id": None,
                    "status": "auto",
                    "tool_name": tool_name,
                    "reason": "read_only_or_unknown_tool",
                },
            }

        # ---------- 幂等防重放：按 run_id+tool+idempotency_key 找上次遗留的 pending ----------
        # resume 时 policy 重放；初次 pass 创建的审批记录（同 run_id+tool+idem）仍在 pending，
        # 需让它成为"被复用"的那条，保证 execute() 作用在人工已决策的记录上。
        run_key = state.get("run_id") or state.get("thread_id")
        idem_key = tool_args.get("idempotency_key")
        prior_id = None
        for a in self._biz.store.approvals:
            if (
                a.requester == run_key
                and a.tool_name == tool_name
                and a.idempotency_key == idem_key
                and a.status in ("pending", "approved", "superseded")
            ):
                prior_id = a.approval_id
                break

        # ---------- 需要审批：生成 / 复用 approval + 上下文 ----------
        context = {
            "device_id": state.get("device_id") or tool_args.get("device_id"),
            "customer_id": state.get("customer_id"),
            "problem": state.get("problem") or tool_args.get("problem"),
            "diagnosis": (
                state.get("diagnosis").model_dump()
                if hasattr(state.get("diagnosis"), "model_dump")
                else state.get("diagnosis")
            ),
            "evidence": (state.get("retrieval") or {}).get("documents", [])[:5],
            "tool_name": tool_name,
            "tool_args": tool_args,
        }
        approval_record = self._approval.create_approval(
            tool_name=tool_name,
            tool_args=tool_args,
            run_id=run_key,
            context=context,
        )
        # 若找到上次遗留的 pending 记录，优先用它（人工决策作用在那条上）
        if prior_id is not None:
            approval_record["approval_id"] = prior_id

        # ---------- interrupt：图挂起，等人工决策 ----------
        # payload 供前端审批页展示；resume 值 = "approved" / "rejected"
        resume_value = interrupt(
            {
                "kind": "human_approval_required",
                "approval_id": approval_record["approval_id"],
                "tool_name": tool_name,
                "context": context,
            }
        )

        # ---------- resume 后按决策执行 ----------
        if resume_value == "approved":
            exec_res = self._approval.execute(approval_record["approval_id"])
            return {
                **state,
                "approval": {**approval_record, "status": "approved"},
                "tool_result": exec_res,
            }

        # rejected（或其他非法值保守按 rejected 处理）
        logger.info(f"PolicyNode: 审批 {approval_record['approval_id']} 被驳回，不执行 {tool_name}")
        return {
            **state,
            "approval": {**approval_record, "status": "rejected"},
            "tool_result": {"executed": False, "rejected": True, "result": None, "error": None},
        }
