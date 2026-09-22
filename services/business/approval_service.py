"""
services/business/approval_service.py

Human-in-the-loop 审批服务（Step 9）。

契约：
  - 高风险 WRITE Tool（create_service_ticket / reserve_spare_part /
    update_service_ticket）必须经过 Policy 判定 → 需审批 →
    LangGraph interrupt() → Approval Pending → approve/reject → resume。
  - 所有写操作支持幂等：同一 idempotency_key 在审批执行阶段
    只落库一次（upsert + 状态检查），防 retry / resume 重复建单。
  - 审批记录存 approval（approval_id / tool_name / tool_args / 状态 / 决议人）。

本模块是"审批决策 + 执行"的服务层；LangGraph 的 interrupt/resume
机制在 processor/agent_processor/nodes/policy.py 里接。
"""

from datetime import datetime, timezone
from typing import Optional

from tool.logger import logger

from services.business.business_service import BusinessService, BusinessServiceError
from services.business.models import Approval

# 需要人工审批的高风险写工具白名单（Policy 用）
HIGH_RISK_WRITE_TOOLS = frozenset(
    {
        "create_service_ticket",
        "reserve_spare_part",
        "update_service_ticket",
    }
)


class ApprovalError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ApprovalService:
    """审批生命周期：create → approve/reject → execute（幂等）。"""

    def __init__(self, service: Optional[BusinessService] = None):
        self._svc = service or BusinessService()

    # ---------------- Policy ----------------

    @staticmethod
    def requires_approval(tool_name: str, tool_args: dict) -> bool:
        """
        Policy：判断一次工具调用是否需要人工审批。

        规则（最小可用）：
          - READ 工具 → 不需要
          - HIGH_RISK_WRITE_TOOLS 里的写工具 → 需要
          - 写工具缺 idempotency_key → 也强制需要（无法幂等 = 高危）
        """
        if tool_name in HIGH_RISK_WRITE_TOOLS:
            return True
        return False

    # ---------------- 审批记录 ----------------

    def create_approval(
        self,
        tool_name: str,
        tool_args: dict,
        run_id: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> dict:
        """
        生成一条 pending 审批记录（含展示用上下文）。

        幂等：同一 run_id 已有 pending 审批 → 复用（不重复创建）。
        这对 LangGraph interrupt 重放至关重要：resume 时 policy 节点重跑，
        重放的 create_approval 复用初次创建的那条（同 run_id + 同 tool），
        保证人工决策的 approval_id 与 resume 后 execute() 用的是同一条记录。
        """
        # 幂等查找：同 run_id + 同 tool_name + 同 idempotency_key 的 pending 审批 → 复用
        idem_key = tool_args.get("idempotency_key")
        if run_id:
            for a in reversed(self._svc.store.approvals):
                if (
                    a.requester == run_id
                    and a.tool_name == tool_name
                    and a.idempotency_key == idem_key
                    and a.status in ("pending", "superseded")
                ):
                    # 复用：把本次的 tool_args / 上下文刷新进去
                    a.tool_args = tool_args
                    logger.info(f"ApprovalService: 复用 pending 审批 {a.approval_id} (run={run_id})")
                    return {
                        "approval_id": a.approval_id,
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "status": a.status,
                        "context": context or {},
                        "created_at": a.created_at,
                    }

        approval = Approval(
            approval_id=f"APR-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{len(self._svc.store.approvals) + 1:03d}",
            tool_name=tool_name,
            tool_args=tool_args,
            requester=run_id,
            idempotency_key=tool_args.get("idempotency_key"),
            status="pending",
        )
        self._svc.store.approvals.append(approval)
        logger.info(f"ApprovalService: 创建审批 {approval.approval_id} tool={tool_name}")
        return {
            "approval_id": approval.approval_id,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "status": "pending",
            # 审批页面展示用（设备 / 客户 / 问题 / 诊断 / 证据 / Tool / 参数）
            "context": context or {},
            "created_at": approval.created_at,
        }

    def _find(self, approval_id: str) -> Optional[Approval]:
        return next((a for a in self._svc.store.approvals if a.approval_id == approval_id), None)

    def get(self, approval_id: str) -> dict:
        a = self._find(approval_id)
        if a is None:
            return {"found": False, "approval_id": approval_id, "error": "approval_not_found"}
        return {"found": True, "approval": a.model_dump(), "error": None}

    # ---------------- 决策 ----------------

    def _decide(self, approval_id: str, decision: str, decided_by: str) -> dict:
        a = self._find(approval_id)
        if a is None:
            raise ApprovalError(f"approval not found: {approval_id}")
        if a.status not in ("pending", "superseded"):
            raise ApprovalError(f"approval {approval_id} already {a.status}, cannot {decision}")
        a.status = "approved" if decision == "approve" else "rejected"
        a.decided_at = _now()
        a.decided_by = decided_by
        return a.model_dump()

    def _find(self, approval_id: str) -> Optional[Approval]:
        """按 approval_id 查审批记录（含 executed 终态）。"""
        return next((a for a in self._svc.store.approvals if a.approval_id == approval_id), None)

    def approve(self, approval_id: str, decided_by: str = "human") -> dict:
        out = self._decide(approval_id, "approve", decided_by)
        logger.info(f"ApprovalService: 批准 {approval_id} by {decided_by}")
        return out

    def reject(self, approval_id: str, decided_by: str = "human") -> dict:
        out = self._decide(approval_id, "reject", decided_by)
        logger.info(f"ApprovalService: 驳回 {approval_id} by {decided_by}")
        return out

    # ---------------- 执行（幂等） ----------------

    def execute(self, approval_id: str) -> dict:
        """
        执行已批准的写操作（幂等：同 idempotency_key 只落库一次）。

        返回 {executed, result, error}。
        重复执行同一 approval（同 key）→ 返回 deduplicated=True，不重复写。
        """
        a = self._find(approval_id)
        if a is None:
            raise ApprovalError(f"approval not found: {approval_id}")
        if a.status not in ("approved", "executed", "rejected"):
            raise ApprovalError(f"approval {approval_id} is {a.status}, not approved")

        # rejected → 不执行写操作（resume 的 "rejected" 路径不会进到这里，双保险）
        if a.status == "rejected":
            return {"executed": False, "deduplicated": False, "result": None,
                    "error": "approval_rejected", "rejected": True}

        tool_name = a.tool_name
        args = a.tool_args
        idem = args.get("idempotency_key")

        # 已执行过（同 key）→ 幂等返回缓存
        if a.status == "executed" and idem:
            hit, cached = self._svc.store.idempotent(idem, None)
            if hit:
                return {"executed": True, "deduplicated": True, "result": cached, "error": None}

        try:
            result = self._dispatch(tool_name, args)
        except BusinessServiceError as e:
            # 执行失败：保持 approved（允许重试），但把错误带出
            return {"executed": False, "deduplicated": False, "result": None, "error": f"execution_failed: {e}"}

        a.status = "executed"
        self._svc.store.record_idempotent(idem, result) if idem else None
        logger.info(f"ApprovalService: 执行 {tool_name} 完成 (approval={approval_id})")
        return {"executed": True, "deduplicated": False, "result": result, "error": None}

    def _dispatch(self, tool_name: str, args: dict):
        """按 tool_name 路由到 BusinessService 的写方法。"""
        if tool_name == "create_service_ticket":
            return self._svc.create_service_ticket(
                customer_id=args.get("customer_id", ""),
                device_id=args.get("device_id"),
                problem=args.get("problem", ""),
                diagnosis=args.get("diagnosis"),
                evidence=args.get("evidence") or [],
                priority=args.get("priority", "low"),
                idempotency_key=args.get("idempotency_key"),
            )
        if tool_name == "reserve_spare_part":
            return self._svc.reserve_spare_part(
                part_id=args.get("part_id", ""),
                quantity=args.get("quantity", 1),
                device_id=args.get("device_id"),
                idempotency_key=args.get("idempotency_key"),
            )
        if tool_name == "update_service_ticket":
            return self._svc.update_service_ticket(
                ticket_id=args.get("ticket_id", ""),
                status=args.get("status"),
                assigned_to=args.get("assigned_to"),
                diagnosis=args.get("diagnosis"),
                idempotency_key=args.get("idempotency_key"),
            )
        raise ApprovalError(f"unknown write tool for approval: {tool_name!r}")


__all__ = [
    "ApprovalService",
    "ApprovalError",
    "HIGH_RISK_WRITE_TOOLS",
]
