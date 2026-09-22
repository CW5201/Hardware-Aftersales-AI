"""
services/business/ticket_workflow.py

Ticket Workflow（Step 8）：工单状态机 + 事件流。

状态机：
    PENDING → IN_PROGRESS → WAITING_APPROVAL → COMPLETED → CLOSED
                ↖____________|

规则（合法迁移表）：
    PENDING:          → IN_PROGRESS / CLOSED
    IN_PROGRESS:      → WAITING_APPROVAL / COMPLETED / CLOSED
    WAITING_APPROVAL: → IN_PROGRESS / COMPLETED / CLOSED
    COMPLETED:        → CLOSED
    CLOSED:           → (终态，不可再迁移)

所有迁移写 ticket_events（ticket_id 维度可回放）。
"""

from datetime import datetime, timezone
from typing import Optional

from tool.logger import logger

from services.business.business_service import BusinessService, BusinessServiceError
from services.business.models import TicketEvent

VALID_STATUSES = ("PENDING", "IN_PROGRESS", "WAITING_APPROVAL", "COMPLETED", "CLOSED")

# 合法迁移表：from_status → allowed to_statuses
ALLOWED_TRANSITIONS: dict[str, tuple] = {
    "PENDING": ("IN_PROGRESS", "CLOSED"),
    "IN_PROGRESS": ("WAITING_APPROVAL", "COMPLETED", "CLOSED"),
    "WAITING_APPROVAL": ("IN_PROGRESS", "COMPLETED", "CLOSED"),
    "COMPLETED": ("CLOSED",),
    "CLOSED": (),
}


class TicketWorkflowError(BusinessServiceError):
    pass


class TicketWorkflow:
    """工单状态机（经 BusinessService 读/写，不直接碰 store 内部）。"""

    def __init__(self, service: Optional[BusinessService] = None):
        self._svc = service or BusinessService()

    # ---------------- 查询 ----------------

    def get_ticket(self, ticket_id: str) -> dict:
        return self._svc.get_ticket(ticket_id)

    def list_tickets(
        self,
        customer_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> dict:
        """按 customer / status 过滤工单列表。"""
        tickets = self._svc.store.tickets
        if customer_id:
            tickets = [t for t in tickets if t.customer_id == customer_id]
        if status:
            if status not in VALID_STATUSES:
                raise TicketWorkflowError(f"invalid status filter: {status!r}")
            tickets = [t for t in tickets if t.status == status]
        return {
            "count": len(tickets),
            "tickets": [t.model_dump() for t in tickets],
            "error": None,
        }

    # ---------------- 状态迁移 ----------------

    def transition(self, ticket_id: str, new_status: str, reason: str = "") -> dict:
        """
        执行一次状态迁移（校验合法性 + 写事件）。

        非法迁移抛 TicketWorkflowError（不静默）。
        """
        if new_status not in VALID_STATUSES:
            raise TicketWorkflowError(f"invalid target status: {new_status!r}")

        res = self._svc.get_ticket(ticket_id)
        if not res["found"]:
            raise TicketWorkflowError(f"ticket not found: {ticket_id!r}")

        ticket = res["ticket"]
        current = ticket["status"]
        if current == "CLOSED":
            raise TicketWorkflowError(f"ticket {ticket_id} is CLOSED (terminal), cannot transition to {new_status}")
        if new_status not in ALLOWED_TRANSITIONS[current]:
            raise TicketWorkflowError(
                f"illegal transition: {current} → {new_status} "
                f"(allowed: {list(ALLOWED_TRANSITIONS[current])})"
            )

        # 经 BusinessService 更新状态（幂等由调用方传 idempotency_key）
        update = self._svc.update_service_ticket(ticket_id=ticket_id, status=new_status)
        if update["updated"]:
            logger.info(f"TicketWorkflow: {ticket_id} {current} → {new_status} ({reason})")
        return {
            "ticket_id": ticket_id,
            "from": current,
            "to": new_status,
            "reason": reason,
            "events": self._svc.get_ticket_events(ticket_id)["events"],
            "error": None,
        }

    # ---------------- 事件回放 ----------------

    def get_events(self, ticket_id: str) -> dict:
        return self._svc.get_ticket_events(ticket_id)


__all__ = ["TicketWorkflow", "TicketWorkflowError", "ALLOWED_TRANSITIONS", "VALID_STATUSES"]
