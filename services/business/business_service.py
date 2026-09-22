"""
services/business/business_service.py

业务服务层（Step 6）。Agent → MCP Tool → 本 Service → 模拟"PostgreSQL"。

READ 类：
    get_device_info / get_device_status / get_device_warranty /
    get_repair_history / get_spare_parts

WRITE 类（必须支持 idempotency_key，防 retry/resume 重复执行）：
    create_service_ticket / reserve_spare_part / update_service_ticket

数据隔离：
    所有按 device_id 的查询都会反查 customer_id；跨 customer 访问需显式
    customer_id 匹配（Step 7 强化）。

禁止 Agent 直接 SQL —— 只能经本 Service。
"""

from typing import Optional

from datetime import datetime, timezone

from tool.logger import logger

from services.business.models import (
    Customer,
    Device,
    RepairRecord,
    SparePart,
    Ticket,
    TicketEvent,
    Approval,
)
from services.business.store import InMemoryBusinessStore, seed_demo_data


class BusinessServiceError(Exception):
    """业务层异常（Tool 侧统一捕获并转成带 error 标记的返回值）。"""


class BusinessService:
    """业务门面。默认用演示 seed 数据；可注入 store 做隔离测试。"""

    def __init__(self, store: Optional[InMemoryBusinessStore] = None, seed: bool = True):
        self.store = store if store is not None else InMemoryBusinessStore()
        if seed and not self.store.customers:
            seed_demo_data(self.store)

    # ---------------- 内部查找 ----------------

    def _find_device(self, device_id: str) -> Optional[Device]:
        return next((d for d in self.store.devices if d.device_id == device_id), None)

    def _find_customer(self, customer_id: str) -> Optional[Customer]:
        return next((c for c in self.store.customers if c.customer_id == customer_id), None)

    def _find_ticket(self, ticket_id: str) -> Optional[Ticket]:
        return next((t for t in self.store.tickets if t.ticket_id == ticket_id), None)

    def _find_spare_part(self, part_id: str) -> Optional[SparePart]:
        return next((p for p in self.store.spare_parts if p.part_id == part_id), None)

    def _require(self, obj, name: str, key: str):
        if obj is None:
            raise BusinessServiceError(f"{name} not found: {key!r}")
        return obj

    # ================= READ =================

    def get_device_info(self, device_id: str) -> dict:
        """返回设备 + 所属客户 + 基础状态。"""
        device = self._find_device(device_id)
        if device is None:
            return {"found": False, "device_id": device_id, "error": "device_not_found"}
        customer = self._find_customer(device.customer_id)
        return {
            "found": True,
            "device": device.model_dump(),
            "customer": customer.model_dump() if customer else None,
        }

    def get_device_status(self, device_id: str) -> dict:
        """设备当前状态 + 最近一次维修（若有）。"""
        device = self._find_device(device_id)
        if device is None:
            return {"found": False, "device_id": device_id, "error": "device_not_found"}
        recent = [r for r in self.store.repairs if r.device_id == device_id]
        recent.sort(key=lambda r: r.repaired_at, reverse=True)
        return {
            "found": True,
            "device_id": device_id,
            "status": device.status,
            "firmware_version": device.firmware_version,
            "recent_repairs": [r.model_dump() for r in recent[:3]],
        }

    def get_device_warranty(self, device_id: str) -> dict:
        """质保信息；无质保记录返回 warranty_valid=False + error=None（区别于故障）。"""
        device = self._find_device(device_id)
        if device is None:
            return {"found": False, "device_id": device_id, "error": "device_not_found"}
        has_warranty = device.warranty_until is not None
        return {
            "found": True,
            "device_id": device_id,
            "warranty_valid": has_warranty,
            "warranty_until": device.warranty_until,
            "error": None,
        }

    def get_repair_history(self, device_id: str) -> dict:
        """设备历史维修记录；无记录返回 records=[] + error=None。"""
        device = self._find_device(device_id)
        if device is None:
            return {"found": False, "device_id": device_id, "error": "device_not_found"}
        records = [r for r in self.store.repairs if r.device_id == device_id]
        records.sort(key=lambda r: r.repaired_at)
        return {
            "found": True,
            "device_id": device_id,
            "records": [r.model_dump() for r in records],
            "error": None,
        }

    def get_spare_parts(self, model: Optional[str] = None) -> dict:
        """备件库存；model 过滤（None = 全部）。"""
        parts = self.store.spare_parts
        if model:
            parts = [p for p in parts if model in p.compatible_models]
        return {
            "model": model,
            "parts": [p.model_dump() for p in parts],
            "error": None,
        }

    # ================= WRITE（幂等） =================

    def create_service_ticket(
        self,
        customer_id: str,
        device_id: Optional[str] = None,
        problem: str = "",
        diagnosis: Optional[str] = None,
        evidence: Optional[list] = None,
        priority: str = "low",
        idempotency_key: Optional[str] = None,
    ) -> dict:
        """
        创建工单（HITL 幂等契约）：
          1. 同 (customer_id, device_id) 已有 OPEN 工单 → 去重返回（防 resume/retry 重复建单）
          2. 同 idempotency_key 已有缓存结果 → 去重返回
          3. 否则创建新工单
        """
        # 1) 自然键去重：同客户 + 同设备的未关闭工单只保留一张
        existing = self.store.find_tickets_by_customer_and_device(customer_id, device_id)
        open_statuses = ("PENDING", "IN_PROGRESS", "WAITING_APPROVAL")
        for t in existing:
            if t.status in open_statuses:
                logger.info(
                    f"create_service_ticket 自然键去重: {customer_id}/{device_id} 已有 {t.ticket_id}"
                )
                return {"created": False, "deduplicated": True, "ticket": t.model_dump()}

        # 2) 显式 idempotency_key 去重
        if idempotency_key:
            hit, cached = self.store.idempotent(idempotency_key, None)
            if hit:
                logger.info(f"create_service_ticket 幂等命中: {idempotency_key}")
                return {"created": False, "deduplicated": True, "ticket": cached}

        # 2) 校验
        if not customer_id:
            raise BusinessServiceError("customer_id required")
        self._require(self._find_customer(customer_id), "customer", customer_id)
        if device_id and self._find_device(device_id) is None:
            raise BusinessServiceError(f"device not found: {device_id}")
        if not problem or not problem.strip():
            raise BusinessServiceError("problem required")
        if priority not in ("low", "medium", "high"):
            raise BusinessServiceError(f"invalid priority: {priority!r}")

        # 3) 创建
        ticket = Ticket(
            ticket_id=self.store.next_id("TICK"),
            customer_id=customer_id,
            device_id=device_id,
            problem=problem,
            diagnosis=diagnosis,
            evidence=evidence or [],
            priority=priority,
            status="PENDING",
        )
        self.store.tickets.append(ticket)
        self.store.ticket_events.append(
            TicketEvent(
                event_id=self.store.next_id("EVT"),
                ticket_id=ticket.ticket_id,
                event_type="created",
                detail=f"priority={priority}, device={device_id}",
            )
        )
        payload = ticket.model_dump()
        self.store.record_idempotent(idempotency_key, payload)
        logger.info(f"create_service_ticket: {ticket.ticket_id}")
        return {"created": True, "deduplicated": False, "ticket": payload}

    def reserve_spare_part(
        self,
        part_id: str,
        quantity: int = 1,
        device_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> dict:
        """预留备件。库存不足 / 数量非法 → 抛 BusinessServiceError。"""
        if quantity <= 0:
            raise BusinessServiceError(f"quantity must be positive: {quantity}")
        part = self._require(self._find_spare_part(part_id), "spare_part", part_id)
        available = part.stock - part.reserved
        if quantity > available:
            raise BusinessServiceError(
                f"insufficient stock for {part_id}: available={available}, requested={quantity}"
            )
        # 幂等：已预留过同 key 则直接返回（不重复扣减）
        if idempotency_key:
            hit, cached = self.store.idempotent(idempotency_key, None)
            if hit:
                return {"reserved": False, "deduplicated": True, "part": cached}

        part.reserved += quantity
        payload = {
            "part_id": part_id,
            "quantity": quantity,
            "device_id": device_id,
            "available_after": part.stock - part.reserved,
        }
        self.store.record_idempotent(idempotency_key, payload)
        logger.info(f"reserve_spare_part: {part_id} x{quantity} reserved")
        return {"reserved": True, "deduplicated": False, "part": payload}

    def update_service_ticket(
        self,
        ticket_id: str,
        status: Optional[str] = None,
        assigned_to: Optional[str] = None,
        diagnosis: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> dict:
        """更新工单（状态 / 指派 / 诊断）。非法状态 → 抛异常。"""
        ticket = self._require(self._find_ticket(ticket_id), "ticket", ticket_id)
        valid_status = ("PENDING", "IN_PROGRESS", "WAITING_APPROVAL", "COMPLETED", "CLOSED")
        if status is not None and status not in valid_status:
            raise BusinessServiceError(f"invalid status: {status!r}, expected one of {list(valid_status)}")

        if idempotency_key:
            hit, cached = self.store.idempotent(idempotency_key, None)
            if hit:
                return {"updated": False, "deduplicated": True, "ticket": cached}

        if status is not None and status != ticket.status:
            self.store.ticket_events.append(
                TicketEvent(
                    event_id=self.store.next_id("EVT"),
                    ticket_id=ticket_id,
                    event_type="status_changed",
                    detail=f"{ticket.status} → {status}",
                )
            )
            ticket.status = status
        if assigned_to is not None:
            ticket.assigned_to = assigned_to
            self.store.ticket_events.append(
                TicketEvent(
                    event_id=self.store.next_id("EVT"),
                    ticket_id=ticket_id,
                    event_type="assigned",
                    detail=f"assigned_to={assigned_to}",
                )
            )
        if diagnosis is not None:
            ticket.diagnosis = diagnosis
        ticket.updated_at = datetime.now(timezone.utc).isoformat()

        payload = ticket.model_dump()
        self.store.record_idempotent(idempotency_key, payload)
        logger.info(f"update_service_ticket: {ticket_id} status={ticket.status}")
        return {"updated": True, "deduplicated": False, "ticket": payload}

    # ================= 工单事件 / 审批（供 Step 8/9 用） =================

    def get_ticket(self, ticket_id: str) -> dict:
        ticket = self._find_ticket(ticket_id)
        if ticket is None:
            return {"found": False, "ticket_id": ticket_id, "error": "ticket_not_found"}
        events = [e.model_dump() for e in self.store.ticket_events if e.ticket_id == ticket_id]
        return {"found": True, "ticket": ticket.model_dump(), "events": events}

    def get_ticket_events(self, ticket_id: str) -> dict:
        events = [e.model_dump() for e in self.store.ticket_events if e.ticket_id == ticket_id]
        return {"ticket_id": ticket_id, "events": events, "error": None}

    def create_approval(self, approval: Approval) -> Approval:
        self.store.approvals.append(approval)
        return approval

    def get_approval(self, approval_id: str) -> dict:
        approval = next((a for a in self.store.approvals if a.approval_id == approval_id), None)
        if approval is None:
            return {"found": False, "approval_id": approval_id, "error": "approval_not_found"}
        return {"found": True, "approval": approval.model_dump(), "error": None}


__all__ = ["BusinessService", "BusinessServiceError"]
