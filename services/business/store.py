"""
services/business/store.py

本地模拟业务数据层（Step 6）。

设计：
  - InMemoryBusinessStore：进程内 dict，可 seed，可注入到 BusinessService
    （测试里用干净的小 store，默认 seed 用一份演示数据）。
  - 模拟"PostgreSQL"语义：按主键 CRUD + 简单查询。
  - 不接真实数据库；所有写操作幂等（idempotency_key 去重）。
"""

from typing import Optional

from services.business.models import (
    Approval,
    Customer,
    Device,
    RepairRecord,
    SparePart,
    Ticket,
    TicketEvent,
)


class InMemoryBusinessStore:
    """进程内模拟业务库。每个实体一张"表"（list），按 id 索引。"""

    def __init__(self):
        self.customers: list[Customer] = []
        self.devices: list[Device] = []
        self.repairs: list[RepairRecord] = []
        self.spare_parts: list[SparePart] = []
        self.tickets: list[Ticket] = []
        self.ticket_events: list[TicketEvent] = []
        self.approvals: list[Approval] = []
        # 幂等记录：idempotency_key -> 已执行的写操作结果
        self._idempotency: dict[str, dict] = {}
        # 自增 id 计数
        self._seq = 0

    def next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}-{self._seq:04d}"

    # ---------- 幂等 ----------
    def idempotent(self, key: str, payload: dict) -> tuple:
        """若 key 已存在返回 (True, 缓存结果)；否则 (False, None)。"""
        if key and key in self._idempotency:
            return True, self._idempotency[key]
        return False, None

    def record_idempotent(self, key: str, payload: dict) -> None:
        if key:
            self._idempotency[key] = payload


def seed_demo_data(store: InMemoryBusinessStore) -> None:
    """注入一份演示业务数据（本地模拟，非真实企业客户/数据）。"""
    c1 = Customer(customer_id="CUST-0001", name="演示客户A", contact="ops@demo-a.example")
    c2 = Customer(customer_id="CUST-0002", name="演示客户B", contact="support@demo-b.example")
    store.customers.extend([c1, c2])

    store.devices.extend(
        [
            Device(
                device_id="DEV-X200-001",
                customer_id="CUST-0001",
                model="X200",
                serial_number="SN-X200-0001",
                firmware_version="2.4.1",
                status="in_repair",
                warranty_until="2027-06-30T00:00:00Z",
                last_maintenance_date="2026-08-01T00:00:00Z",
            ),
            Device(
                device_id="DEV-X200-002",
                customer_id="CUST-0001",
                model="X200",
                serial_number="SN-X200-0002",
                firmware_version="2.4.1",
                status="active",
                warranty_until="2027-06-30T00:00:00Z",
            ),
            Device(
                device_id="DEV-H3C-003",
                customer_id="CUST-0002",
                model="H3CLA2608",
                serial_number="SN-H3C-0003",
                firmware_version="5.2.0",
                status="offline",
            ),
        ]
    )

    store.repairs.extend(
        [
            RepairRecord(
                record_id="REP-0001",
                device_id="DEV-X200-001",
                repaired_at="2026-07-10T00:00:00Z",
                fault_description="ERR-203 供电电压异常",
                repair_action="更换电源模块",
                technician="TECH-01",
            ),
        ]
    )

    store.spare_parts.extend(
        [
            SparePart(
                part_id="SP-PSU-X200",
                name="X200 电源模块",
                compatible_models=["X200"],
                stock=12,
                reserved=2,
            ),
            SparePart(
                part_id="SP-FAN-X200",
                name="X200 散热风扇",
                compatible_models=["X200"],
                stock=0,
            ),
        ]
    )


__all__ = ["InMemoryBusinessStore", "seed_demo_data"]
