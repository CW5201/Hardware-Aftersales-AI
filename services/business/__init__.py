"""
services/business/__init__.py

v2 业务服务层（Step 6）。

架构契约：
    Agent → MCP Tool → Business Service → "PostgreSQL"（本地模拟）

  - 禁止 Agent 直接 SQL：所有业务读写只能经过这里的 Service。
  - 使用本地模拟业务数据（in-memory seed + 可注入 store），
    不接不存在的真实 ERP / 数据库。
  - READ / WRITE 分类：WRITE 类操作必须带 idempotency_key（Step 9 强化）。
"""

from services.business.business_service import BusinessService
from services.business.models import (
    Customer,
    Device,
    RepairRecord,
    SparePart,
    Ticket,
    TicketEvent,
    Approval,
)

__all__ = [
    "BusinessService",
    "Customer",
    "Device",
    "RepairRecord",
    "SparePart",
    "Ticket",
    "TicketEvent",
    "Approval",
]
