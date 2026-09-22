"""
services/business/models.py

v2 业务实体模型（Pydantic）。

实体（Step 6 要求）：
    customers / devices / repair_records / tickets / ticket_events / approvals
    + spare_parts（get_spare_parts / reserve_spare_part 需要）

设计：
  - 全部 Pydantic BaseModel，JSON 可序列化（Tool 返回值直接 .model_dump()）
  - 时间字段统一 ISO 8601 字符串（避免 datetime 序列化差异）
  - 本地模拟数据，不依赖真实数据库
"""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------- customers ----------------
class Customer(BaseModel):
    customer_id: str = Field(..., description="客户 ID（模拟环境里也当 customer key）")
    name: str
    contact: Optional[str] = None
    created_at: str = Field(default_factory=_now_iso)


# ---------------- devices ----------------
class Device(BaseModel):
    device_id: str
    customer_id: str
    model: str
    serial_number: Optional[str] = None
    firmware_version: Optional[str] = None
    status: str = Field(
        default="active", description="active / offline / in_repair / decommissioned"
    )
    warranty_until: Optional[str] = Field(
        default=None, description="质保截止日期 ISO8601；None 表示无质保记录"
    )
    last_maintenance_date: Optional[str] = None
    created_at: str = Field(default_factory=_now_iso)


# ---------------- repair_records ----------------
class RepairRecord(BaseModel):
    record_id: str
    device_id: str
    repaired_at: str
    fault_description: str
    repair_action: str
    technician: Optional[str] = None


# ---------------- spare_parts ----------------
class SparePart(BaseModel):
    part_id: str
    name: str
    compatible_models: list = Field(default_factory=list)
    stock: int = Field(default=0, ge=0)
    reserved: int = Field(default=0, ge=0)


# ---------------- tickets ----------------
class Ticket(BaseModel):
    ticket_id: str
    customer_id: str
    device_id: Optional[str] = None
    problem: str
    diagnosis: Optional[str] = None
    evidence: list = Field(default_factory=list)
    priority: str = Field(default="low", description="low / medium / high")
    status: str = Field(
        default="PENDING",
        description="PENDING / IN_PROGRESS / WAITING_APPROVAL / COMPLETED / CLOSED",
    )
    assigned_to: Optional[str] = None
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)


# ---------------- ticket_events ----------------
class TicketEvent(BaseModel):
    event_id: str
    ticket_id: str
    event_type: str  # created / status_changed / assigned / note / closed ...
    detail: Optional[str] = None
    occurred_at: str = Field(default_factory=_now_iso)


# ---------------- approvals ----------------
class Approval(BaseModel):
    approval_id: str
    tool_name: str
    tool_args: dict = Field(default_factory=dict)
    requester: Optional[str] = None  # 通常是 agent run_id
    idempotency_key: Optional[str] = None
    status: str = Field(
        default="pending", description="pending / approved / rejected / executed"
    )
    created_at: str = Field(default_factory=_now_iso)
    decided_at: Optional[str] = None
    decided_by: Optional[str] = None
