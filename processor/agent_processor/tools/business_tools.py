"""
processor/agent_processor/tools/business_tools.py

MCP Business Tools（Step 6）：Agent → Tool → BusinessService → 模拟"PostgreSQL"。

READ 类：
    get_device_info / get_device_status / get_device_warranty /
    get_repair_history / get_spare_parts

WRITE 类（每个都必须带 idempotency_key；Step 9 会经 Policy + HITL 拦截）：
    create_service_ticket / reserve_spare_part / update_service_ticket

契约：
  - Tool 不直接 SQL，只调 BusinessService。
  - 所有 Tool 返回 dict（JSON 可序列化），带 found / error 标记，
    与 search_knowledge_base 的 Evidence 契约对齐（不静默吞异常）。
  - 参数 schema 用 Pydantic BaseModel（StructuredTool.args_schema），
    LLM 可感知；非法参数由 Pydantic 拦截或 Service 校验后抛错。

service 可注入（默认全局单例，带演示 seed 数据），便于单测 mock。
"""

from __future__ import annotations

import threading
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError

from services.business.business_service import BusinessService, BusinessServiceError
from services.business.store import InMemoryBusinessStore, seed_demo_data

# ---------------- 全局单例（惰性构造，避免 import 阶段 seed） ----------------
_service_lock = threading.Lock()
_shared_service: Optional[BusinessService] = None


def get_business_service() -> BusinessService:
    global _shared_service
    if _shared_service is None:
        with _service_lock:
            if _shared_service is None:
                _shared_service = BusinessService()
    return _shared_service


# ---------------- 入参 schema ----------------

class DeviceIDArgs(BaseModel):
    device_id: str = Field(..., description="设备 ID（如 DEV-X200-001）")


class DeviceWarrantyArgs(BaseModel):
    device_id: str = Field(..., description="设备 ID")


class RepairHistoryArgs(BaseModel):
    device_id: str = Field(..., description="设备 ID")


class SparePartsArgs(BaseModel):
    model: Optional[str] = Field(default=None, description="按兼容型号过滤；None=全部")


class CreateTicketArgs(BaseModel):
    customer_id: str = Field(..., description="客户 ID")
    device_id: Optional[str] = Field(default=None, description="相关设备 ID")
    problem: str = Field(..., description="问题描述")
    diagnosis: Optional[str] = Field(default=None, description="诊断结论（可选）")
    evidence: list = Field(default_factory=list, description="证据摘要列表")
    priority: str = Field(default="low", description="low / medium / high")
    idempotency_key: Optional[str] = Field(
        default=None, description="幂等键；同一 key 只创建一次（防 retry/resume 重复建单）"
    )


class ReservePartArgs(BaseModel):
    part_id: str = Field(..., description="备件 ID")
    quantity: int = Field(default=1, ge=1, description="预留数量（正整数）")
    device_id: Optional[str] = Field(default=None, description="预留目标设备（可选）")
    idempotency_key: Optional[str] = Field(default=None, description="幂等键")


class UpdateTicketArgs(BaseModel):
    ticket_id: str = Field(..., description="工单 ID")
    status: Optional[str] = Field(
        default=None,
        description="PENDING / IN_PROGRESS / WAITING_APPROVAL / COMPLETED / CLOSED",
    )
    assigned_to: Optional[str] = Field(default=None, description="指派给谁")
    diagnosis: Optional[str] = Field(default=None, description="更新诊断结论")
    idempotency_key: Optional[str] = Field(default=None, description="幂等键")


# ---------------- 执行器（异常统一转 error 标记） ----------------

def _safe(fn, args_model, kwargs: dict, service) -> dict:
    """调 service 方法；参数由 Pydantic schema 校验；service 异常 → error 标记（不静默）。"""
    try:
        params = args_model(**kwargs)
    except ValidationError as e:
        return {"result": None, "error": f"invalid_args: {e.errors()}"}
    try:
        return {"result": fn(service, params), "error": None}
    except BusinessServiceError as e:
        return {"result": None, "error": f"business_error: {e}"}
    except Exception as e:  # noqa: BLE001 —— 显式带出，不静默吞
        return {"result": None, "error": f"unexpected: {type(e).__name__}: {e}"}


def get_device_info_impl(svc, p: DeviceIDArgs) -> dict:
    return svc.get_device_info(p.device_id)


def get_device_status_impl(svc, p: DeviceIDArgs) -> dict:
    return svc.get_device_status(p.device_id)


def get_device_warranty_impl(svc, p: DeviceWarrantyArgs) -> dict:
    return svc.get_device_warranty(p.device_id)


def get_repair_history_impl(svc, p: RepairHistoryArgs) -> dict:
    return svc.get_repair_history(p.device_id)


def get_spare_parts_impl(svc, p: SparePartsArgs) -> dict:
    return svc.get_spare_parts(p.model)


def create_service_ticket_impl(svc, p: CreateTicketArgs) -> dict:
    return svc.create_service_ticket(
        customer_id=p.customer_id,
        device_id=p.device_id,
        problem=p.problem,
        diagnosis=p.diagnosis,
        evidence=p.evidence,
        priority=p.priority,
        idempotency_key=p.idempotency_key,
    )


def reserve_spare_part_impl(svc, p: ReservePartArgs) -> dict:
    return svc.reserve_spare_part(
        part_id=p.part_id,
        quantity=p.quantity,
        device_id=p.device_id,
        idempotency_key=p.idempotency_key,
    )


def update_service_ticket_impl(svc, p: UpdateTicketArgs) -> dict:
    return svc.update_service_ticket(
        ticket_id=p.ticket_id,
        status=p.status,
        assigned_to=p.assigned_to,
        diagnosis=p.diagnosis,
        idempotency_key=p.idempotency_key,
    )


# ---------------- Tool 工厂 ----------------

def _mk_tool(name, desc, impl, args_schema, service=None) -> StructuredTool:
    svc = service

    def _invoke(**kwargs) -> dict:
        target = svc if svc is not None else get_business_service()
        return _safe(impl, args_schema, kwargs, target)

    return StructuredTool.from_function(
        func=_invoke,
        name=name,
        description=desc,
        args_schema=args_schema,
    )


def build_business_tools(service: Optional[BusinessService] = None) -> dict:
    """返回 {tool_name: StructuredTool}（READ + WRITE 全量）。"""
    tools = {
        "get_device_info": _mk_tool(
            "get_device_info",
            "查询设备基础信息（含所属客户）。READ。",
            get_device_info_impl,
            DeviceIDArgs,
            service,
        ),
        "get_device_status": _mk_tool(
            "get_device_status",
            "查询设备当前状态 + 最近维修记录。READ。",
            get_device_status_impl,
            DeviceIDArgs,
            service,
        ),
        "get_device_warranty": _mk_tool(
            "get_device_warranty",
            "查询设备质保信息。READ。",
            get_device_warranty_impl,
            DeviceWarrantyArgs,
            service,
        ),
        "get_repair_history": _mk_tool(
            "get_repair_history",
            "查询设备历史维修记录。READ。",
            get_repair_history_impl,
            RepairHistoryArgs,
            service,
        ),
        "get_spare_parts": _mk_tool(
            "get_spare_parts",
            "查询备件库存（可按型号过滤）。READ。",
            get_spare_parts_impl,
            SparePartsArgs,
            service,
        ),
        "create_service_ticket": _mk_tool(
            "create_service_ticket",
            "创建服务工单。WRITE（高风险，Step 9 起需人工审批 + 幂等）。",
            create_service_ticket_impl,
            CreateTicketArgs,
            service,
        ),
        "reserve_spare_part": _mk_tool(
            "reserve_spare_part",
            "预留备件。WRITE（高风险，Step 9 起需人工审批 + 幂等）。",
            reserve_spare_part_impl,
            ReservePartArgs,
            service,
        ),
        "update_service_ticket": _mk_tool(
            "update_service_ticket",
            "更新工单状态 / 指派 / 诊断。WRITE（高风险，Step 9 起需人工审批 + 幂等）。",
            update_service_ticket_impl,
            UpdateTicketArgs,
            service,
        ),
    }
    return tools


def make_business_tools(service: Optional[BusinessService] = None):
    """便捷别名（与 build_business_tools 等价）。"""
    return build_business_tools(service)


if __name__ == "__main__":
    svc = BusinessService()
    t = build_business_tools(svc)
    print(t["get_device_info"].invoke({"device_id": "DEV-X200-001"}))
    print(t["create_service_ticket"].invoke(
        {"customer_id": "CUST-0001", "device_id": "DEV-X200-001",
         "problem": "ERR-203", "priority": "medium", "idempotency_key": "k1"}))
