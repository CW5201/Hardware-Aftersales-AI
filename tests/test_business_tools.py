"""
tests/test_business_tools.py

Step 6 MCP Business Tools 测试（全部本地模拟数据，不打真实 DB / ERP）：

  每个 Tool 覆盖：
    - schema（Pydantic 入参校验 / 非法参数）
    - 正常参数
    - 无数据（found=False / error 标记，区别于故障）
    - 非法参数（service 层业务校验）
    - service exception（error 标记，不静默）
    - 返回值结构

READ 工具：get_device_info / get_device_status / get_device_warranty /
           get_repair_history / get_spare_parts
WRITE 工具：create_service_ticket / reserve_spare_part / update_service_ticket
            （幂等 key 去重验证）
"""

import sys
from pathlib import Path
from unittest import mock

import pytest
from pydantic import ValidationError

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from services.business.business_service import BusinessService, BusinessServiceError
from processor.agent_processor.tools.business_tools import build_business_tools


def _svc():
    return BusinessService()


def _tools(svc=None):
    return build_business_tools(svc or _svc())


# ================= READ: get_device_info =================

def test_get_device_info_normal():
    out = _tools()["get_device_info"].invoke({"device_id": "DEV-X200-001"})
    assert out["error"] is None
    assert out["result"]["found"] is True
    assert out["result"]["device"]["model"] == "X200"
    assert out["result"]["customer"]["customer_id"] == "CUST-0001"


def test_get_device_info_missing():
    out = _tools()["get_device_info"].invoke({"device_id": "DEV-NOPE"})
    assert out["error"] is None
    assert out["result"]["found"] is False
    assert out["result"]["error"] == "device_not_found"


def test_get_device_info_schema_requires_device_id():
    tool = _tools()["get_device_info"]
    with pytest.raises(ValidationError):
        tool.invoke({})


def test_get_device_info_service_exception_marked():
    tools = _tools()
    with mock.patch.object(BusinessService, "get_device_info", side_effect=RuntimeError("db down")):
        out = tools["get_device_info"].invoke({"device_id": "DEV-X200-001"})
    assert "unexpected" in out["error"]
    assert out["result"] is None


# ================= READ: get_device_status =================

def test_get_device_status_normal():
    out = _tools()["get_device_status"].invoke({"device_id": "DEV-X200-001"})
    assert out["error"] is None
    assert out["result"]["status"] == "in_repair"
    # 有最近维修记录
    assert len(out["result"]["recent_repairs"]) == 1
    assert out["result"]["recent_repairs"][0]["record_id"] == "REP-0001"


def test_get_device_status_missing():
    out = _tools()["get_device_status"].invoke({"device_id": "DEV-NOPE"})
    assert out["result"]["found"] is False
    assert out["result"]["error"] == "device_not_found"


# ================= READ: get_device_warranty =================

def test_get_device_warranty_valid():
    out = _tools()["get_device_warranty"].invoke({"device_id": "DEV-X200-001"})
    assert out["result"]["warranty_valid"] is True
    assert out["result"]["warranty_until"] == "2027-06-30T00:00:00Z"
    assert out["result"]["error"] is None


def test_get_device_warranty_no_record():
    out = _tools()["get_device_warranty"].invoke({"device_id": "DEV-H3C-003"})
    # 无质保记录 ≠ 故障：warranty_valid=False, error=None
    assert out["result"]["warranty_valid"] is False
    assert out["result"]["error"] is None


def test_get_device_warranty_missing_device():
    out = _tools()["get_device_warranty"].invoke({"device_id": "DEV-NOPE"})
    assert out["result"]["found"] is False


# ================= READ: get_repair_history =================

def test_get_repair_history_normal():
    out = _tools()["get_repair_history"].invoke({"device_id": "DEV-X200-001"})
    assert out["result"]["found"] is True
    assert len(out["result"]["records"]) == 1
    assert out["result"]["error"] is None


def test_get_repair_history_empty():
    out = _tools()["get_repair_history"].invoke({"device_id": "DEV-X200-002"})
    # 无维修记录 ≠ 故障
    assert out["result"]["records"] == []
    assert out["result"]["error"] is None


# ================= READ: get_spare_parts =================

def test_get_spare_parts_all():
    out = _tools()["get_spare_parts"].invoke({})
    assert out["result"]["error"] is None
    assert len(out["result"]["parts"]) == 2


def test_get_spare_parts_by_model():
    out = _tools()["get_spare_parts"].invoke({"model": "X200"})
    assert len(out["result"]["parts"]) == 2
    assert all("X200" in p["compatible_models"] for p in out["result"]["parts"])


def test_get_spare_parts_unknown_model_empty():
    out = _tools()["get_spare_parts"].invoke({"model": "ZZZ"})
    assert out["result"]["parts"] == []
    assert out["result"]["error"] is None


def test_get_spare_parts_stock_filter():
    out = _tools()["get_spare_parts"].invoke({"model": "X200"})
    by_id = {p["part_id"]: p for p in out["result"]["parts"]}
    assert by_id["SP-FAN-X200"]["stock"] == 0


# ================= WRITE: create_service_ticket =================

def test_create_ticket_normal():
    tools = _tools()
    out = tools["create_service_ticket"].invoke(
        {"customer_id": "CUST-0001", "device_id": "DEV-X200-001",
         "problem": "ERR-203", "priority": "high", "idempotency_key": "t1"}
    )
    assert out["error"] is None
    assert out["result"]["created"] is True
    assert out["result"]["ticket"]["status"] == "PENDING"


def test_create_ticket_idempotent():
    tools = _tools()
    payload = {"customer_id": "CUST-0001", "device_id": "DEV-X200-001",
               "problem": "ERR-203", "idempotency_key": "t2"}
    first = tools["create_service_ticket"].invoke(payload)
    second = tools["create_service_ticket"].invoke(payload)
    assert first["result"]["created"] is True
    assert second["result"]["deduplicated"] is True
    assert second["result"]["ticket"]["ticket_id"] == first["result"]["ticket"]["ticket_id"]


def test_create_ticket_invalid_priority():
    out = _tools()["create_service_ticket"].invoke(
        {"customer_id": "CUST-0001", "problem": "x", "priority": "URGENT"}
    )
    assert "business_error" in out["error"]


def test_create_ticket_unknown_customer():
    out = _tools()["create_service_ticket"].invoke(
        {"customer_id": "CUST-NOPE", "problem": "x"}
    )
    assert "business_error" in out["error"]


def test_create_ticket_empty_problem():
    out = _tools()["create_service_ticket"].invoke(
        {"customer_id": "CUST-0001", "problem": "   "}
    )
    assert "business_error" in out["error"]


# ================= WRITE: reserve_spare_part =================

def test_reserve_part_normal():
    tools = _tools()
    out = tools["reserve_spare_part"].invoke(
        {"part_id": "SP-PSU-X200", "quantity": 2, "idempotency_key": "r1"}
    )
    assert out["result"]["reserved"] is True
    assert out["result"]["part"]["available_after"] == 12 - 2 - 2  # seed reserved=2, 再 +2


def test_reserve_part_insufficient_stock():
    out = _tools()["reserve_spare_part"].invoke({"part_id": "SP-FAN-X200", "quantity": 5})
    assert "business_error" in out["error"]
    assert "insufficient stock" in out["error"]


def test_reserve_part_invalid_quantity():
    tools = _tools()
    tool = tools["reserve_spare_part"]
    with pytest.raises(ValidationError):
        tool.invoke({"part_id": "SP-PSU-X200", "quantity": 0})
    # 也验证 _safe 层对 schema 违规的兜底（直接调内部 _safe 路径）
    from processor.agent_processor.tools.business_tools import _safe, ReservePartArgs
    out = _safe(
        lambda svc, p: svc.reserve_spare_part(part_id=p.part_id, quantity=p.quantity),
        ReservePartArgs,
        {"part_id": "SP-PSU-X200", "quantity": 0},
        _svc(),
    )
    assert out["result"] is None
    assert "invalid_args" in out["error"]


def test_tool_schema_rejects_missing_required():
    tools = _tools()
    tool = tools["create_service_ticket"]
    with pytest.raises(ValidationError):
        tool.invoke({"problem": "x"})
    from processor.agent_processor.tools.business_tools import _safe, CreateTicketArgs
    out = _safe(
        lambda svc, p: svc.create_service_ticket(customer_id=p.customer_id, problem=p.problem),
        CreateTicketArgs,
        {"problem": "x"},
        _svc(),
    )
    assert out["result"] is None
    assert "invalid_args" in out["error"]


def test_reserve_part_idempotent():
    tools = _tools()
    payload = {"part_id": "SP-PSU-X200", "quantity": 1, "idempotency_key": "r2"}
    first = tools["reserve_spare_part"].invoke(payload)
    second = tools["reserve_spare_part"].invoke(payload)
    assert first["result"]["reserved"] is True
    assert second["result"]["deduplicated"] is True


# ================= WRITE: update_service_ticket =================

def test_update_ticket_normal():
    tools = _tools()
    created = tools["create_service_ticket"].invoke(
        {"customer_id": "CUST-0001", "problem": "ERR-203", "idempotency_key": "u1"}
    )
    tid = created["result"]["ticket"]["ticket_id"]
    out = tools["update_service_ticket"].invoke(
        {"ticket_id": tid, "status": "IN_PROGRESS", "assigned_to": "TECH-02"}
    )
    assert out["error"] is None
    assert out["result"]["ticket"]["status"] == "IN_PROGRESS"
    assert out["result"]["ticket"]["assigned_to"] == "TECH-02"


def test_update_ticket_invalid_status():
    tools = _tools()
    created = tools["create_service_ticket"].invoke(
        {"customer_id": "CUST-0001", "problem": "x", "idempotency_key": "u2"}
    )
    tid = created["result"]["ticket"]["ticket_id"]
    out = tools["update_service_ticket"].invoke(
        {"ticket_id": tid, "status": "BROKEN"}
    )
    assert "business_error" in out["error"]


def test_update_ticket_missing():
    out = _tools()["update_service_ticket"].invoke(
        {"ticket_id": "TICK-NOPE", "status": "COMPLETED"}
    )
    assert "business_error" in out["error"]


# ================= service 异常统一带出 =================

def test_service_exception_on_read_tool():
    tools = _tools()
    with mock.patch.object(BusinessService, "get_repair_history", side_effect=BusinessServiceError("sim")):
        out = tools["get_repair_history"].invoke({"device_id": "DEV-X200-001"})
    assert out["error"] == "business_error: sim"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
