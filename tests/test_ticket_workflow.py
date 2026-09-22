"""
tests/test_ticket_workflow.py

Step 8 Ticket Workflow 测试（状态机 + 事件流 + 幂等，全部本地模拟数据）：

  状态机：
    1. 合法迁移 PENDING → IN_PROGRESS
    2. 合法迁移 IN_PROGRESS → WAITING_APPROVAL → COMPLETED → CLOSED
    3. 非法迁移 PENDING → COMPLETED（必须经 IN_PROGRESS）→ 报错
    4. CLOSED 是终态 → 再迁移报错
    5. 不存在的 ticket 迁移 → 报错
    6. 非法目标状态值 → 报错

  事件流：
    7. 每次迁移写 ticket_events（created / status_changed）
    8. get_events 回放完整事件

  幂等 / 数据隔离：
    9. 同 idempotency_key 重复创建 → 去重（不重复建单）
   10. 按 customer 过滤工单列表（隔离）
   11. 按 status 过滤 + 非法 status filter 报错
"""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from services.business.business_service import BusinessService
from services.business.ticket_workflow import (
    ALLOWED_TRANSITIONS,
    TicketWorkflow,
    TicketWorkflowError,
    VALID_STATUSES,
)


def _wf() -> TicketWorkflow:
    # 默认 seed（演示客户 CUST-0001 / CUST-0002 存在）
    return TicketWorkflow(BusinessService())


def _create(wf: TicketWorkflow, key: str, customer="CUST-0001") -> str:
    res = wf._svc.create_service_ticket(
        customer_id=customer, problem="ERR-203", idempotency_key=key
    )
    return res["ticket"]["ticket_id"]


# ================= 状态机 =================

def test_transition_pending_to_in_progress():
    wf = _wf()
    tid = _create(wf, "k1")
    res = wf.transition(tid, "IN_PROGRESS", reason="开始处理")
    assert res["from"] == "PENDING"
    assert res["to"] == "IN_PROGRESS"
    assert res["error"] is None


def test_full_happy_path_to_closed():
    wf = _wf()
    tid = _create(wf, "k2")
    for target, _ in [("IN_PROGRESS", 0), ("WAITING_APPROVAL", 1), ("COMPLETED", 2), ("CLOSED", 3)]:
        res = wf.transition(tid, target)
        assert res["to"] == target
    # CLOSED 后再迁移 → 报错（终态）
    with pytest.raises(TicketWorkflowError):
        wf.transition(tid, "IN_PROGRESS")


def test_illegal_transition_pending_to_completed():
    wf = _wf()
    tid = _create(wf, "k3")
    # PENDING 不能直接 → COMPLETED（必须经 IN_PROGRESS）
    assert "COMPLETED" not in ALLOWED_TRANSITIONS["PENDING"]
    with pytest.raises(TicketWorkflowError):
        wf.transition(tid, "COMPLETED")


def test_transition_missing_ticket():
    wf = _wf()
    with pytest.raises(TicketWorkflowError):
        wf.transition("TICK-NOPE", "IN_PROGRESS")


def test_transition_invalid_status_value():
    wf = _wf()
    tid = _create(wf, "k4")
    with pytest.raises(TicketWorkflowError):
        wf.transition(tid, "BROKEN")


# ================= 事件流 =================

def test_transition_writes_events():
    wf = _wf()
    tid = _create(wf, "k5")
    events0 = wf.get_events(tid)["events"]
    assert any(e["event_type"] == "created" for e in events0)

    wf.transition(tid, "IN_PROGRESS", reason="start")
    events1 = wf.get_events(tid)["events"]
    assert any(e["event_type"] == "status_changed" and "IN_PROGRESS" in e["detail"] for e in events1)
    assert len(events1) > len(events0)


def test_events_replay_order():
    wf = _wf()
    tid = _create(wf, "k6")
    wf.transition(tid, "IN_PROGRESS")
    wf.transition(tid, "WAITING_APPROVAL")
    events = wf.get_events(tid)["events"]
    types = [e["event_type"] for e in events]
    # created 一定在第一个
    assert types[0] == "created"
    assert "status_changed" in types


# ================= 幂等 / 隔离 =================

def test_create_idempotent():
    wf = _wf()
    r1 = wf._svc.create_service_ticket(customer_id="CUST-0001", problem="ERR-203", idempotency_key="idem1")
    r2 = wf._svc.create_service_ticket(customer_id="CUST-0001", problem="ERR-203", idempotency_key="idem1")
    assert r1["created"] is True
    assert r2["deduplicated"] is True
    assert r1["ticket"]["ticket_id"] == r2["ticket"]["ticket_id"]
    # 库里只有一条
    listing = wf.list_tickets(customer_id="CUST-0001")
    assert listing["count"] == 1


def test_list_by_customer_isolation():
    wf = _wf()
    _create(wf, "kA1", customer="CUST-0001")
    _create(wf, "kA2", customer="CUST-0002")
    a = wf.list_tickets(customer_id="CUST-0001")
    b = wf.list_tickets(customer_id="CUST-0002")
    assert a["count"] == 1 and b["count"] == 1
    assert a["tickets"][0]["customer_id"] == "CUST-0001"
    assert b["tickets"][0]["customer_id"] == "CUST-0002"


def test_list_by_status_filter():
    wf = _wf()
    tid = _create(wf, "k7")
    wf.transition(tid, "IN_PROGRESS")
    inprog = wf.list_tickets(status="IN_PROGRESS")
    assert any(t["ticket_id"] == tid for t in inprog["tickets"])
    pending = wf.list_tickets(status="PENDING")
    assert all(t["ticket_id"] != tid for t in pending["tickets"])


def test_invalid_status_filter_raises():
    wf = _wf()
    with pytest.raises(TicketWorkflowError):
        wf.list_tickets(status="BOGUS")


# ================= 常量契约 =================

def test_valid_statuses_constant():
    assert set(VALID_STATUSES) == {"PENDING", "IN_PROGRESS", "WAITING_APPROVAL", "COMPLETED", "CLOSED"}
    assert "CLOSED" in ALLOWED_TRANSITIONS
    assert ALLOWED_TRANSITIONS["CLOSED"] == ()  # 终态


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
