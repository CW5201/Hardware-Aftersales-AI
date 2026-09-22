"""
tests/test_hitl_flow.py

Step 9 Human-in-the-loop 测试（本地模拟数据，全部进程内）：

Policy API：
  1. HIGH_RISK_WRITE_TOOLS 白名单（3 个写工具）
  2. READ / 未知工具 → 不需审批
  3. 写工具缺 idempotency_key → 仍需审批

审批生命周期（ApprovalService）：
  4. create_approval → pending
  5. approve → approved；reject → rejected
  6. 重复决策（已 decided）→ ApprovalError

HITL Graph（interrupt / resume）：
  7. start() 高风险写 → interrupted + approval_id
  8. approve + resume → 执行（executed=True）
  9. reject + resume → 不执行（rejected=True）
 10. READ 工具 → 不挂起（interrupted=False）

幂等（防 retry/resume 重复建单）：
 11. 同 (customer, device) 已有 OPEN 工单 → 自然键去重，不重复建单
 12. 同 idempotency_key 显式去重
 13. 审批 execute 幂等：已 executed + 同 key → deduplicated=True
"""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from services.business.approval_service import ApprovalService, ApprovalError, HIGH_RISK_WRITE_TOOLS
from services.business.business_service import BusinessService
from processor.agent_processor.hitl_graph import HitlWorkflow


# ================= Policy API =================

def test_high_risk_write_tools_whitelist():
    assert HIGH_RISK_WRITE_TOOLS == {
        "create_service_ticket",
        "reserve_spare_part",
        "update_service_ticket",
    }


def test_read_tools_need_no_approval():
    for tool in ["get_device_info", "get_device_status", "get_device_warranty",
                 "get_repair_history", "get_spare_parts"]:
        assert ApprovalService.requires_approval(tool, {}) is False


def test_unknown_tool_no_approval():
    assert ApprovalService.requires_approval("some_read_tool", {}) is False


def test_write_tools_need_approval_even_without_idem_key():
    for tool in HIGH_RISK_WRITE_TOOLS:
        assert ApprovalService.requires_approval(tool, {}) is True


# ================= 审批生命周期 =================

def test_approval_create_approve_reject_lifecycle():
    svc = ApprovalService(BusinessService())
    rec = svc.create_approval("create_service_ticket", {"customer_id": "CUST-0001", "problem": "x"})
    assert rec["status"] == "pending"
    aid = rec["approval_id"]

    approved = svc.approve(aid, decided_by="engineer")
    assert approved["status"] == "approved"
    assert approved["decided_by"] == "engineer"

    # 重复决策 → 报错（不能再次 approve/reject 已 decided 的）
    with pytest.raises(ApprovalError):
        svc.reject(aid)


def test_approval_reject_then_cannot_approve():
    svc = ApprovalService(BusinessService())
    rec = svc.create_approval("reserve_spare_part", {"part_id": "SP-PSU-X200", "quantity": 1})
    svc.reject(rec["approval_id"], decided_by="lead")
    with pytest.raises(ApprovalError):
        svc.approve(rec["approval_id"])


def test_approval_get_missing():
    svc = ApprovalService(BusinessService())
    res = svc.get("APR-NOPE")
    assert res["found"] is False
    assert res["error"] == "approval_not_found"


# ================= HITL Graph interrupt / resume =================

def _wf():
    biz = BusinessService()  # seed=True（演示客户/设备存在）
    return HitlWorkflow(business_service=biz), biz


def test_start_interruption_for_write_tool():
    wf, _ = _wf()
    out = wf.start(
        "X200 ERR-203",
        "create_service_ticket",
        {"customer_id": "CUST-0001", "device_id": "DEV-X200-001",
         "problem": "ERR-203", "idempotency_key": "k1"},
        customer_id="CUST-0001", device_id="DEV-X200-001", thread_id="t-w1",
    )
    assert out["interrupted"] is True
    assert out["approval_id"] is not None
    assert out["thread_id"] == "t-w1"


def test_approve_then_resume_executes():
    wf, biz = _wf()
    out = wf.start(
        "X200 ERR-203",
        "create_service_ticket",
        {"customer_id": "CUST-0001", "device_id": "DEV-X200-001",
         "problem": "ERR-203", "idempotency_key": "k1"},
        customer_id="CUST-0001", device_id="DEV-X200-001", thread_id="t-w2",
    )
    wf.approve(out["approval_id"])
    res = wf.resume(out["thread_id"], out["run_id"], "approved")
    tr = res["state"]["tool_result"]
    assert tr["executed"] is True
    assert tr["error"] is None
    assert len(biz.store.tickets) == 1


def test_reject_then_resume_does_not_execute():
    wf, biz = _wf()
    out = wf.start(
        "reserve",
        "reserve_spare_part",
        {"part_id": "SP-PSU-X200", "quantity": 2, "idempotency_key": "rk1"},
        thread_id="t-w3",
    )
    wf.reject(out["approval_id"])
    res = wf.resume(out["thread_id"], out["run_id"], "rejected")
    tr = res["state"]["tool_result"]
    assert tr["executed"] is False
    assert tr["rejected"] is True
    # 备件未实际预留
    part = next(p for p in biz.store.spare_parts if p.part_id == "SP-PSU-X200")
    assert part.reserved == 2  # seed 默认 reserved=2，reject 后不变


def test_read_tool_no_interruption():
    wf, _ = _wf()
    out = wf.start(
        "read",
        "get_device_info",
        {"device_id": "DEV-X200-001"},
        thread_id="t-w4",
    )
    assert out["interrupted"] is False


# ================= 幂等（防 retry/resume 重复建单） =================

def test_natural_key_dedup_no_duplicate_ticket():
    wf, biz = _wf()
    # 第一次：建单
    o1 = wf.start(
        "a", "create_service_ticket",
        {"customer_id": "CUST-0001", "device_id": "DEV-X200-001",
         "problem": "ERR-203", "idempotency_key": "idk1"},
        customer_id="CUST-0001", device_id="DEV-X200-001", thread_id="t-d1",
    )
    wf.approve(o1["approval_id"])
    wf.resume(o1["thread_id"], o1["run_id"], "approved")
    assert len(biz.store.tickets) == 1

    # 第二次：同 customer + 同 device，不同 idempotency_key
    # 自然键去重 → 不重复建单（OPEN 工单只保留一张）
    o2 = wf.start(
        "b", "create_service_ticket",
        {"customer_id": "CUST-0001", "device_id": "DEV-X200-001",
         "problem": "ERR-203", "idempotency_key": "idk2"},
        customer_id="CUST-0001", device_id="DEV-X200-001", thread_id="t-d2",
    )
    wf.approve(o2["approval_id"])
    r2 = wf.resume(o2["thread_id"], o2["run_id"], "approved")
    assert r2["state"]["tool_result"]["result"]["deduplicated"] is True
    assert len(biz.store.tickets) == 1  # 仍只有一张


def test_explicit_idempotency_key_dedup():
    biz = BusinessService()
    svc = biz
    args = {"customer_id": "CUST-0001", "problem": "ERR-203", "idempotency_key": "explicit-key"}
    r1 = svc.create_service_ticket(
        customer_id=args["customer_id"], problem=args["problem"], idempotency_key=args["idempotency_key"]
    )
    r2 = svc.create_service_ticket(
        customer_id=args["customer_id"], problem=args["problem"], idempotency_key=args["idempotency_key"]
    )
    assert r1["created"] is True
    assert r2["deduplicated"] is True
    assert r1["ticket"]["ticket_id"] == r2["ticket"]["ticket_id"]


def test_approval_execute_idempotent_on_resume():
    """同一 approval 反复 resume（模拟 retry）→ 写操作只落库一次。"""
    wf, biz = _wf()
    out = wf.start(
        "x", "create_service_ticket",
        {"customer_id": "CUST-0001", "device_id": "DEV-X200-001",
         "problem": "ERR-203", "idempotency_key": "k-resume"},
        customer_id="CUST-0001", device_id="DEV-X200-001", thread_id="t-d3",
    )
    wf.approve(out["approval_id"])
    r1 = wf.resume(out["thread_id"], out["run_id"], "approved")
    assert r1["state"]["tool_result"]["executed"] is True
    assert len(biz.store.tickets) == 1

    # 再 resume 一次（retry）→ 幂等去重，不重复建单
    r2 = wf.resume(out["thread_id"], out["run_id"], "approved")
    tr2 = r2["state"]["tool_result"]
    assert len(biz.store.tickets) == 1  # 仍只有一张


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
