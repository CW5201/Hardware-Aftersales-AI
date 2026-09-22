"""
tests/test_memory_service.py

Step 7 Memory 层测试（全部进程内模拟数据，不打真实 DB / LLM）：

Short-term：
  1. put/get/update 基本读写
  2. thread 隔离（不同 thread 互不可见）
  3. TTL 过期清理
  4. record_tool_call 记录 tool state

Long-term：
  5. 按 customer 隔离：customer A 查不到 customer B 的设备/工单/故障
  6. 设备关联（customer 名下设备列表）
  7. 故障/维修历史（含业务库已有维修记录）
  8. 工单关联（经 BusinessService，幂等）

MemoryService：
  9. build_memory_context 组装 short_term + long_term
 10. MemoryRetrieve / MemoryPersist 节点在 Graph 里正确读写 state
"""

import sys
import time
from pathlib import Path
from unittest import mock

import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from services.business.business_service import BusinessService
from services.business.store import InMemoryBusinessStore
from services.memory.memory_service import (
    InMemoryLongTermStore,
    MemoryBackend,
    MemoryService,
    ShortTermMemory,
    ShortTermSlot,
)


# ================= Short-term =================

def test_short_term_put_get_update():
    st = ShortTermMemory()
    st.put("t1", ShortTermSlot())
    slot = st.update("t1", user_query="X200 报错", triage={"intent": "fault_diagnosis"})
    assert slot["user_query"] == "X200 报错"
    got = st.get("t1")
    assert got["user_query"] == "X200 报错"
    assert got["triage"] == {"intent": "fault_diagnosis"}


def test_short_term_thread_isolation():
    st = ShortTermMemory()
    st.update("tA", user_query="A 的问题")
    st.update("tB", user_query="B 的问题")
    assert st.get("tA")["user_query"] == "A 的问题"
    assert st.get("tB")["user_query"] == "B 的问题"
    # 两个 thread 互不串
    assert st.get("tA")["user_query"] != st.get("tB")["user_query"]


def test_short_term_ttl_expiry():
    st = ShortTermMemory(ttl_seconds=0.05)
    st.update("t1", user_query="x")
    time.sleep(0.08)
    assert st.get("t1") is None  # 过期被清理
    assert len(st) == 0


def test_short_term_record_tool():
    st = ShortTermMemory()
    st.record_tool_call("t1", "get_device_info", {"device_id": "DEV-1"}, {"found": True})
    slot = st.get("t1")
    assert slot["tool_calls"][0]["tool_name"] == "get_device_info"
    assert slot["tool_results"]["get_device_info"] == {"found": True}


# ================= Long-term（数据隔离核心） =================

def _biz_with_two_customers() -> BusinessService:
    """构造两个 customer（A 有设备/工单，B 有设备），用于隔离测试。"""
    svc = BusinessService(seed=False)
    from services.business.store import seed_demo_data
    seed_demo_data(svc.store)  # seed 里 CUST-0001 有 2 设备 + 维修记录，CUST-0002 有 1 设备
    return svc


def test_long_term_customer_isolation():
    svc = _biz_with_two_customers()
    be = MemoryBackend(business_service=svc)
    # A 的工单
    svc.create_service_ticket(customer_id="CUST-0001", problem="ERR-203", idempotency_key="iso1")
    # B 的工单
    svc.create_service_ticket(customer_id="CUST-0002", problem="H3C 离线", idempotency_key="iso2")

    a_tickets = be.get_tickets("CUST-0001")
    b_tickets = be.get_tickets("CUST-0002")
    # A 只看得到 A 的工单
    assert len(a_tickets) == 1
    assert a_tickets[0]["problem"] == "ERR-203"
    assert len(b_tickets) == 1
    assert b_tickets[0]["problem"] == "H3C 离线"
    # 互不泄露
    assert not any(t["problem"] == "H3C 离线" for t in a_tickets)


def test_long_term_device_list_isolation():
    svc = _biz_with_two_customers()
    be = MemoryBackend(business_service=svc)
    a_devs = be.get_device_list("CUST-0001")
    b_devs = be.get_device_list("CUST-0002")
    assert {d["device_id"] for d in a_devs} == {"DEV-X200-001", "DEV-X200-002"}
    assert {d["device_id"] for d in b_devs} == {"DEV-H3C-003"}
    # 交叉验证：A 的设备集不含 B 的设备
    assert "DEV-H3C-003" not in {d["device_id"] for d in a_devs}


def test_long_term_fault_history_includes_repairs():
    svc = _biz_with_two_customers()
    be = MemoryBackend(business_service=svc)
    faults = be.get_fault_history("CUST-0001")
    # seed 里有 REP-0001（DEV-X200-001 的维修）→ 算故障历史
    assert any(f["record_id"] == "REP-0001" for f in faults)
    # CUST-0002 无维修记录
    assert be.get_fault_history("CUST-0002") == []


def test_long_term_record_fault_and_query():
    svc = _biz_with_two_customers()
    be = MemoryBackend(business_service=svc)
    be.record_fault("CUST-0001", "DEV-X200-001", {"diagnosis": "电源模块故障"})
    faults = be.get_fault_history("CUST-0001")
    assert any(f.get("diagnosis") == "电源模块故障" for f in faults)
    # 不影响 CUST-0002
    assert not be.get_fault_history("CUST-0002")


# ================= MemoryService 门面 =================

def test_build_memory_context_structure():
    svc = _biz_with_two_customers()
    mem = MemoryService(backend=MemoryBackend(business_service=svc))
    ctx = mem.build_memory_context("t1", customer_id="CUST-0001", device_id="DEV-X200-001")
    assert "short_term" in ctx and "long_term" in ctx
    assert ctx["long_term"]["customer_id"] == "CUST-0001"
    assert ctx["long_term"]["device_id"] == "DEV-X200-001"
    assert len(ctx["long_term"]["devices"]) == 2


def test_short_term_persists_across_runs():
    svc = _biz_with_two_customers()
    mem = MemoryService(backend=MemoryBackend(business_service=svc))
    mem.short_update("t9", user_query="第一次提问")
    ctx = mem.build_memory_context("t9", customer_id="CUST-0001")
    assert ctx["short_term"]["user_query"] == "第一次提问"


# ================= Graph 节点 =================

def test_memory_nodes_in_graph():
    from processor.agent_processor.main_graph import AgentWorkflow
    from processor.agent_processor.state import DiagnosisResult, TriageResult
    from core.retrieval.retrieval_service import RetrievalDocument, RetrievalResult

    # 隔离：用独立 MemoryService（不污染全局单例）
    svc = _biz_with_two_customers()
    mem = MemoryService(backend=MemoryBackend(business_service=svc))

    class _FakeLLM:
        def __init__(self, triage_json, diagnosis_json):
            self._answers = [triage_json, diagnosis_json]
            self.calls = 0

        def invoke(self, messages):
            idx = min(self.calls, 1)
            self.calls += 1

            class _R:
                pass

            r = _R()
            r.content = self._answers[idx]
            return r

    class _FakeService:
        def search(self, query, product_model=None, strategy="hybrid_rrf_rerank", top_k=5):
            return RetrievalResult(
                documents=[RetrievalDocument(chunk_id="c1", content="证据段落", item_name="X200", score=0.9)],
                strategy=strategy, latency=0.1, metadata={},
            )

    llm = _FakeLLM(
        '{"intent":"fault_diagnosis","product_model":"X200","device_id":"DEV-X200-001",'
        '"retrieval_strategy":"hybrid"}',
        '{"diagnosis_status":"diagnosed","hypotheses":[{"text":"h","evidence_indices":[0],'
        '"confidence":"medium"}],"evidence":["证据段落"],"diagnosis_result":"结论",'
        '"recommended_actions":["a"],"need_more_information":[],"need_human_review":false}',
    )
    wf = AgentWorkflow(llm=llm, retrieval_service=_FakeService(), memory=mem)
    state = wf.run(
        "X200 出现 ERR-203",
        thread_id="graph-t1",
        customer_id="CUST-0001",
        device_id="DEV-X200-001",
    )
    # memory 字段已注入
    assert "memory" in state
    assert state["memory"]["long_term"]["customer_id"] == "CUST-0001"
    # 短期记忆已被 persist 写回
    slot = mem.short.get("graph-t1")
    assert slot is not None
    assert slot["user_query"] == "X200 出现 ERR-203"
    assert slot["diagnosis"] is not None
    # 长期记忆记了本次诊断的 fault（diagnosed + 有 customer/device）
    faults = mem.long.get_fault_history("CUST-0001")
    assert any(f.get("diagnosis") == "结论" for f in faults)


def test_memory_node_data_isolation_across_customers():
    """Graph 跑两次不同 customer，短期互不影响；长期按 customer 隔离。"""
    from processor.agent_processor.main_graph import AgentWorkflow
    from core.retrieval.retrieval_service import RetrievalDocument, RetrievalResult

    svc = _biz_with_two_customers()
    mem = MemoryService(backend=MemoryBackend(business_service=svc))

    class _FakeLLM:
        def __init__(self, diag_json):
            self._diag = diag_json
            self._triage_calls = 0
            self.calls = 0

        def invoke(self, messages):
            idx = min(self.calls, 1)
            self.calls += 1

            class _R:
                pass

            r = _R()
            r.content = (
                '{"intent":"fault_diagnosis","retrieval_strategy":"hybrid"}'
                if idx == 0
                else self._diag
            )
            return r

    class _FakeService:
        def search(self, query, product_model=None, strategy="hybrid_rrf_rerank", top_k=5):
            return RetrievalResult(documents=[], strategy=strategy, latency=0.1, metadata={})

    # customer A
    wfA = AgentWorkflow(
        llm=_FakeLLM('{"diagnosis_status":"diagnosed","hypotheses":[{"text":"hA","evidence_indices":[0],"confidence":"low"}],"evidence":[],"diagnosis_result":"A结论","recommended_actions":[],"need_more_information":[],"need_human_review":false}'),
        retrieval_service=_FakeService(), memory=mem,
    )
    sA = wfA.run("A 设备报错", thread_id="tA", customer_id="CUST-0001", device_id="DEV-X200-001")
    # customer B（不同 thread）
    wfB = AgentWorkflow(
        llm=_FakeLLM('{"diagnosis_status":"diagnosed","hypotheses":[{"text":"hB","evidence_indices":[0],"confidence":"low"}],"evidence":[],"diagnosis_result":"B结论","recommended_actions":[],"need_more_information":[],"need_human_review":false}'),
        retrieval_service=_FakeService(), memory=mem,
    )
    sB = wfB.run("B 设备报错", thread_id="tB", customer_id="CUST-0002", device_id="DEV-H3C-003")

    # 短期隔离
    assert mem.short.get("tA")["user_query"] == "A 设备报错"
    assert mem.short.get("tB")["user_query"] == "B 设备报错"
    # 长期隔离：用 record_fault 显式写不同 customer 的故障（diagnosed 才自动记 fault，
    # 本用例 retrieval 为空 → insufficient，所以直接调 long API 验证隔离）
    mem.long.record_fault("CUST-0001", "DEV-X200-001", {"diagnosis": "A结论"})
    mem.long.record_fault("CUST-0002", "DEV-H3C-003", {"diagnosis": "B结论"})
    a_faults = mem.long.get_fault_history("CUST-0001")
    b_faults = mem.long.get_fault_history("CUST-0002")
    assert any(f.get("diagnosis") == "A结论" for f in a_faults)
    assert not any(f.get("diagnosis") == "B结论" for f in a_faults)
    assert any(f.get("diagnosis") == "B结论" for f in b_faults)
    assert not any(f.get("diagnosis") == "A结论" for f in b_faults)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
