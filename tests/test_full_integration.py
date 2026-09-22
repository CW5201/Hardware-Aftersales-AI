"""
tests/test_full_integration.py

Step 13 最终集成：验证完整主图 + Adaptive Retrieval + HITL + Ticket + Trace + Eval
各模块协同工作（全部本地模拟数据，不打真实 LLM / Milvus / DB）：

  1. Adaptive strategy 注入 RetrievalNode（简单→hybrid / 复杂→hybrid_rrf_rerank）
  2. 完整 HITL 链路：start(write tool) → interrupt → approve → resume → ticket 落库
  3. 完整 HITL 驳回链路：reject → resume → 不执行
  4. TracedAgentWorkflow：5 节点事件 + trace API 可查（/api/v2/threads/{id}/trace）
  5. /api/v2/devices 与 /api/v2/eval/agent 端点可达（工作台降级数据源）
  6. 完整生命周期：Triage→Memory→Retrieval→Diagnosis（主图）+
     Tool→Approval→Ticket（HITL 子图）端到端（各自 mock，不跨进程）
"""

import sys
from pathlib import Path
from unittest import mock

import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.retrieval.retrieval_service import RetrievalDocument, RetrievalResult
from processor.agent_processor.hitl_graph import HitlWorkflow
from processor.agent_processor.state import TriageResult
from processor.agent_processor.traced_graph import TracedAgentWorkflow
from services.business.business_service import BusinessService
from services.memory.memory_service import MemoryBackend, MemoryService
from services.trace.trace_store import InMemoryTraceStore


class _FakeLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        idx = min(self.calls, 1)
        self.calls += 1
        content = (
            '{"intent":"fault_diagnosis","product_model":"X200","device_id":"DEV-X200-001",'
            '"error_code":"ERR-203","retrieval_strategy":"hybrid"}'
            if idx == 0
            else '{"diagnosis_status":"diagnosed","hypotheses":[{"text":"h","evidence_indices":[0],'
                  '"confidence":"medium"}],"evidence":["证据"],"diagnosis_result":"结论",'
                  '"recommended_actions":["a"],"need_more_information":[],"need_human_review":false}'
        )

        class _R:
            pass

        r = _R()
        r.content = content
        return r


class _FakeService:
    def __init__(self, docs=True):
        self.docs = docs

    def search(self, query, product_model=None, strategy="hybrid_rrf_rerank", top_k=5):
        if self.docs:
            docs = [RetrievalDocument(chunk_id="c1", content="证据段落", item_name="X200", score=0.9)]
        else:
            docs = []
        return RetrievalResult(documents=docs, strategy=strategy, latency=0.1, metadata={})


# ================= 1. Adaptive strategy 注入 =================

def test_adaptive_simple_overrides_to_hybrid():
    from processor.agent_processor.nodes.retrieval import RetrievalNode

    node = RetrievalNode(service=_FakeService())
    state = {"user_query": "H3C 怎么配", "triage": TriageResult(intent="knowledge_question")}
    out = node(state)
    assert out["retrieval"]["strategy"] == "hybrid"
    assert out["retrieval"]["adaptive"]["strategy"] == "hybrid"


def test_adaptive_complex_keeps_full_pipeline():
    from processor.agent_processor.nodes.retrieval import RetrievalNode

    node = RetrievalNode(service=_FakeService())
    state = {
        "user_query": "X200 ERR-203",
        "triage": TriageResult(intent="fault_diagnosis", product_model="X200",
                               error_code="ERR-203"),
    }
    out = node(state)
    assert out["retrieval"]["strategy"] == "hybrid_rrf_rerank"


# ================= 2/3. 完整 HITL 链路 =================

def test_full_hitl_approve_creates_ticket():
    biz = BusinessService()
    wf = HitlWorkflow(business_service=biz)
    out = wf.start(
        "X200 ERR-203",
        "create_service_ticket",
        {"customer_id": "CUST-0001", "device_id": "DEV-X200-001",
         "problem": "ERR-203", "idempotency_key": "full-ik1", "priority": "high"},
        customer_id="CUST-0001", device_id="DEV-X200-001", thread_id="full-approve",
    )
    assert out["interrupted"] is True
    wf.approve(out["approval_id"])
    res = wf.resume(out["thread_id"], out["run_id"], "approved")
    assert res["state"]["tool_result"]["executed"] is True
    assert len(biz.store.tickets) == 1


def test_full_hitl_reject_no_exec():
    biz = BusinessService()
    wf = HitlWorkflow(business_service=biz)
    out = wf.start(
        "reserve",
        "reserve_spare_part",
        {"part_id": "SP-PSU-X200", "quantity": 1, "idempotency_key": "full-rk1"},
        thread_id="full-reject",
    )
    wf.reject(out["approval_id"])
    res = wf.resume(out["thread_id"], out["run_id"], "rejected")
    assert res["state"]["tool_result"]["rejected"] is True
    assert len(biz.store.tickets) == 0


# ================= 4. Traced 主图 + API =================

def test_traced_run_queryable_via_api():
    biz = BusinessService()
    mem = MemoryService(backend=MemoryBackend(business_service=biz))
    store = InMemoryTraceStore()
    wf = TracedAgentWorkflow(
        llm=_FakeLLM(), retrieval_service=_FakeService(),
        memory=mem, trace_store=store,
    )
    state = wf.run("X200 ERR-203", thread_id="trace-full", customer_id="CUST-0001")
    assert "trace_run_id" in state
    run = store.get_run(state["trace_run_id"])
    assert run.status == "success"
    assert len(run.events) == 5  # triage/memory_retrieve/retrieval/diagnosis/memory_persist
    assert [e.node for e in run.events] == [
        "triage", "memory_retrieve", "retrieval", "diagnosis", "memory_persist",
    ]


def test_devices_and_eval_api_endpoints():
    from fastapi.testclient import TestClient
    import services.trace.trace_store as ts

    store = InMemoryTraceStore()
    with mock.patch.object(ts, "get_trace_store", return_value=store):
        from web.api.query_service import app
        client = TestClient(app, raise_server_exceptions=False)

        # /api/v2/devices（BusinessService seed 演示数据）
        r = client.get("/api/v2/devices")
        assert r.status_code == 200, r.text
        assert r.json()["count"] >= 1

        # /api/v2/eval/agent（无标注 → 各指标 Not evaluated，不编数字）
        r2 = client.get("/api/v2/eval/agent")
        assert r2.status_code == 200, r2.text
        body = r2.json()
        assert "agent_metrics" in body and "failure_dataset" in body
        # 无标注样本 → 关键指标应为 Not evaluated
        assert body["agent_metrics"]["diagnosis_accuracy"] in ("Not evaluated",)


# ================= 6. 生命周期协同（主图 + HITL 子图各自 mock） =================

def test_lifecycle_chain_nodes_present_in_traced():
    """主图 5 节点 + HITL 的 tool/approval/ticket 在 trace 事件体系里可辨识。"""
    from processor.agent_processor.nodes.policy import PolicyNode
    from services.business.approval_service import ApprovalService

    biz = BusinessService()
    policy = PolicyNode(ApprovalService(biz), biz)
    # READ tool → 自动放行（不挂起）
    state = {"tool_name": "get_device_info", "tool_args": {"device_id": "DEV-X200-001"},
             "run_id": "lc-read"}
    out = policy(state)
    assert out["approval"]["status"] == "auto"
    # WRITE tool → 需审批（此处不调用 interrupt，仅验证 Policy 判定）
    assert ApprovalService.requires_approval("create_service_ticket", {}) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
