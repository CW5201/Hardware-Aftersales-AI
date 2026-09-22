"""
tests/test_agent_trace.py

Step 10 Agent Trace 测试（全部 mock LLM / 检索，进程内 trace store）：

Trace 存储（InMemoryTraceStore）：
  1. start_run / finish 记录 run 摘要（status / latency / token）
  2. add_event 有序（seq 单调增）
  3. 按 thread 查 runs；按 run 查详情
  4. estimate_tokens 粗估；TokenUsage.from_usage_metadata

TracedAgentWorkflow：
  5. run 完成后 trace 里有 5 个节点事件（triage/memory_retrieve/retrieval/diagnosis/memory_persist）
  6. 事件含 input/output/latency/token/status
  7. retrieval 节点事件带 retrieved_docs_count
  8. run 级 token_usage 已汇总；total_latency_ms 非空

API：
  9. GET /api/v2/threads/{thread_id}/trace 返回 runs
 10. GET /api/v2/runs/{run_id} 返回全量事件；缺 run → 404
"""

import sys
import time
from pathlib import Path
from unittest import mock

import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.retrieval.retrieval_service import RetrievalDocument, RetrievalResult
from processor.agent_processor.traced_graph import TracedAgentWorkflow
from services.memory.memory_service import MemoryService, MemoryBackend
from services.business.business_service import BusinessService
from services.trace.trace_store import (
    InMemoryTraceStore,
    TokenUsage,
    TraceEvent,
    estimate_tokens,
)


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
            else '{"diagnosis_status":"diagnosed","hypotheses":[{"text":"h","evidence_indices":[0],"confidence":"medium"}],'
                  '"evidence":["证据"],"diagnosis_result":"结论","recommended_actions":["a"],'
                  '"need_more_information":[],"need_human_review":false}'
        )

        class _R:
            pass

        r = _R()
        r.content = content
        return r


class _FakeService:
    def search(self, query, product_model=None, strategy="hybrid_rrf_rerank", top_k=5):
        return RetrievalResult(
            documents=[
                RetrievalDocument(chunk_id="c1", content="证据段落A", item_name="X200", score=0.9),
                RetrievalDocument(chunk_id="c2", content="证据段落B", item_name="X200", score=0.8),
            ],
            strategy=strategy,
            latency=0.1,
            metadata={},
        )


def _wf():
    biz = BusinessService()
    mem = MemoryService(backend=MemoryBackend(business_service=biz))
    store = InMemoryTraceStore()  # 隔离的 trace store
    wf = TracedAgentWorkflow(
        llm=_FakeLLM(),
        retrieval_service=_FakeService(),
        memory=mem,
        trace_store=store,
    )
    return wf, store, biz


# ================= Trace 存储 =================

def test_start_run_finish_records_summary():
    store = InMemoryTraceStore()
    run = store.start_run("run-1", "thread-1", "X200 报错")
    run.finish(status="success")
    d = run.to_dict()
    assert d["status"] == "success"
    assert d["total_latency_ms"] is not None
    assert d["run_id"] == "run-1"
    assert d["thread_id"] == "thread-1"


def test_events_sequential():
    store = InMemoryTraceStore()
    run = store.start_run("run-2", "thread-2", "q")
    for i in range(3):
        store.add_event("run-2", TraceEvent("run-2", 0, node=f"n{i}"))
    assert [e.seq for e in run.events] == [1, 2, 3]


def test_query_by_thread_and_run():
    store = InMemoryTraceStore()
    store.start_run("r1", "t1", "q1")
    store.start_run("r2", "t1", "q2")
    store.start_run("r3", "t2", "q3")
    assert len(store.get_thread_runs("t1")) == 2
    assert len(store.get_thread_runs("t2")) == 1
    assert store.get_run("r3").user_query == "q3"
    assert store.get_run("NOPE") is None


def test_estimate_tokens_and_usage():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abc") == 1
    assert estimate_tokens("a" * 100) == 25
    tu = TokenUsage(10, 5)
    assert tu.to_dict() == {"input": 10, "output": 5, "total": 15}
    tu2 = TokenUsage.from_usage_metadata({"input_tokens": 3, "output_tokens": 7})
    assert tu2.total == 10


# ================= TracedAgentWorkflow =================

def test_traced_run_records_five_node_events():
    wf, store, _ = _wf()
    state = wf.run("X200 ERR-203", thread_id="tt-1", customer_id="CUST-0001")
    run = store.get_run(state["trace_run_id"])
    node_names = [e.node for e in run.events]
    assert node_names == [
        "triage",
        "memory_retrieve",
        "retrieval",
        "diagnosis",
        "memory_persist",
    ]


def test_traced_event_fields_present():
    wf, store, _ = _wf()
    state = wf.run("X200 ERR-203", thread_id="tt-2", customer_id="CUST-0001")
    run = store.get_run(state["trace_run_id"])
    for e in run.events:
        d = e.to_dict()
        for key in ("event_id", "run_id", "seq", "node", "input", "output",
                    "latency_ms", "status", "token_usage", "occurred_at"):
            assert key in d
    # 每个节点事件 latency_ms 非空
    assert all(e.latency_ms is not None for e in run.events)


def test_retrieval_event_has_doc_count():
    wf, store, _ = _wf()
    state = wf.run("X200 ERR-203", thread_id="tt-3", customer_id="CUST-0001")
    run = store.get_run(state["trace_run_id"])
    retrieval_evt = next(e for e in run.events if e.node == "retrieval")
    assert retrieval_evt.retrieved_docs_count == 2


def test_traced_run_token_and_latency_summarized():
    wf, store, _ = _wf()
    state = wf.run("X200 ERR-203", thread_id="tt-4", customer_id="CUST-0001")
    run = store.get_run(state["trace_run_id"])
    assert run.total_latency_ms is not None
    assert run.token_usage.total is not None
    assert run.status == "success"
    assert run.error is None


def test_traced_run_error_path():
    """LLM 全抛 → run status=error，retrieval 带 error 标记。"""
    biz = BusinessService()
    mem = MemoryService(backend=MemoryBackend(business_service=biz))
    store = InMemoryTraceStore()

    class _BoomLLM:
        def invoke(self, messages):
            raise RuntimeError("llm down")

    wf = TracedAgentWorkflow(
        llm=_BoomLLM(),
        retrieval_service=_FakeService(),
        memory=mem,
        trace_store=store,
    )
    state = wf.run("X200 ERR-203", thread_id="tt-err")
    run = store.get_run(state["trace_run_id"])
    assert run.status == "error"
    assert run.error is not None
    # 检索仍带了 error 标记（不静默）
    assert "agent_run_failed" in (state.get("retrieval") or {}).get("error", "")


# ================= API =================

def test_api_thread_trace():
    from fastapi.testclient import TestClient
    import web.api.v2_service as v2svc
    wf, store, _ = _wf()
    state = wf.run("X200 ERR-203", thread_id="api-1", customer_id="CUST-0001")
    run_id = state["trace_run_id"]

    # mock 两个绑定：模块级 get_trace_store + 已绑定的函数名
    with mock.patch.object(v2svc, "get_trace_store", return_value=store):
        from web.api.query_service import app
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/api/v2/threads/api-1/trace")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["thread_id"] == "api-1"
        assert len(body["runs"]) == 1
        assert body["runs"][0]["run_id"] == run_id
        assert len(body["runs"][0]["events"]) == 5

        # 查单次 run
        r2 = client.get(f"/api/v2/runs/{run_id}")
        assert r2.status_code == 200
        assert r2.json()["run"]["status"] == "success"

        # 缺 run → 404
        r3 = client.get("/api/v2/runs/run-NOPE")
        assert r3.status_code == 404


def test_api_thread_trace_empty():
    """无 run 的 thread 返回空 runs（不报错）。"""
    from fastapi.testclient import TestClient
    store = InMemoryTraceStore()
    import services.trace.trace_store as ts_module
    with mock.patch.object(ts_module, "get_trace_store", return_value=store):
        from web.api.query_service import app
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/api/v2/threads/empty-thread/trace")
        assert r.status_code == 200
        assert r.json()["runs"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
