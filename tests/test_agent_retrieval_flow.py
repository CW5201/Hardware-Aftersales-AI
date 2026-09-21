"""
tests/test_agent_retrieval_flow.py

Step 4 全链路测试：User Query → Triage → Retrieval → Evidence。

全部 mock（Triage LLM → fake；RetrievalService → fake），
不打真实 LLM / DashScope / Milvus / WebSearch。

覆盖（对应 Step 4 要求）：
  Case 1: 完整 Graph 可执行，State 里能拿到 triage + retrieval
  Case 2: Triage strategy=hybrid 正确透传到 Tool
  Case 3: Triage product_model=X200 正确透传到 Tool
  Case 4: Tool 返回有证据（documents 非空, error=None）
  Case 5: Tool 返回空 evidence（documents=[], error=None），Graph 正常结束
  Case 6: Tool 返回 error 标记，Graph State 保存该错误（不静默吞掉）
  Case 7: API POST /api/v2/agent/run 返回 triage + retrieval
"""

import sys
from pathlib import Path
from unittest import mock

import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.retrieval.retrieval_service import RetrievalDocument, RetrievalResult
from processor.agent_processor.main_graph import AgentWorkflow
from processor.agent_processor.state import TriageResult


class _FakeLLM:
    def __init__(self, content):
        self._content = content

    def invoke(self, messages):
        class _R:
            content = self._content
        return _R()


class _FakeService:
    """最小 RetrievalService 桩：可配置返回 evidence / 空 evidence / 异常。"""

    def __init__(self, mode="ok"):
        self.mode = mode
        self.calls = []

    def search(self, query, product_model=None, strategy="hybrid_rrf_rerank", top_k=5):
        self.calls.append({"query": query, "product_model": product_model,
                          "strategy": strategy, "top_k": top_k})
        if self.mode == "exception":
            raise RuntimeError("milvus down")
        if self.mode == "empty":
            return RetrievalResult(documents=[], strategy=strategy, latency=0.05, metadata={})
        return RetrievalResult(
            documents=[
                RetrievalDocument(chunk_id="c1", content="X200 维修手册段落", item_name="X200", score=0.93),
                RetrievalDocument(chunk_id="c2", content="ERR-203 说明", item_name="X200", score=0.88),
            ],
            strategy=strategy,
            latency=0.2,
            metadata={"mode": "ok"},
        )


def _make_wf(mode="ok", llm_json='{"intent":"fault_diagnosis","product_model":"X200",'
                                 '"error_code":"ERR-203","urgency":"medium",'
                                 '"retrieval_strategy":"hybrid"}'):
    return AgentWorkflow(llm=_FakeLLM(llm_json), retrieval_service=_FakeService(mode=mode))


# ---------- Case 1: 完整 Graph 可执行 ----------

def test_full_graph_produces_triage_and_retrieval():
    wf = _make_wf()
    state = wf.run("X200 出现 ERR-203")
    assert "triage" in state and "retrieval" in state
    assert isinstance(state["triage"], TriageResult)
    assert isinstance(state["retrieval"], dict)
    # evidence 结构字段齐全
    for key in ("documents", "scores", "strategy", "sources", "metadata", "latency", "error"):
        assert key in state["retrieval"]


# ---------- Case 2: strategy 透传 ----------

def test_triage_strategy_passes_to_tool():
    svc = _FakeService("ok")
    wf = AgentWorkflow(llm=_FakeLLM('{"intent":"fault_diagnosis","retrieval_strategy":"hybrid"}'),
                       retrieval_service=svc)
    wf.run("X200 报错")
    # Tool/Service 收到的 strategy 应等于 Triage 产出
    assert svc.calls[0]["strategy"] == "hybrid"


# ---------- Case 3: product_model 透传 ----------

def test_product_model_passes_to_tool():
    svc = _FakeService("ok")
    wf = AgentWorkflow(
        llm=_FakeLLM('{"intent":"fault_diagnosis","product_model":"X200","retrieval_strategy":"hybrid_rrf"}'),
        retrieval_service=svc,
    )
    wf.run("X200 报错")
    assert svc.calls[0]["product_model"] == "X200"


# ---------- Case 4: 有证据 ----------

def test_evidence_with_documents():
    wf = _make_wf("ok")
    state = wf.run("X200 报错")
    ev = state["retrieval"]
    assert ev["error"] is None
    assert len(ev["documents"]) == 2
    assert ev["documents"][0]["chunk_id"] == "c1"
    assert ev["scores"][0] == 0.93
    # sources 收集的是文档的 source 字段（"local"/"web"），不是商品名
    assert ev["sources"] == ["local"]


# ---------- Case 5: 空 evidence（正常无结果） ----------

def test_empty_evidence_is_not_error():
    wf = _make_wf("empty")
    state = wf.run("X200 报错")
    ev = state["retrieval"]
    assert ev["documents"] == []
    assert ev["error"] is None  # 关键：无结果 ≠ 故障
    assert ev["strategy"] in ("hybrid", "hybrid_rrf", "hybrid_rrf_rerank", "hyde_hybrid_rrf_rerank")


# ---------- Case 6: 检索故障（error 标记，不静默吞） ----------

def test_retrieval_failure_propagates_as_error():
    wf = _make_wf("exception")
    state = wf.run("X200 报错")
    ev = state["retrieval"]
    assert ev["error"] is not None
    assert "retrieval_failed" in ev["error"] or "tool_failed" in ev["error"]
    assert ev["documents"] == []


# ---------- Case 7: API 返回 triage + retrieval ----------

def test_api_agent_run_returns_triage_and_retrieval():
    from fastapi.testclient import TestClient
    import processor.agent_processor.main_graph as mg

    # 用一个固定的 fake service 替换全局检索服务，避免打真实 Milvus
    fake_service = _FakeService("ok")
    with mock.patch.object(
        mg, "AgentWorkflow", lambda: AgentWorkflow(llm=_FakeLLM(
            '{"intent":"fault_diagnosis","product_model":"X200","error_code":"ERR-203",'
            '"retrieval_strategy":"hybrid_rrf_rerank"}'
        ), retrieval_service=fake_service),
    ):
        from web.api.query_service import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/v2/agent/run", json={"query": "我的 X200 出现 ERR-203"})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "triage" in body and "retrieval" in body
    assert body["triage"]["product_model"] == "X200"
    assert body["retrieval"]["error"] is None
    assert len(body["retrieval"]["documents"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
