"""
tests/test_agent_retrieval_flow.py

Step 4 全链路测试：User Query → Triage → Retrieval → Evidence。
Step 5 起 Graph 扩展为 Triage → Retrieval → Diagnosis → END，
本文件的 fake LLM 需按节点顺序返回 triage JSON / diagnosis JSON。

全部 mock（Triage LLM → fake；RetrievalService → fake），
不打真实 LLM / DashScope / Milvus / WebSearch。

覆盖（对应 Step 4 要求）：
  Case 1: 完整 Graph 可执行，State 里能拿到 triage + retrieval（+ diagnosis）
  Case 2: Triage strategy=hybrid 正确透传到 Tool
  Case 3: Triage product_model=X200 正确透传到 Tool
  Case 4: Tool 返回有证据（documents 非空, error=None）
  Case 5: Tool 返回空 evidence（documents=[], error=None），Graph 正常结束
  Case 6: Tool 返回 error 标记，Graph State 保存该错误（不静默吞掉）
  Case 7: API POST /api/v2/agent/run 返回 triage + retrieval + diagnosis
"""

import sys
from pathlib import Path
from unittest import mock

import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.retrieval.retrieval_service import RetrievalDocument, RetrievalResult
from processor.agent_processor.main_graph import AgentWorkflow
from processor.agent_processor.state import DiagnosisResult, TriageResult


DEFAULT_DIAGNOSIS_JSON = (
    '{"diagnosis_status":"diagnosed",'
    '"hypotheses":[{"text":"ERR-203 由电源模块供电不足引起","evidence_indices":[0,1],'
    '"confidence":"medium"}],'
    '"evidence":["X200 维修手册段落","ERR-203 说明"],'
    '"diagnosis_result":"电源模块供电不足，建议更换电源模块",'
    '"recommended_actions":["断电","更换电源模块","上电复测"],'
    '"need_more_information":[],"need_human_review":false}'
)


class _FakeLLM:
    """按节点顺序返回固定 JSON：第 1 次 = triage，第 2 次 = diagnosis。"""

    def __init__(self, triage_json, diagnosis_json=DEFAULT_DIAGNOSIS_JSON):
        self._answers = [triage_json, diagnosis_json]
        self.calls = 0

    def invoke(self, messages):
        idx = min(self.calls, len(self._answers) - 1)
        self.calls += 1
        content = self._answers[idx]

        class _R:
            pass

        r = _R()
        r.content = content
        return r


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
    assert "triage" in state and "retrieval" in state and "diagnosis" in state
    assert isinstance(state["triage"], TriageResult)
    assert isinstance(state["retrieval"], dict)
    assert isinstance(state["diagnosis"], DiagnosisResult)
    # evidence 结构字段齐全
    for key in ("documents", "scores", "strategy", "sources", "metadata", "latency", "error"):
        assert key in state["retrieval"]


# ---------- Case 1b (Step 5): 有证据 → 正常诊断 ----------

def test_diagnosis_ok_with_evidence():
    wf = _make_wf("ok")
    state = wf.run("X200 出现 ERR-203")
    d = state["diagnosis"]
    assert d.diagnosis_status == "diagnosed"
    assert d.diagnosis_result == "电源模块供电不足，建议更换电源模块"
    assert len(d.hypotheses) == 1
    assert d.hypotheses[0].evidence_indices == [0, 1]
    assert not d.need_human_review


# ---------- Case 1c (Step 5): 无证据 → insufficient_evidence（不编造） ----------

def test_diagnosis_insufficient_when_no_evidence():
    wf = _make_wf("empty")
    state = wf.run("X200 报错")
    d = state["diagnosis"]
    assert d.diagnosis_status == "insufficient_evidence"
    assert d.diagnosis_result is None
    assert d.hypotheses == []
    assert not d.need_human_review


# ---------- Case 1d (Step 5): 检索故障 → retrieval_failed + need_human_review ----------

def test_diagnosis_retrieval_failed():
    wf = _make_wf("exception")
    state = wf.run("X200 报错")
    d = state["diagnosis"]
    assert d.diagnosis_status == "retrieval_failed"
    assert d.need_human_review is True
    assert d.hypotheses == []
    assert d.diagnosis_result is None


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


# ---------- Case 7: API 返回 triage + retrieval + diagnosis ----------

def test_api_agent_run_returns_triage_and_retrieval():
    from fastapi.testclient import TestClient
    import processor.agent_processor.main_graph as mg
    from services.memory.memory_service import MemoryService, MemoryBackend
    from services.business.business_service import BusinessService

    # 用一个固定的 fake service 替换全局检索服务，避免打真实 Milvus
    fake_service = _FakeService("ok")
    # 隔离的 memory（不污染全局单例）
    iso_mem = MemoryService(backend=MemoryBackend(business_service=BusinessService(seed=False)))

    def _mk_wf():
        return AgentWorkflow(
            llm=_FakeLLM(
                '{"intent":"fault_diagnosis","product_model":"X200","error_code":"ERR-203",'
                '"retrieval_strategy":"hybrid_rrf_rerank"}'
            ),
            retrieval_service=fake_service,
            memory=iso_mem,
        )

    with mock.patch.object(mg, "AgentWorkflow", _mk_wf):
        from web.api.query_service import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/v2/agent/run", json={"query": "我的 X200 出现 ERR-203"})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "triage" in body and "retrieval" in body and "diagnosis" in body and "memory" in body
    assert body["triage"]["product_model"] == "X200"
    assert body["retrieval"]["error"] is None
    assert len(body["retrieval"]["documents"]) == 2
    assert body["diagnosis"]["diagnosis_status"] == "diagnosed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
