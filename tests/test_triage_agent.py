"""
tests/test_triage_agent.py

Triage Agent 单元测试（不依赖真实 LLM/Milvus）：
  - parse_triage_response 的健壮性（正常 / 坏 JSON / 未知 strategy）
  - TriageNode 在 mock LLM 下的结构化输出
  - AgentWorkflow 端到端（mock LLM）返回 TriageResult
  - POST /api/v2/agent/triage（mock LLM）返回 200 + TriageResult

通过向 TriageNode / AgentWorkflow 注入 fake LLM，避免打真实 API。
"""

import sys
from pathlib import Path
from unittest import mock

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pytest

from processor.agent_processor.nodes.triage import TriageNode, parse_triage_response
from processor.agent_processor.state import TriageResult
from core.retrieval.retrieval_service import SUPPORTED_STRATEGIES


class FakeLLM:
    """最小 LLM 桩：返回固定文本，模拟 LLM 输出。"""

    def __init__(self, content: str):
        self._content = content

    def invoke(self, messages):
        class _Resp:
            content = self._content
        return _Resp()


class _NoopService:
    """最小 RetrievalService 桩：返回空证据，不触碰 C 扩展/Milvus。"""

    def search(self, query, product_model=None, strategy="hybrid_rrf_rerank", top_k=5):
        from core.retrieval.retrieval_service import RetrievalResult
        return RetrievalResult(documents=[], strategy=strategy, latency=0.0, metadata={})


def test_parse_valid_json():
    raw = (
        '{"intent": "fault_diagnosis", "product_model": "X200", '
        '"device_id": null, "error_code": "ERR-203", "symptom": "重启后报错", '
        '"urgency": "medium", "missing_information": ["firmware_version"], '
        '"retrieval_strategy": "hybrid_rrf_rerank"}'
    )
    r = parse_triage_response(raw)
    assert isinstance(r, TriageResult)
    assert r.intent == "fault_diagnosis"
    assert r.product_model == "X200"
    assert r.error_code == "ERR-203"
    assert r.missing_information == ["firmware_version"]
    assert r.retrieval_strategy in SUPPORTED_STRATEGIES


def test_parse_handles_code_fence():
    raw = '```json\n{"intent": "device_query", "product_model": "HAK180", ' \
          '"retrieval_strategy": "hybrid"}\n```'
    r = parse_triage_response(raw)
    assert r.intent == "device_query"
    assert r.product_model == "HAK180"
    assert r.retrieval_strategy == "hybrid"


def test_parse_invalid_json_falls_back_unknown():
    r = parse_triage_response("这不是 JSON {{{")
    assert r.intent == "unknown"
    # 兜底后 retrieval_strategy 仍必须落在合法集合内
    assert r.retrieval_strategy in SUPPORTED_STRATEGIES


def test_parse_unknown_strategy_coerced_to_default():
    raw = '{"intent": "fault_diagnosis", "retrieval_strategy": "some_made_up_strategy"}'
    r = parse_triage_response(raw)
    assert r.retrieval_strategy in SUPPORTED_STRATEGIES  # 未知值被纠偏


def test_triage_node_uses_injected_llm():
    raw = '{"intent": "fault_diagnosis", "product_model": "X200", ' \
          '"error_code": "ERR-203", "urgency": "low", "retrieval_strategy": "hyde_hybrid_rrf_rerank"}'
    node = TriageNode(llm=FakeLLM(raw))
    state = node({"user_query": "X200 报 ERR-203"})
    triage = state["triage"]
    assert isinstance(triage, TriageResult)
    assert triage.intent == "fault_diagnosis"
    assert triage.product_model == "X200"
    assert triage.error_code == "ERR-203"
    assert triage.retrieval_strategy == "hyde_hybrid_rrf_rerank"


def test_agent_workflow_returns_triage():
    from processor.agent_processor.main_graph import AgentWorkflow
    raw = '{"intent": "knowledge_question", "product_model": "HAK180", ' \
          '"retrieval_strategy": "hybrid_rrf"}'
    # Step 4 后 wf.run() 会走 Triage→Retrieval；注入 fake service 避免真调 Milvus/C 扩展
    wf = AgentWorkflow(llm=FakeLLM(raw), retrieval_service=_NoopService())
    state = wf.run("HAK180 怎么调温度")
    assert isinstance(state["triage"], TriageResult)
    assert state["triage"].intent == "knowledge_question"
    assert "user_query" in state


def test_v2_triage_endpoint_returns_200():
    from fastapi.testclient import TestClient
    # 构造一个"已注入 fake LLM"的 AgentWorkflow，patch 掉模块级名字，走真实 Graph
    raw = '{"intent": "fault_diagnosis", "product_model": "X200", ' \
          '"error_code": "ERR-203", "urgency": "medium", "retrieval_strategy": "hybrid_rrf_rerank"}'
    import processor.agent_processor.main_graph as mg
    wf_stub = _PatchedWorkflow(FakeLLM(raw))
    with mock.patch.object(mg, "AgentWorkflow", lambda: wf_stub):
        from web.api.query_service import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/api/v2/agent/triage",
            json={"query": "我的X200出现ERR-203，重启之后还是报错"},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["intent"] == "fault_diagnosis"
    assert body["error_code"] == "ERR-203"
    assert body["retrieval_strategy"] in SUPPORTED_STRATEGIES


class _PatchedWorkflow:
    """stub：委托给注入 fake LLM 的真实 AgentWorkflow。"""

    def __init__(self, llm):
        from processor.agent_processor.main_graph import AgentWorkflow
        self._wf = AgentWorkflow(llm=llm)

    def run(self, query):
        return self._wf.run(query)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
