"""
tests/test_diagnosis_agent.py

Step 5 Diagnosis Agent 单元 + 集成测试（全部 mock，不打真实 LLM / Milvus）：

单元（纯函数）：
  1. parse_diagnosis_response 正常 JSON
  2. parse_diagnosis_response 坏 JSON → 保守 insufficient_evidence
  3. parse_diagnosis_response 未知 status 值 → 兜底 insufficient_evidence
  4. build_diagnosis_prompt 截断超长文档（max_docs）
  5. Hypothesis evidence_indices 归一化（非数字项被丢弃）

节点（mock LLM）：
  6. 有证据 → 正常诊断，hypotheses 非空
  7. 无证据（documents=[] error=None）→ insufficient_evidence，不编造
  8. 检索故障（error!=None）→ retrieval_failed + need_human_review
  9. LLM 声称 diagnosed 但 hypotheses 全空 → 降级 insufficient_evidence + need_human_review
 10. LLM 调用异常 → 保守 insufficient_evidence + need_human_review
 11. 缺 triage（state 无 triage）时 diagnosis 节点仍可运行（用空 triage 兜底）

不碰 C 扩展（RetrievalService 用 fake；diagnosis 只依赖 triage + retrieval 两个字段）。
"""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from processor.agent_processor.nodes.diagnosis import (
    DiagnosisNode,
    build_diagnosis_prompt,
    parse_diagnosis_response,
)
from processor.agent_processor.state import DiagnosisResult, Hypothesis, TriageResult


# ---------------- fake 件 ----------------

class _FakeLLM:
    def __init__(self, content=None, exception=None):
        self._content = content
        self._exception = exception

    def invoke(self, messages):
        if self._exception:
            raise self._exception

        class _R:
            pass

        r = _R()
        r.content = self._content or "{}"
        return r


def _triage(**kw):
    defaults = {
        "intent": "fault_diagnosis",
        "product_model": "X200",
        "error_code": "ERR-203",
        "urgency": "medium",
        "retrieval_strategy": "hybrid",
    }
    defaults.update(kw)
    return TriageResult(**defaults)


def _evidence_ok():
    return {
        "documents": [
            {"chunk_id": "c1", "content": "X200 电源模块规格", "score": 0.91, "source": "local"},
            {"chunk_id": "c2", "content": "ERR-203 含义：供电电压异常", "score": 0.87, "source": "local"},
        ],
        "scores": [0.91, 0.87],
        "strategy": "hybrid",
        "sources": ["local"],
        "metadata": {},
        "latency": 0.2,
        "error": None,
    }


def _evidence_empty():
    return {
        "documents": [],
        "scores": [],
        "strategy": "hybrid",
        "sources": [],
        "metadata": {},
        "latency": 0.1,
        "error": None,
    }


def _evidence_error():
    return {
        "documents": [],
        "scores": [],
        "strategy": "hybrid",
        "sources": [],
        "metadata": {},
        "latency": 0.0,
        "error": "retrieval_failed: MilvusException: down",
    }


GOOD_DIAG = {
    "diagnosis_status": "diagnosed",
    "hypotheses": [
        {"text": "供电电压异常", "evidence_indices": [0, 1], "confidence": "high"},
    ],
    "evidence": ["X200 电源模块规格", "ERR-203 含义：供电电压异常"],
    "diagnosis_result": "电源模块供电电压异常，建议更换",
    "recommended_actions": ["更换电源模块"],
    "need_more_information": [],
    "need_human_review": False,
}


# ================= 单元：parse_diagnosis_response =================

def test_parse_good_json():
    d = parse_diagnosis_response(json.dumps(GOOD_DIAG, ensure_ascii=False))
    assert isinstance(d, DiagnosisResult)
    assert d.diagnosis_status == "diagnosed"
    assert len(d.hypotheses) == 1
    assert isinstance(d.hypotheses[0], Hypothesis)
    assert d.hypotheses[0].evidence_indices == [0, 1]
    assert d.diagnosis_result == "电源模块供电电压异常，建议更换"
    assert d.need_human_review is False


def test_parse_bad_json_falls_back():
    d = parse_diagnosis_response("这不是JSON{{")
    assert isinstance(d, DiagnosisResult)
    assert d.diagnosis_status == "insufficient_evidence"
    assert d.diagnosis_result is None


def test_parse_unknown_status_falls_back():
    payload = json.loads(json.dumps(GOOD_DIAG))
    payload["diagnosis_status"] = "bizzar"
    d = parse_diagnosis_response(json.dumps(payload, ensure_ascii=False))
    assert d.diagnosis_status == "insufficient_evidence"


def test_parse_indices_normalized():
    payload = json.loads(json.dumps(GOOD_DIAG))
    payload["hypotheses"][0]["evidence_indices"] = [0, "x", 3, ""]
    d = parse_diagnosis_response(json.dumps(payload, ensure_ascii=False))
    assert d.hypotheses[0].evidence_indices == [0, 3]


def test_parse_markdown_fenced_json():
    raw = "```json\n" + json.dumps(GOOD_DIAG, ensure_ascii=False) + "\n```"
    d = parse_diagnosis_response(raw)
    assert d.diagnosis_status == "diagnosed"


# ================= 单元：build_diagnosis_prompt =================

def test_prompt_truncates_docs():
    ev = _evidence_ok()
    ev["documents"].extend(
        [{"chunk_id": f"c{i}", "content": "x" * 2000, "score": 0.5, "source": "local"} for i in range(5, 20)]
    )
    system, user = build_diagnosis_prompt("X200 报错", _triage(), ev)
    # 只取前 8 条文档进 prompt
    assert "X200 电源模块规格" in user
    assert '"index": 7' in user
    assert '"index": 8' not in user
    assert "retrieval_strategy" in user  # triage JSON 里有该字段


# ================= 节点：有证据 / 无证据 / 故障 =================

def test_node_diagnosed_with_evidence():
    node = DiagnosisNode(llm=_FakeLLM(json.dumps(GOOD_DIAG, ensure_ascii=False)))
    state = {"user_query": "X200 报错", "triage": _triage(), "retrieval": _evidence_ok()}
    out = node(state)
    assert out["diagnosis"].diagnosis_status == "diagnosed"
    assert len(out["diagnosis"].hypotheses) == 1


def test_node_insufficient_no_evidence():
    node = DiagnosisNode(llm=_FakeLLM(json.dumps(GOOD_DIAG, ensure_ascii=False)))
    state = {"user_query": "X200 报错", "triage": _triage(), "retrieval": _evidence_empty()}
    out = node(state)
    assert out["diagnosis"].diagnosis_status == "insufficient_evidence"
    assert out["diagnosis"].diagnosis_result is None
    # need_more_information 来自 triage.missing_information
    assert out["diagnosis"].need_more_information == []


def test_node_retrieval_failed():
    node = DiagnosisNode(llm=_FakeLLM(json.dumps(GOOD_DIAG, ensure_ascii=False)))
    state = {"user_query": "X200 报错", "triage": _triage(), "retrieval": _evidence_error()}
    out = node(state)
    d = out["diagnosis"]
    assert d.diagnosis_status == "retrieval_failed"
    assert d.need_human_review is True
    assert d.hypotheses == []
    assert d.diagnosis_result is None


def test_node_downgrade_when_diagnosed_but_no_hypotheses():
    bad = json.loads(json.dumps(GOOD_DIAG))
    bad["hypotheses"] = []
    node = DiagnosisNode(llm=_FakeLLM(json.dumps(bad, ensure_ascii=False)))
    state = {"user_query": "X200 报错", "triage": _triage(), "retrieval": _evidence_ok()}
    d = node(state)["diagnosis"]
    assert d.diagnosis_status == "insufficient_evidence"
    assert d.diagnosis_result is None
    assert d.need_human_review is True


def test_node_llm_exception_fallback():
    node = DiagnosisNode(llm=_FakeLLM(exception=RuntimeError("llm down")))
    state = {"user_query": "X200 报错", "triage": _triage(), "retrieval": _evidence_ok()}
    d = node(state)["diagnosis"]
    assert d.diagnosis_status == "insufficient_evidence"
    assert d.need_human_review is True
    assert "llm_unavailable" in d.need_more_information


def test_node_missing_triage_still_runs():
    node = DiagnosisNode(llm=_FakeLLM(json.dumps(GOOD_DIAG, ensure_ascii=False)))
    state = {"user_query": "X200 报错", "retrieval": _evidence_ok()}  # 无 triage
    d = node(state)["diagnosis"]
    assert isinstance(d, DiagnosisResult)
    assert d.diagnosis_status == "diagnosed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
