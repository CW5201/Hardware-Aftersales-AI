"""
tests/test_adaptive_retrieval.py

Step 13 Adaptive Retrieval 测试（纯函数，不打真实检索 / LLM）：

  策略判定（pick_retrieval_strategy / choose_first_strategy）：
    1. 简单知识问题 → hybrid（默认最简）
    2. 复杂故障诊断 → hybrid_rrf_rerank
    3. 低置信度证据（有文档但最高分 < 0.3）→ hyde_hybrid_rrf_rerank
    4. 知识不足（复杂 + 无证据）→ hyde + suggest_web_search=True
    5. 检索带 error → 低置信 → hyde
    6. 所有决策 strategy ∈ SUPPORTED_STRATEGIES（不另造词汇）

  RetrievalNode 集成：
    7. 简单 triage 的 strategy 被 adaptive 覆盖为 hybrid（不是 triage 默认的 hybrid_rrf_rerank）
    8. 复杂 triage 的 strategy 被 adaptive 覆盖为 hybrid_rrf_rerank
    9. retrieval evidence 里带 adaptive 决策字段
"""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.retrieval.adaptive import (
    SUPPORTED_STRATEGIES,
    choose_first_strategy,
    pick_retrieval_strategy,
)
from processor.agent_processor.state import TriageResult


def _triage(intent="knowledge_question", model=None, err=None, strategy="hybrid_rrf_rerank",
            missing=None):
    return TriageResult(
        intent=intent,
        product_model=model,
        error_code=err,
        urgency="low",
        retrieval_strategy=strategy,
        missing_information=missing or [],
    )


# ================= 纯函数判定 =================

def test_simple_question_defaults_to_hybrid():
    d = choose_first_strategy(_triage(intent="knowledge_question"))
    assert d.strategy == "hybrid"
    assert d.suggest_web_search is False
    assert d.confidence == "high"


def test_complex_fault_diagnosis_to_full_pipeline():
    d = choose_first_strategy(_triage(intent="fault_diagnosis", model="X200", err="ERR-203"))
    assert d.strategy == "hybrid_rrf_rerank"
    assert d.suggest_web_search is False


def test_low_confidence_evidence_to_hyde():
    triage = _triage(intent="fault_diagnosis", model="X200", err="ERR-203")
    retrieval = {"documents": [{"score": 0.1}], "error": None}
    d = pick_retrieval_strategy(triage, retrieval=retrieval)
    # 有证据但最高分 0.1 < 0.3 → 低置信 → HyDE
    assert d.strategy == "hyde_hybrid_rrf_rerank"
    assert d.confidence == "low"


def test_knowledge_insufficient_suggests_web_search():
    triage = _triage(intent="fault_diagnosis", model="X200", err="ERR-999")
    retrieval = {"documents": [], "error": None}
    d = pick_retrieval_strategy(triage, retrieval=retrieval)
    assert d.strategy == "hyde_hybrid_rrf_rerank"
    assert d.suggest_web_search is True


def test_retrieval_error_is_low_confidence():
    triage = _triage(intent="fault_diagnosis")
    retrieval = {"documents": [], "error": "retrieval_failed: Milvus down"}
    d = pick_retrieval_strategy(triage, retrieval=retrieval)
    assert d.strategy == "hyde_hybrid_rrf_rerank"


def test_all_decisions_use_supported_strategies():
    cases = [
        choose_first_strategy(_triage()),
        choose_first_strategy(_triage(intent="fault_diagnosis", model="X", err="E")),
        pick_retrieval_strategy(_triage(), retrieval={"documents": [{"score": 0.1}], "error": None}),
        pick_retrieval_strategy(_triage(intent="fault_diagnosis"), retrieval={"documents": [], "error": None}),
    ]
    for d in cases:
        assert d.strategy in SUPPORTED_STRATEGIES


# ================= RetrievalNode 集成 =================

def _fake_service(mode="ok"):
    from core.retrieval.retrieval_service import RetrievalDocument, RetrievalResult

    class _S:
        def search(self, query, product_model=None, strategy="hybrid_rrf_rerank", top_k=5):
            if mode == "empty":
                return RetrievalResult(documents=[], strategy=strategy, latency=0.1, metadata={})
            return RetrievalResult(
                documents=[
                    RetrievalDocument(chunk_id="c1", content="证据", item_name="X200", score=0.9),
                ],
                strategy=strategy, latency=0.1, metadata={},
            )

    return _S()


def test_node_overrides_simple_strategy_to_hybrid():
    from processor.agent_processor.nodes.retrieval import RetrievalNode

    node = RetrievalNode(service=_fake_service("ok"))
    state = {"user_query": "X200 报错", "triage": _triage(intent="knowledge_question")}
    out = node(state)
    # 简单问题 → adaptive 覆盖为 hybrid（不是 triage 默认的 hybrid_rrf_rerank）
    assert out["retrieval"]["strategy"] == "hybrid"
    assert out["retrieval"]["adaptive"]["strategy"] == "hybrid"


def test_node_keeps_complex_strategy():
    from processor.agent_processor.nodes.retrieval import RetrievalNode

    node = RetrievalNode(service=_fake_service("ok"))
    state = {
        "user_query": "X200 报错",
        "triage": _triage(intent="fault_diagnosis", model="X200", err="ERR-203"),
    }
    out = node(state)
    assert out["retrieval"]["strategy"] == "hybrid_rrf_rerank"


def test_retrieval_evidence_has_adaptive_field():
    from processor.agent_processor.nodes.retrieval import RetrievalNode

    node = RetrievalNode(service=_fake_service("ok"))
    state = {"user_query": "X200 报错", "triage": _triage()}
    out = node(state)
    assert "adaptive" in out["retrieval"]
    for key in ("strategy", "reason", "suggest_web_search", "confidence"):
        assert key in out["retrieval"]["adaptive"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
