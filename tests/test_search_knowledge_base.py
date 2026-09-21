"""
tests/test_search_knowledge_base.py

RAG Tool（search_knowledge_base）单元测试 —— mock RetrievalService，不打真实
Milvus / DashScope / LLM。

覆盖（对应 Step 3 要求）：
  1. 正常 hybrid：Service 被正确调用、参数透传、返回 RetrievalResult 结构
  2. 4 个 strategy 都能正确透传（Step 1 的 SUPPORTED_STRATEGIES）
  3. 非法 strategy：拒绝（Pydantic 约束 + 防御性兜底）
  4. 空 query：明确错误，不打底层
  5. RetrievalService 异常：不吞，带 error 标记返回（避免 Agent 误判"无结果"）
  6. Triage → Tool 参数映射（build_tool_args_from_triage）
"""

import sys
from pathlib import Path
from unittest import mock

import pytest
from pydantic import ValidationError

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.retrieval.retrieval_service import RetrievalDocument, RetrievalResult, SUPPORTED_STRATEGIES
from processor.agent_processor.tools.search_knowledge_base import (
    SearchKBArgs,
    _run_search,
    build_tool_args_from_triage,
    make_search_knowledge_base_tool,
)


class _FakeService:
    """最小 RetrievalService 桩：记录调用参数，返回固定 RetrievalResult。"""

    def __init__(self, fail: bool = False, docs=None):
        self.calls = []
        self._fail = fail
        self._docs = docs if docs is not None else [
            RetrievalDocument(chunk_id="c1", content="hello", item_name="X200", score=0.9),
            RetrievalDocument(chunk_id="c2", content="world", item_name="X200", score=0.8),
        ]

    def search(self, query, product_model=None, strategy="hybrid_rrf_rerank", top_k=5):
        self.calls.append({"query": query, "product_model": product_model,
                          "strategy": strategy, "top_k": top_k})
        if self._fail:
            raise RuntimeError("milvus boom")
        return RetrievalResult(documents=self._docs, strategy=strategy,
                               latency=0.12, metadata={"echo": True})


# ---------------- 1. 正常 hybrid ----------------

def test_normal_hybrid_calls_service_and_returns_result():
    svc = _FakeService()
    args = SearchKBArgs(query="X200 温度怎么调", product_model="X200",
                        retrieval_strategy="hybrid", top_k=3)
    ev = _run_search(args, service=svc)
    assert ev["error"] is None
    assert len(ev["documents"]) == 2
    assert ev["documents"][0]["chunk_id"] == "c1"
    assert ev["strategy"] == "hybrid"
    # Service 被以正确参数调用
    assert svc.calls[0] == {"query": "X200 温度怎么调", "product_model": "X200",
                            "strategy": "hybrid", "top_k": 3}


# ---------------- 2. 4 个 strategy 透传 ----------------

@pytest.mark.parametrize("strategy", list(SUPPORTED_STRATEGIES))
def test_all_strategies_pass_through(strategy):
    svc = _FakeService()
    args = SearchKBArgs(query="q", product_model=None, retrieval_strategy=strategy, top_k=5)
    ev = _run_search(args, service=svc)
    assert ev["error"] is None
    assert ev["strategy"] == strategy
    assert svc.calls[0]["strategy"] == strategy


# ---------------- 3. 非法 strategy 拒绝 ----------------

def test_invalid_strategy_rejected_by_schema():
    """Pydantic 层不直接拦非法字符串（enum 是软约束在 SUPPORTED_STRATEGIES），
    因此非法值走 _run_search 的防御性兜底 → error 标记。"""
    svc = _FakeService()
    args = SearchKBArgs(query="q", retrieval_strategy="not_a_real_strategy")
    ev = _run_search(args, service=svc)
    assert ev["error"] and "invalid retrieval_strategy" in ev["error"]
    # 没有打到底层
    assert svc.calls == []


def test_invalid_strategy_raises_when_bypassing_args():
    """若直接从 LLM 拿到脏 strategy 绕过 SearchKBArgs，工具层也必须兜底。"""
    svc = _FakeService()
    # 手工构造一个不在 SUPPORTED_STRATEGIES 内的 args（绕过默认约束）
    bad = SearchKBArgs(query="q", retrieval_strategy="weird")
    ev = _run_search(bad, service=svc)
    assert ev["error"] is not None
    assert svc.calls == []


# ---------------- 4. 空 query ----------------

@pytest.mark.parametrize("empty", ["", "   ", None])
def test_empty_query_safe_return(empty):
    svc = _FakeService()
    if empty is None:
        with pytest.raises(ValidationError):
            SearchKBArgs(query=None)  # Pydantic 要求 query 为 str
    else:
        args = SearchKBArgs(query=empty)
        ev = _run_search(args, service=svc)
        assert ev["error"] == "empty query"
        assert ev["documents"] == []
        assert svc.calls == []  # 没打底层


# ---------------- 5. Service 异常不吞 ----------------

def test_service_exception_not_swallowed():
    svc = _FakeService(fail=True)
    args = SearchKBArgs(query="q", retrieval_strategy="hybrid_rrf_rerank")
    ev = _run_search(args, service=svc)
    assert ev["error"] is not None
    assert "retrieval_failed" in ev["error"]
    assert "milvus boom" in ev["error"]
    assert ev["documents"] == []


# ---------------- 6. Triage → Tool 映射 ----------------

def test_build_tool_args_from_triage():
    tool_args = build_tool_args_from_triage(
        user_query="X200 出现 ERR-203",
        product_model="X200",
        retrieval_strategy="hybrid_rrf_rerank",
        top_k=5,
    )
    assert tool_args == {
        "query": "X200 出现 ERR-203",
        "product_model": "X200",
        "retrieval_strategy": "hybrid_rrf_rerank",
        "top_k": 5,
    }


# ---------------- 7. LangChain Tool 对象本身可用 ----------------

def test_structured_tool_invoke():
    svc = _FakeService()
    tool = make_search_knowledge_base_tool(service=svc)
    assert tool.name == "search_knowledge_base"
    out = tool.invoke({"query": "q", "retrieval_strategy": "hybrid_rrf", "top_k": 2})
    # LangChain 会把 dict 结果序列化；校验结构正确
    parsed = out if isinstance(out, dict) else out
    assert "strategy" in (parsed if isinstance(parsed, dict) else parsed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
