"""
tests/test_retrieval_service_regression.py

回归锚点：把 RetrievalService 与既有 eval/retriever.py 的 4 个策略做对齐验证。

本测试【不依赖真实 Milvus/LLM】——通过 monkeypatch 注入
RetrieverEvaluator 的 4 个 search_* 方法，直接对比 RetrievalService
与 RetrieverEvaluator 在相同输入下的输出是否一致（结构 + 排序 + 数量）。

这样做的目的（对应 Step 1 要求）：
  1. 证明 Facade 封装没有改变现有检索行为的【接口/字段/排序语义】；
  2. 在真实 Milvus 环境跑时（把 monkeypatch 撤掉）可以直接对比 chunk_id 级别一致性；
  3. 若封装引入结果差异，本测试会立即 fail，先停下来分析，再决定是否放行。
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pytest

from core.retrieval import RetrievalService, RetrievalResult, RetrievalDocument
from eval.retriever import RetrieverEvaluator


class TestRetrievalServiceFacade:
    """Facade 基础契约"""

    def test_supported_strategies_match_eval_vocabulary(self):
        """Service 支持的策略名必须与 eval/retriever.py 的 4 个方法名一一对应"""
        from core.retrieval.retrieval_service import SUPPORTED_STRATEGIES
        expected = {"hybrid", "hybrid_rrf", "hybrid_rrf_rerank", "hyde_hybrid_rrf_rerank"}
        assert set(SUPPORTED_STRATEGIES) == expected

    def test_unknown_strategy_raises(self):
        svc = RetrievalService()
        with pytest.raises(ValueError):
            svc.search("anything", strategy="not_a_strategy")

    def test_result_is_structured_not_llm_answer(self):
        """RetrievalResult 不应包含 LLM 生成的 answer 字段（契约）"""
        r = RetrievalResult(documents=[], strategy="hybrid", latency=0.0)
        d = r.to_dict()
        assert "answer" not in d
        assert d["strategy"] == "hybrid"
        assert d["documents"] == []
        assert "metadata" in d


class TestRegressionAgainstEval:
    """与 eval/retriever.py 对齐：Service 输出 vs RetrieverEvaluator 输出"""

    @pytest.fixture
    def svc(self):
        """不实际连 Milvus，构造一个可注入 fake 客户端的 Service"""
        return RetrievalService(client=object())  # client 仅作占位，下面全部走 monkeypatch

    def _run_both(self, svc, monkeypatch, strategy, query="H3C LA2608 如何配置无线控制器", top_k=5):
        """跑 Service + RetrieverEvaluator 同一策略，对比结构。

        关键：必须把 Service._run_hybrid 也指向同一个 fake，
        否则 Service 走真实 Milvus、eval 走 fake，两边输入不一致，无法对齐。
        本测试只验证【Service 的编排/字段/排序语义】与 eval 一致，
        真实 Milvus 层面的 chunk_id 一致性另由 eval/run_eval.py 验证。
        """
        # 1) 给 Service 注入与 eval 相同的 fake hybrid
        svc_result = svc.search(query, strategy=strategy, top_k=top_k)

        # 2) eval 侧用同一个 fake（monkeypatch eval._search_hybrid）
        ev = RetrieverEvaluator()
        eval_result = {
            "hybrid": ev.search_hybrid(query, top_k=top_k),
            "hybrid_rrf": ev.search_hybrid_rrf(query, top_k=top_k),
            "hybrid_rrf_rerank": ev.search_hybrid_rrf_rerank(query, top_k=top_k),
            "hyde_hybrid_rrf_rerank": ev.search_hyde_hybrid_rrf_rerank(query, top_k=top_k),
        }[strategy]

        eval_docs = eval_result.get("results") if isinstance(eval_result, dict) else eval_result
        eval_chunk_ids = [d.get("chunk_id") for d in eval_docs if d.get("chunk_id")]
        svc_chunk_ids = [d.chunk_id for d in svc_result.documents if d.chunk_id]

        return svc_chunk_ids, eval_chunk_ids, svc_result

    def _fake_hybrid(self, prefix="c", top_k=10):
        """构造 10 条 fake hybrid 结果（chunk_id 前缀 + 0..9）"""
        return [{"chunk_id": f"{prefix}{i}", "content": f"content-{prefix}{i}",
                 "item_name": "", "file_title": "", "title": "", "score": 0.9 - 0.05 * i}
                for i in range(top_k)]

    def test_hybrid_matches_eval(self, svc, monkeypatch):
        # 两边都用同一份 fake：Service._run_hybrid + eval._search_hybrid
        fake = lambda q, top_k=10, item_names=None: self._fake_hybrid("c", top_k)
        monkeypatch.setattr("core.retrieval.retrieval_service.RetrievalService._run_hybrid",
                            lambda self, text, product_model, limit: fake(text, limit))
        monkeypatch.setattr(RetrieverEvaluator, "_search_hybrid",
                            lambda self, q, top_k=10, item_names=None: fake(q, top_k))
        svc_chunk_ids, eval_chunk_ids, svc_result = self._run_both(svc, monkeypatch, "hybrid", top_k=5)
        assert svc_chunk_ids == eval_chunk_ids  # 同一份数据，hybrid 是直出，顺序必须完全一致
        assert isinstance(svc_result, RetrievalResult)
        assert svc_result.strategy == "hybrid"

    def test_hybrid_rrf_matches_eval(self, svc, monkeypatch):
        fake = lambda q, top_k=10, item_names=None: self._fake_hybrid("a", top_k)
        monkeypatch.setattr("core.retrieval.retrieval_service.RetrievalService._run_hybrid",
                            lambda self, text, product_model, limit: fake(text, limit))
        monkeypatch.setattr(RetrieverEvaluator, "_search_hybrid",
                            lambda self, q, top_k=10, item_names=None: fake(q, top_k))
        svc_chunk_ids, eval_chunk_ids, svc_result = self._run_both(svc, monkeypatch, "hybrid_rrf", top_k=5)
        # RRF 融合：两边都用同样的 _rrf_merge 算法 + 同样输入 → 排序必须一致
        assert svc_chunk_ids == eval_chunk_ids

    def test_hybrid_rrf_rerank_matches_eval(self, svc, monkeypatch):
        fake = lambda q, top_k=10, item_names=None: self._fake_hybrid("r", top_k)
        monkeypatch.setattr("core.retrieval.retrieval_service.RetrievalService._run_hybrid",
                            lambda self, text, product_model, limit: fake(text, limit))
        monkeypatch.setattr(RetrieverEvaluator, "_search_hybrid",
                            lambda self, q, top_k=10, item_names=None: fake(q, top_k))
        # rerank_documents 是 DashScope 真实调用 → monkeypatch 两边共用同一个 fake 打分器
        import core.retrieval.retrieval_service as rs_mod
        fake_rerank = lambda q, docs: [0.95, 0.90, 0.85, 0.70, 0.55, 0.40, 0.30, 0.20, 0.10, 0.05][:len(docs)]
        monkeypatch.setattr(rs_mod, "rerank_documents", fake_rerank)
        monkeypatch.setattr("eval.retriever.rerank_documents", fake_rerank)
        svc_chunk_ids, eval_chunk_ids, svc_result = self._run_both(svc, monkeypatch, "hybrid_rrf_rerank", top_k=5)
        assert svc_chunk_ids == eval_chunk_ids
        assert isinstance(svc_result, RetrievalResult)

    def test_hyde_hybrid_rrf_rerank_matches_eval(self, svc, monkeypatch):
        fake = lambda q, top_k=10, item_names=None: self._fake_hybrid("h", top_k)
        monkeypatch.setattr("core.retrieval.retrieval_service.RetrievalService._run_hybrid",
                            lambda self, text, product_model, limit: fake(text, limit))
        monkeypatch.setattr(RetrieverEvaluator, "_search_hybrid",
                            lambda self, q, top_k=10, item_names=None: fake(q, top_k))
        import core.retrieval.retrieval_service as rs_mod
        fake_rerank = lambda q, docs: [0.9, 0.85, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0][:len(docs)]
        monkeypatch.setattr(rs_mod, "rerank_documents", fake_rerank)
        monkeypatch.setattr("eval.retriever.rerank_documents", fake_rerank)
        # HyDE 文档生成：两边都返回同一份 fake 文档，保证拼接后检索输入一致
        monkeypatch.setattr(rs_mod.RetrievalService, "_generate_hyde_doc", lambda self, q: "FAKE_HYDE_DOC")
        svc_chunk_ids, eval_chunk_ids, svc_result = self._run_both(svc, monkeypatch, "hyde_hybrid_rrf_rerank", top_k=5)
        assert svc_chunk_ids == eval_chunk_ids
        assert svc_result.metadata.get("hyde_doc") == "FAKE_HYDE_DOC"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
