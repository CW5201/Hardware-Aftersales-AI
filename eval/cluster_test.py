"""
RRF 和 Cliff Cutoff 的单元测试。
不依赖任何外部服务，可直接运行。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestRRF:
    """Reciprocal Rank Fusion 单元测试"""

    def test_basic_merge(self):
        """测试基础 RRF 融合"""
        from processor.query_processor.nodes.node_rrf import NodeRrf

        rrf = NodeRrf()
        inputs = [
            (
                [
                    {"chunk_id": "c1", "content": "文档1"},
                    {"chunk_id": "c2", "content": "文档2"},
                    {"chunk_id": "c3", "content": "文档3"},
                ],
                1.0,
            ),
            (
                [
                    {"chunk_id": "c2", "content": "文档2"},
                    {"chunk_id": "c4", "content": "文档4"},
                    {"chunk_id": "c1", "content": "文档1"},
                ],
                1.0,
            ),
        ]

        result = rrf._rrf_merge(inputs, k=60, max_results=5)
        result_ids = [doc["chunk_id"] for doc, _ in result]

        # c2 在两路都排第1，应该得分最高
        # c1 在第1路排第1，第2路排第3
        # c3 只在第1路，排第3
        # c4 只在第2路，排第2
        assert result_ids[0] == "c2", f"Expected c2 first, got {result_ids[0]}"
        assert result_ids[1] == "c1", f"Expected c1 second, got {result_ids[1]}"

    def test_single_source(self):
        """测试单路检索"""
        from processor.query_processor.nodes.node_rrf import NodeRrf

        rrf = NodeRrf()
        inputs = [
            (
                [
                    {"chunk_id": "c1", "content": "A"},
                    {"chunk_id": "c2", "content": "B"},
                    {"chunk_id": "c3", "content": "C"},
                ],
                1.0,
            ),
        ]

        result = rrf._rrf_merge(inputs, k=60, max_results=3)
        result_ids = [doc["chunk_id"] for doc, _ in result]
        assert result_ids == ["c1", "c2", "c3"], f"Expected ordered list, got {result_ids}"

    def test_empty_input(self):
        """测试空输入"""
        from processor.query_processor.nodes.node_rrf import NodeRrf

        rrf = NodeRrf()
        result = rrf._rrf_merge([], k=60, max_results=5)
        assert result == []

    def test_max_results_limit(self):
        """测试 max_results 限制"""
        from processor.query_processor.nodes.node_rrf import NodeRrf

        rrf = NodeRrf()
        inputs = [
            (
                [
                    {"chunk_id": f"c{i}", "content": f"Doc{i}"}
                    for i in range(10)
                ],
                1.0,
            ),
        ]

        result = rrf._rrf_merge(inputs, k=60, max_results=3)
        assert len(result) == 3, f"Expected 3 results, got {len(result)}"

    def test_score_formula(self):
        """验证 RRF 分数公式：weight / (k + rank)"""
        from processor.query_processor.nodes.node_rrf import NodeRrf

        rrf = NodeRrf()
        # 单路：rank 1 → score = 1/(60+1), rank 2 → score = 1/(60+2)
        inputs = [(
            [{"chunk_id": "c1", "content": "A"}, {"chunk_id": "c2", "content": "B"}],
            1.0,
        )]
        result = rrf._rrf_merge(inputs, k=60, max_results=2)

        score_c1 = result[0][1]
        score_c2 = result[1][1]
        assert score_c1 > score_c2, "Higher rank should have higher score"
        assert abs(score_c1 - 1.0 / 61) < 1e-6
        assert abs(score_c2 - 1.0 / 62) < 1e-6


class TestCliffCutoff:
    """断崖截断单元测试"""

    def test_cliff_detected(self):
        """测试检测到断崖时正确截断"""
        from processor.query_processor.nodes.node_rerank import NodeRerank

        rerank = NodeRerank()
        docs = [
            {"chunk_id": "c1", "score": 0.95},
            {"chunk_id": "c2", "score": 0.90},
            {"chunk_id": "c3", "score": 0.50},  # 断崖: 0.90->0.50 gap=0.40 > 0.30
            {"chunk_id": "c4", "score": 0.45},
            {"chunk_id": "c5", "score": 0.40},
        ]
        result = rerank._step_3_cliff_cutoff(docs)
        result_ids = [d["chunk_id"] for d in result]
        # 在 c2→c3 处检测到断崖（gap=0.40>0.30），cutoff_pos=2，保留 c1, c2
        assert "c1" in result_ids, f"Expected c1 in result, got {result_ids}"
        assert "c2" in result_ids, f"Expected c2 in result, got {result_ids}"
        assert "c3" not in result_ids, f"Expected c3 NOT in result (cliff cutoff), got {result_ids}"

    def test_no_cliff(self):
        """测试没有断崖时保留全部（不超过 max_topk）"""
        from processor.query_processor.nodes.node_rerank import NodeRerank

        rerank = NodeRerank()
        docs = [
            {"chunk_id": f"c{i}", "score": 0.95 - i * 0.01}
            for i in range(10)
        ]
        result = rerank._step_3_cliff_cutoff(docs)
        # 没有断崖，取 max_topk=5
        assert len(result) == 5, f"Expected 5, got {len(result)}"

    def test_cliff_ratio(self):
        """测试相对断崖检测"""
        from processor.query_processor.nodes.node_rerank import NodeRerank

        rerank = NodeRerank()
        # 绝对差值 < 0.3，但相对差值 > 0.25: (0.2-0.1)/0.2 = 0.5
        docs = [
            {"chunk_id": "c1", "score": 0.8},
            {"chunk_id": "c2", "score": 0.2},
            {"chunk_id": "c3", "score": 0.1},  # gap_ratio = 0.1/0.2 = 0.5 > 0.25 → 截断
            {"chunk_id": "c4", "score": 0.09},
        ]
        result = rerank._step_3_cliff_cutoff(docs)
        result_ids = [d["chunk_id"] for d in result]
        assert "c3" not in result_ids, f"Expected cliff at c3, got {result_ids}"

    def test_min_topk(self):
        """测试最小 TopK 下限"""
        from processor.query_processor.nodes.node_rerank import NodeRerank

        rerank = NodeRerank()
        # 只有2个文档，应该至少保留2个
        docs = [
            {"chunk_id": "c1", "score": 0.9},
            {"chunk_id": "c2", "score": 0.1},
        ]
        result = rerank._step_3_cliff_cutoff(docs)
        assert len(result) >= 2, f"Expected at least 2, got {len(result)}"

    def test_empty_docs(self):
        """测试空文档列表"""
        from processor.query_processor.nodes.node_rerank import NodeRerank

        rerank = NodeRerank()
        result = rerank._step_3_cliff_cutoff([])
        assert result == []

    def test_single_doc(self):
        """测试单文档"""
        from processor.query_processor.nodes.node_rerank import NodeRerank

        rerank = NodeRerank()
        docs = [{"chunk_id": "c1", "score": 0.95}]
        result = rerank._step_3_cliff_cutoff(docs)
        assert len(result) == 1


class TestMetrics:
    """指标计算单元测试"""

    def test_hit_at_k(self):
        from eval.metrics import hit_at_k
        assert hit_at_k(["d1", "d2", "d3"], ["d1"], 1) == 1.0
        assert hit_at_k(["d2", "d1", "d3"], ["d1"], 1) == 0.0
        assert hit_at_k(["d2", "d1", "d3"], ["d1"], 2) == 1.0
        assert hit_at_k(["d_a", "d_b"], ["d_x"], 2) == 0.0
        assert hit_at_k([], ["d1"], 5) == 0.0

    def test_mrr(self):
        from eval.metrics import mrr
        assert mrr(["d1", "d2", "d3"], ["d1"]) == 1.0  # rank=1
        assert mrr(["d2", "d1", "d3"], ["d1"]) == 0.5  # rank=2
        assert mrr(["d2", "d3", "d1"], ["d1"]) == 0.3333333333333333  # rank=3
        assert mrr(["d_a", "d_b"], ["d_x"]) == 0.0
        assert mrr([], ["d1"]) == 0.0

    def test_recall_at_k(self):
        from eval.metrics import recall_at_k
        # 3个相关，Top-3命中2个 → 2/3
        r = recall_at_k(["d1", "d2", "d4"], ["d1", "d2", "d3"], 3)
        assert abs(r - 2.0 / 3.0) < 1e-6
        # 全部命中
        assert recall_at_k(["d1", "d2", "d3"], ["d1", "d2", "d3"], 3) == 1.0
        # 无命中
        assert recall_at_k(["d_a"], ["d1"], 1) == 0.0

    def test_ndcg(self):
        from eval.metrics import ndcg_at_k
        # 完美排序
        assert ndcg_at_k(["d1", "d2", "d3"], ["d1", "d2", "d3"], 3) == 1.0
        # 部分命中（d1,d3,d2 都是 relevant 中的，只是顺序不同）
        r = ndcg_at_k(["d1", "d3", "d2"], ["d1", "d2", "d3"], 3)
        # 因为3个相关文档都在top3中，DCG=IDCG，所以NDCG=1.0
        assert abs(r - 1.0) < 1e-6, f"Expected 1.0, got {r}"
        # 真正的部分命中测试：d1是相关，d3/d2不相关
        r2 = ndcg_at_k(["d1", "d4", "d5"], ["d1", "d2", "d3"], 3)
        assert 0 < r2 < 1.0, f"Expected between 0 and 1, got {r2}"

    def test_aggregate(self):
        from eval.metrics import aggregate_metrics
        results = [
            {"hit_at_1": 1.0, "mrr": 1.0},
            {"hit_at_1": 0.0, "mrr": 0.5},
        ]
        agg = aggregate_metrics(results)
        assert abs(agg["hit_at_1"] - 0.5) < 1e-6
        assert abs(agg["mrr"] - 0.75) < 1e-6


def run_all_tests():
    """运行所有单元测试"""
    import traceback

    test_classes = [TestRRF, TestCliffCutoff, TestMetrics]
    total = 0
    passed = 0
    failed = 0

    for test_class in test_classes:
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in methods:
            total += 1
            try:
                getattr(instance, method_name)()
                passed += 1
                print(f"  ✓ {test_class.__name__}.{method_name}")
            except Exception as e:
                failed += 1
                print(f"  ✗ {test_class.__name__}.{method_name}: {e}")
                traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'='*50}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
