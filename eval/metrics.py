# RAG Evaluation Metrics

"""
纯函数式指标计算，不依赖任何外部服务。
支持：Hit@K, MRR, Recall@K, NDCG@K
"""

from typing import List, Dict, Any, Optional
import math


def hit_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Hit@K: 在Top-K结果中是否至少有一个相关文档。

    Args:
        retrieved: 检索返回的文档ID列表（按排名顺序）
        relevant: 标准答案中的相关文档ID列表
        k: Top-K截断位置

    Returns:
        1.0 如果相关文档在Top-K中，否则 0.0
    """
    if not retrieved or not relevant:
        return 0.0
    top_k = retrieved[:k]
    return 1.0 if any(r in relevant for r in top_k) else 0.0


def mrr(retrieved: List[str], relevant: List[str]) -> float:
    """
    MRR (Mean Reciprocal Rank): 第一个相关文档的倒数排名。

    Args:
        retrieved: 检索返回的文档ID列表（按排名顺序）
        relevant: 标准答案中的相关文档ID列表

    Returns:
        1/rank if found, else 0.0
    """
    if not retrieved or not relevant:
        return 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Recall@K: Top-K中命中的相关文档占比。

    Args:
        retrieved: 检索返回的文档ID列表
        relevant: 标准答案中的相关文档ID列表
        k: Top-K截断位置

    Returns:
        |retrieved[:k] ∩ relevant| / |relevant|
    """
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    relevant_set = set(relevant)
    hit_count = len(top_k & relevant_set)
    return hit_count / len(relevant_set)


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    NDCG@K (Normalized Discounted Cumulative Gain):
    考虑排序质量的评估指标。

    DCG@K = sum(rel_i / log2(i+1)) for i in 1..K
    where rel_i = 1 if retrieved[i] in relevant else 0

    NDCG@K = DCG@K / IDCG@K
    where IDCG@K is the ideal DCG (all relevant docs at top)
    """
    if not retrieved or not relevant:
        return 0.0

    k = min(k, len(retrieved))
    relevant_set = set(relevant)

    # DCG
    dcg = 0.0
    for i in range(k):
        if retrieved[i] in relevant_set:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1)=0

    # IDCG (ideal: all relevant docs at top positions)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_all_metrics(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k_values: List[int] = None
) -> Dict[str, float]:
    """
    一次性计算所有指标。

    Returns:
        {
            "hit_at_1": float,
            "hit_at_3": float,
            "hit_at_5": float,
            "hit_at_10": float,
            "mrr": float,
            "recall_at_1": float,
            "recall_at_3": float,
            "recall_at_5": float,
            "recall_at_10": float,
            "ndcg_at_1": float,
            "ndcg_at_3": float,
            "ndcg_at_5": float,
            "ndcg_at_10": float,
            "retrieved_count": int,
            "relevant_count": int,
        }
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    metrics = {}
    for k in k_values:
        metrics[f"hit_at_{k}"] = hit_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"recall_at_{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"ndcg_at_{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)
    metrics["mrr"] = mrr(retrieved_ids, relevant_ids)
    metrics["retrieved_count"] = len(retrieved_ids)
    metrics["relevant_count"] = len(relevant_ids)
    return metrics


def aggregate_metrics(results: List[Dict[str, float]]) -> Dict[str, float]:
    """
    对多条测试用例的指标取平均。

    Args:
        results: 每条用例的 compute_all_metrics 结果列表

    Returns:
        各项指标的平均值
    """
    if not results:
        return {}

    keys = results[0].keys()
    aggregated = {}
    for key in keys:
        values = [r.get(key, 0.0) for r in results]
        aggregated[key] = sum(values) / len(values) if values else 0.0
    return aggregated


if __name__ == "__main__":
    # 单元测试
    retrieved = ["doc_1", "doc_2", "doc_3", "doc_5", "doc_10"]
    relevant = ["doc_1", "doc_3", "doc_7"]

    # Hit@3: doc_1在位置1 → 1.0
    assert hit_at_k(retrieved, relevant, 3) == 1.0
    # Hit@1: doc_1在位置1 → 1.0
    assert hit_at_k(retrieved, relevant, 1) == 1.0
    # Hit@5: doc_1在位置1, doc_3在位置3 → 1.0
    assert hit_at_k(retrieved, relevant, 5) == 1.0

    # Recall@3: doc_1, doc_3 命中 / 3个相关 → 2/3
    recall = recall_at_k(retrieved, relevant, 3)
    assert abs(recall - 2.0 / 3.0) < 1e-6

    # MRR: doc_1在位置1 → 1.0
    assert mrr(retrieved, relevant) == 1.0

    # 测试无命中的情况
    assert hit_at_k(["doc_a", "doc_b"], ["doc_x"], 2) == 0.0
    assert mrr(["doc_a", "doc_b"], ["doc_x"]) == 0.0
    assert recall_at_k(["doc_a", "doc_b"], ["doc_x"], 2) == 0.0

    # 测试空列表
    assert hit_at_k([], ["doc_1"], 5) == 0.0
    assert mrr([], ["doc_1"]) == 0.0

    print("All metrics unit tests passed!")
