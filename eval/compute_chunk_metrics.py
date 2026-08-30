#!/usr/bin/env python3
"""
Chunk-level Ground Truth RAG 检索质量评测

用法:
    python eval/compute_chunk_metrics.py
"""

import sys
import os
import json
import csv
import math
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

DATASET_PATH = project_root / "eval" / "dataset" / "rag_eval_chunk_level.jsonl"
RESULTS_CSV = project_root / "eval" / "results" / "retrieval_results.csv"
OUTPUT_DIR = project_root / "eval" / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"

# ============================================================
# 1. 加载数据
# ============================================================

def load_ground_truth(path: str) -> List[Dict]:
    """加载 chunk-level ground truth，每条记录:
       {id, question, gold_chunks: [{chunk_id, ...}], needs_manual_review}
    """
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            dataset.append(json.loads(line))
    return dataset


def load_retrieval_results(path: str) -> List[Dict]:
    """加载已运行的检索结果 CSV，支持两种格式：
       格式A（run_eval.py 输出）: id, method, top_k, question, retrieved_ids (JSON), elapsed_sec, ...
       格式B（compute_chunk_metrics.py 上轮输出）: question_id, method, top_k, question,
               gold_chunks, retrieved_chunks (pipe-separated), hit_at_*, rr, recall_at_*, latency_ms
    """
    results = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        is_format_b = "question_id" in headers
        for row in reader:
            # 统一字段名
            if is_format_b:
                row["id"] = row.get("question_id", "")
                # retrieved_chunks 是 pipe-separated 字符串
                rc = row.get("retrieved_chunks", "")
                row["retrieved_ids"] = [x for x in rc.split("|") if x] if rc else []
                # latency_ms → elapsed_sec
                try:
                    row["elapsed_sec"] = float(row.get("latency_ms", 0)) / 1000.0
                except Exception:
                    row["elapsed_sec"] = 0.0
            else:
                row["id"] = row.get("id", "")
                try:
                    row["retrieved_ids"] = json.loads(row.get("retrieved_ids", "[]"))
                except Exception:
                    row["retrieved_ids"] = []
                try:
                    row["elapsed_sec"] = float(row.get("elapsed_sec", 0))
                except Exception:
                    row["elapsed_sec"] = 0.0
            try:
                row["top_k"] = int(row["top_k"])
            except Exception:
                row["top_k"] = 0
            results.append(row)
    return results


# ============================================================
# 2. 指标计算（纯函数）
# ============================================================

def hit_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not retrieved or not relevant:
        return 0.0
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0


def mrr(retrieved: List[str], relevant: List[str]) -> float:
    if not retrieved or not relevant:
        return 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    relevant_set = set(relevant)
    hit_count = len(top_k & relevant_set)
    return hit_count / len(relevant_set)


def compute_row_metrics(retrieved_ids: List[str], gold_ids: List[str], k_values: List[int] = None) -> Dict[str, float]:
    if k_values is None:
        k_values = [1, 2, 3, 5, 10]
    m = {"retrieved_count": len(retrieved_ids), "relevant_count": len(gold_ids)}
    for k in k_values:
        m[f"hit_at_{k}"] = hit_at_k(retrieved_ids, gold_ids, k)
        m[f"recall_at_{k}"] = recall_at_k(retrieved_ids, gold_ids, k)
    m["mrr"] = mrr(retrieved_ids, gold_ids)
    return m


# ============================================================
# 3. 匹配 + 逐题计算
# ============================================================

def match_results(gt_dataset: List[Dict], ret_results: List[Dict]) -> List[Dict]:
    """将 ground truth 与检索结果匹配，返回逐题+方法的详细记录。"""
    # 构建 GT 查找表
    gt_map = {d["id"]: d for d in gt_dataset}
    # 按 (id, method, top_k) 分组检索结果
    ret_groups = defaultdict(list)
    for r in ret_results:
        key = (r["id"], r["method"], r["top_k"])
        ret_groups[key].append(r)

    matched = []
    for g in gt_dataset:
        gid = g["id"]
        question = g["question"]
        gold_chunks = g.get("gold_chunks", [])
        gold_ids = [str(c["chunk_id"]) for c in gold_chunks if c.get("chunk_id")]
        if not gold_ids:
            continue  # 跳过无 gold 的题目

        for method in ["hybrid", "hybrid_rrf", "hybrid_rrf_rerank", "hyde_hybrid_rrf_rerank"]:
            for top_k in [2, 3, 5, 10]:
                key = (gid, method, top_k)
                rows = ret_groups.get(key, [])
                if not rows:
                    continue

                # 取最新的一条（多次运行可能有多条，这里取 elapsed_sec 最大的作为最终结果）
                row = max(rows, key=lambda x: x.get("elapsed_sec", 0))
                retrieved_ids = row["retrieved_ids"]

                # 去重并保持顺序
                seen = set()
                unique_retrieved = []
                for rid in retrieved_ids:
                    rid_str = str(rid)
                    if rid_str not in seen:
                        seen.add(rid_str)
                        unique_retrieved.append(rid_str)

                metrics = compute_row_metrics(unique_retrieved, gold_ids)

                matched.append({
                    "question_id": gid,
                    "method": method,
                    "top_k": top_k,
                    "question": question,
                    "gold_chunks": "|".join(gold_ids),
                    "retrieved_chunks": "|".join(unique_retrieved[:10]),
                    "hit_at_1": metrics["hit_at_1"],
                    "hit_at_3": metrics["hit_at_3"],
                    "hit_at_5": metrics["hit_at_5"],
                    "hit_at_10": metrics["hit_at_10"],
                    "rr": metrics["mrr"],
                    "recall_at_1": metrics["recall_at_1"],
                    "recall_at_3": metrics["recall_at_3"],
                    "recall_at_5": metrics["recall_at_5"],
                    "recall_at_10": metrics["recall_at_10"],
                    "latency_ms": round(row["elapsed_sec"] * 1000, 1),
                    "retrieved_count": metrics["retrieved_count"],
                    "relevant_count": metrics["relevant_count"],
                    "needs_manual_review": g.get("needs_manual_review", False),
                })

    return matched


# ============================================================
# 4. 生成 per-question CSV
# ============================================================

def write_detailed_csv(matched: List[Dict], path: str):
    fields = [
        "question_id", "method", "top_k", "question",
        "gold_chunks", "retrieved_chunks",
        "hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10",
        "rr", "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
        "latency_ms",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in matched:
            out = {k: row[k] for k in fields}
            writer.writerow(out)


# ============================================================
# 5. 生成 summary.csv
# ============================================================

def generate_summary(matched: List[Dict], path: str) -> List[Dict]:
    groups = defaultdict(list)
    for m in matched:
        groups[(m["method"], m["top_k"])].append(m)

    summary_rows = []
    for (method, top_k), items in sorted(groups.items()):
        n = len(items)
        s = {
            "method": method,
            "top_k": top_k,
            "num_questions": n,
            "hit_at_1": round(sum(i["hit_at_1"] for i in items) / n, 4),
            "hit_at_3": round(sum(i["hit_at_3"] for i in items) / n, 4),
            "hit_at_5": round(sum(i["hit_at_5"] for i in items) / n, 4),
            "hit_at_10": round(sum(i["hit_at_10"] for i in items) / n, 4),
            "mrr": round(sum(i["rr"] for i in items) / n, 4),
            "recall_at_1": round(sum(i["recall_at_1"] for i in items) / n, 4),
            "recall_at_3": round(sum(i["recall_at_3"] for i in items) / n, 4),
            "recall_at_5": round(sum(i["recall_at_5"] for i in items) / n, 4),
            "recall_at_10": round(sum(i["recall_at_10"] for i in items) / n, 4),
            "avg_latency_ms": round(sum(i["latency_ms"] for i in items) / n, 1),
        }
        # P50 / P95
        latencies = [i["latency_ms"] for i in items]
        s["p50_latency_ms"] = round(float(np.percentile(latencies, 50)), 1)
        s["p95_latency_ms"] = round(float(np.percentile(latencies, 95)), 1)
        s["avg_retrieved_chunks"] = round(sum(i["retrieved_count"] for i in items) / n, 2)
        summary_rows.append(s)

    fields = list(summary_rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


# ============================================================
# 6. 生成图表
# ============================================================

def generate_charts(summary: List[Dict], figures_dir: str):
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    os.makedirs(figures_dir, exist_ok=True)

    methods = ["hybrid", "hybrid_rrf", "hybrid_rrf_rerank", "hyde_hybrid_rrf_rerank"]
    method_labels = ["A. hybrid", "B. hybrid+RRF", "C. hybrid+RRF+Rerank", "D. HyDE+RRF+Rerank"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    top_ks = [2, 3, 5, 10]

    # Helper: get metric value for a method@top_k
    def get_val(method, top_k, metric):
        for s in summary:
            if s["method"] == method and s["top_k"] == top_k:
                return s.get(metric, 0.0)
        return 0.0

    # ---- Chart 1: Hit@K comparison (K=5) ----
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(methods))
    width = 0.2
    for i, tk in enumerate(top_ks):
        vals = [get_val(m, tk, f"hit_at_{tk}") for m in methods]
        ax.bar([xi + i * width for xi in x], vals, width, label=f"Hit@{tk}", color=colors[i % len(colors)], alpha=0.85)
    ax.set_xlabel("Retrieval Method")
    ax.set_ylabel("Hit Rate")
    ax.set_title("Hit@K Comparison Across Methods")
    ax.set_xticks([xi + 1.5 * width for xi in x])
    ax.set_xticklabels(method_labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    for i, v in enumerate([get_val(m, 5, "hit_at_5") for m in methods]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/hit_at_5_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Chart 2: MRR comparison ----
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(methods))
    mrr_vals = [get_val(m, 5, "mrr") for m in methods]
    bars = ax.bar(x, mrr_vals, color=colors, alpha=0.85)
    ax.set_xlabel("Retrieval Method")
    ax.set_ylabel("MRR")
    ax.set_title("MRR Comparison (Top-K=5)")
    ax.set_xticklabels(method_labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    for bar, v in zip(bars, mrr_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.4f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/mrr_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Chart 3: Recall@K comparison (K=5) ----
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(methods))
    width = 0.2
    for i, tk in enumerate(top_ks):
        vals = [get_val(m, tk, f"recall_at_{tk}") for m in methods]
        ax.bar([xi + i * width for xi in x], vals, width, label=f"Recall@{tk}", color=colors[i % len(colors)], alpha=0.85)
    ax.set_xlabel("Retrieval Method")
    ax.set_ylabel("Recall Rate")
    ax.set_title("Recall@K Comparison Across Methods")
    ax.set_xticks([xi + 1.5 * width for xi in x])
    ax.set_xticklabels(method_labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    for i, v in enumerate([get_val(m, 5, "recall_at_5") for m in methods]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/recall_at_5_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Chart 4: Top-K trend ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(methods):
        vals = [get_val(m, tk, f"hit_at_{tk}") for tk in top_ks]
        ax.plot(top_ks, vals, marker="o", label=method_labels[i], color=colors[i], linewidth=2, markersize=8)
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Hit Rate")
    ax.set_title("Hit@K vs Top-K for Each Method")
    ax.set_xticks(top_ks)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/top_k_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Chart 5: Latency comparison ----
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(methods))
    width = 0.2
    latency_keys = ["avg_latency_ms"]
    for i, tk in enumerate(top_ks):
        vals = [get_val(m, tk, "avg_latency_ms") for m in methods]
        ax.bar([xi + i * width for xi in x], vals, width, label=f"Top-K={tk}", color=colors[i % len(colors)], alpha=0.85)
    ax.set_xlabel("Retrieval Method")
    ax.set_ylabel("Avg Latency (ms)")
    ax.set_title("Latency Comparison Across Methods & Top-K")
    ax.set_xticks([xi + 1.5 * width for xi in x])
    ax.set_xticklabels(method_labels, fontsize=9)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/latency_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"图表已保存到: {figures_dir}/")


# ============================================================
# 7. 生成 report.md
# ============================================================

def generate_report(
    matched: List[Dict],
    summary: List[Dict],
    gt_dataset: List[Dict],
    output_path: str,
):
    n_total = len(gt_dataset)
    multi_chunk = sum(1 for d in gt_dataset if len(d.get("gold_chunks", [])) >= 2)
    single_chunk = sum(1 for d in gt_dataset if len(d.get("gold_chunks", [])) == 1)
    manual_review = sum(1 for d in gt_dataset if d.get("needs_manual_review"))

    methods = ["hybrid", "hybrid_rrf", "hybrid_rrf_rerank", "hyde_hybrid_rrf_rerank"]
    method_labels = ["A. hybrid", "B. hybrid + RRF", "C. hybrid + RRF + Rerank", "D. HyDE + hybrid + RRF + Rerank"]

    def get_val(method, top_k, metric):
        for s in summary:
            if s["method"] == method and s["top_k"] == top_k:
                return s.get(metric, 0.0)
        return 0.0

    lines = [
        "# RAG 检索质量评测报告（Chunk-Level Ground Truth）",
        "",
        f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 测试集概况",
        f"- 总问题数: {n_total}",
        f"- 有效评测问题（gold_chunks 非空）: {n_total}",
        f"- 多条 gold chunk 问题 (>=2): {multi_chunk}",
        f"- 单条高置信度 gold chunk 问题 (=1): {single_chunk}",
        f"- 人工复核标记 (needs_manual_review=true): {manual_review}",
        "",
        "> **说明**：15 条 `needs_manual_review=true` 题目并非无法评测，而是仅存在 1 个高置信度 gold chunk，",
        "> 因此仍纳入 Hit@K、MRR、Recall@K 计算。Gold chunk 非空即视为有效评测样本。",
        "",
        "## 2. 检索方案",
        "| 编号 | 方案 | 说明 |",
        "|------|------|------|",
        "| A | hybrid | Dense + Sparse 混合检索，无融合/重排 |",
        "| B | hybrid_rrf | 混合检索 + RRF 融合 |",
        "| C | hybrid_rrf_rerank | 混合检索 + RRF + Cross-Encoder 重排 + 断崖截断 |",
        "| D | hyde_hybrid_rrf_rerank | HyDE + 混合检索 + RRF + Rerank + 断崖截断 |",
        "",
        "## 3. Top-K 消融实验（Hit@K）",
        "",
        "| 方案 | K=2 | K=3 | K=5 | K=10 |",
        "|------|-----|-----|-----|------|",
    ]
    for ml, m in zip(method_labels, methods):
        vals = [get_val(m, k, f"hit_at_{k}") for k in [2, 3, 5, 10]]
        lines.append(f"| {ml} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {vals[3]:.4f} |")

    lines += [
        "",
        "## 4. Top-K 消融实验（MRR）",
        "",
        "| 方案 | K=2 | K=3 | K=5 | K=10 |",
        "|------|-----|-----|-----|------|",
    ]
    for ml, m in zip(method_labels, methods):
        vals = [get_val(m, k, "mrr") for k in [2, 3, 5, 10]]
        lines.append(f"| {ml} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {vals[3]:.4f} |")

    lines += [
        "",
        "## 5. Top-K 消融实验（Recall@K）",
        "",
        "| 方案 | K=2 | K=3 | K=5 | K=10 |",
        "|------|-----|-----|-----|------|",
    ]
    for ml, m in zip(method_labels, methods):
        vals = [get_val(m, k, f"recall_at_{k}") for k in [2, 3, 5, 10]]
        lines.append(f"| {ml} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {vals[3]:.4f} |")

    lines += [
        "",
        "## 6. 延迟统计（ms）",
        "",
        "| 方案 | K=2 | K=3 | K=5 | K=10 | P50(K=5) | P95(K=5) |",
        "|------|-----|-----|-----|------|---------|---------|",
    ]
    for ml, m in zip(method_labels, methods):
        vals = [get_val(m, k, "avg_latency_ms") for k in [2, 3, 5, 10]]
        p50 = get_val(m, 5, "p50_latency_ms")
        p95 = get_val(m, 5, "p95_latency_ms")
        lines.append(f"| {ml} | {vals[0]:.0f} | {vals[1]:.0f} | {vals[2]:.0f} | {vals[3]:.0f} | {p50:.0f} | {p95:.0f} |")

    lines += [
        "",
        "## 7. 消融实验分析",
        "",
    ]

    # A vs B: RRF effect
    a_hit5 = get_val("hybrid", 5, "hit_at_5")
    b_hit5 = get_val("hybrid_rrf", 5, "hit_at_5")
    a_mrr = get_val("hybrid", 5, "mrr")
    b_mrr = get_val("hybrid_rrf", 5, "mrr")
    a_recall5 = get_val("hybrid", 5, "recall_at_5")
    b_recall5 = get_val("hybrid_rrf", 5, "recall_at_5")
    lines.append("### RRF 效果（A→B）")
    lines.append(f"- Hit@5: {a_hit5:.4f} → {b_hit5:.4f} ({'↑' if b_hit5 > a_hit5 else '↓'} {abs(b_hit5 - a_hit5):.4f})")
    lines.append(f"- MRR: {a_mrr:.4f} → {b_mrr:.4f} ({'↑' if b_mrr > a_mrr else '↓'} {abs(b_mrr - a_mrr):.4f})")
    lines.append(f"- Recall@5: {a_recall5:.4f} → {b_recall5:.4f} ({'↑' if b_recall5 > a_recall5 else '↓'} {abs(b_recall5 - a_recall5):.4f})")

    # B vs C: Rerank effect
    c_hit5 = get_val("hybrid_rrf_rerank", 5, "hit_at_5")
    c_mrr = get_val("hybrid_rrf_rerank", 5, "mrr")
    c_recall5 = get_val("hybrid_rrf_rerank", 5, "recall_at_5")
    lines.append("")
    lines.append("### Rerank 效果（B→C）")
    lines.append(f"- Hit@5: {b_hit5:.4f} → {c_hit5:.4f} ({'↑' if c_hit5 > b_hit5 else '↓'} {abs(c_hit5 - b_hit5):.4f})")
    lines.append(f"- MRR: {b_mrr:.4f} → {c_mrr:.4f} ({'↑' if c_mrr > b_mrr else '↓'} {abs(c_mrr - b_mrr):.4f})")
    lines.append(f"- Recall@5: {b_recall5:.4f} → {c_recall5:.4f} ({'↑' if c_recall5 > b_recall5 else '↓'} {abs(c_recall5 - b_recall5):.4f})")

    # C vs D: HyDE effect
    d_hit5 = get_val("hyde_hybrid_rrf_rerank", 5, "hit_at_5")
    d_mrr = get_val("hyde_hybrid_rrf_rerank", 5, "mrr")
    d_recall5 = get_val("hyde_hybrid_rrf_rerank", 5, "recall_at_5")
    c_lat = get_val("hybrid_rrf_rerank", 5, "avg_latency_ms")
    d_lat = get_val("hyde_hybrid_rrf_rerank", 5, "avg_latency_ms")
    lines.append("")
    lines.append("### HyDE 效果（C→D）")
    lines.append(f"- Hit@5: {c_hit5:.4f} → {d_hit5:.4f} ({'↑' if d_hit5 > c_hit5 else '↓'} {abs(d_hit5 - c_hit5):.4f})")
    lines.append(f"- MRR: {c_mrr:.4f} → {d_mrr:.4f} ({'↑' if d_mrr > c_mrr else '↓'} {abs(d_mrr - c_mrr):.4f})")
    lines.append(f"- Recall@5: {c_recall5:.4f} → {d_recall5:.4f} ({'↑' if d_recall5 > c_recall5 else '↓'} {abs(d_recall5 - c_recall5):.4f})")
    lines.append(f"- 延迟增加: {c_lat:.0f}ms → {d_lat:.0f}ms (+{d_lat - c_lat:.0f}ms, +{(d_lat/c_lat - 1)*100:.1f}%)")

    lines += [
        "",
        "## 8. 最佳方案汇总（K=5）",
        "",
        "| 指标 | 最佳方案 | 数值 |",
        "|------|---------|------|",
    ]

    best_hit = max(methods, key=lambda m: get_val(m, 5, "hit_at_5"))
    best_mrr = max(methods, key=lambda m: get_val(m, 5, "mrr"))
    best_recall = max(methods, key=lambda m: get_val(m, 5, "recall_at_5"))
    best_lat = min(methods, key=lambda m: get_val(m, 5, "avg_latency_ms"))
    best_hit_v = get_val(best_hit, 5, "hit_at_5")
    best_mrr_v = get_val(best_mrr, 5, "mrr")
    best_recall_v = get_val(best_recall, 5, "recall_at_5")
    best_lat_v = get_val(best_lat, 5, "avg_latency_ms")

    labels_map = dict(zip(methods, method_labels))
    lines.append(f"| Hit@5 | {labels_map[best_hit]} | {best_hit_v:.4f} |")
    lines.append(f"| MRR | {labels_map[best_mrr]} | {best_mrr_v:.4f} |")
    lines.append(f"| Recall@5 | {labels_map[best_recall]} | {best_recall_v:.4f} |")
    lines.append(f"| 延迟(K=5) | {labels_map[best_lat]} | {best_lat_v:.0f}ms |")

    lines += [
        "",
        "## 9. 错误案例（Top-K=5，Hit@5=0）",
        "",
    ]
    miss_rows = [m for m in matched if m["top_k"] == 5 and m["hit_at_5"] == 0]
    if miss_rows:
        lines.append("| 问题ID | 问题 | 方案 | Gold Chunks | Retrieved |")
        lines.append("|--------|------|------|-------------|-----------|")
        for r in miss_rows[:15]:
            q = r["question"][:50]
            lines.append(f"| {r['question_id']} | {q} | {r['method']} | {r['gold_chunks'][:80]} | {r['retrieved_chunks'][:80]} |")
    else:
        lines.append("无（所有问题在 Top-5 均有命中）")

    lines += [
        "",
        "## 10. 测试限制",
        "",
        "- 测试集规模较小（34 题），统计波动较大",
        "- Ground Truth 由人工标注/关键词匹配生成，可能存在漏标",
        "- 15 条 needs_manual_review=true 题目仅有 1 个 gold chunk，无法评估多 chunk 召回",
        "- 评测基于已运行的检索结果 CSV，非实时运行",
        "- RRF 在 hybrid_rrf 中使用单路检索结果做双路融合（模拟），非真实双路",
        "",
        "---",
        "*本报告由 eval/compute_chunk_metrics.py 自动生成*",
    ]

    report = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report


# ============================================================
# 8. Spot Check（抽查 10 题）
# ============================================================

def spot_check(matched: List[Dict], gt_dataset: List[Dict]):
    """逐题打印抽查结果，用于验证 chunk ID 匹配逻辑。"""
    print("\n" + "=" * 80)
    print("SPOT CHECK: 随机抽查 10 题 (method=hybrid_rrf_rerank, K=5)")
    print("=" * 80)

    gt_map = {d["id"]: d for d in gt_dataset}
    check_rows = [m for m in matched if m["method"] == "hybrid_rrf_rerank" and m["top_k"] == 5]

    # 取前 10 题
    sample_ids = [m["question_id"] for m in check_rows[:10]]

    for qid in sample_ids:
        g = gt_map[qid]
        gold_ids = [str(c["chunk_id"]) for c in g["gold_chunks"]]
        ret = next(m for m in check_rows if m["question_id"] == qid)

        retrieved = ret["retrieved_chunks"].split("|") if ret["retrieved_chunks"] else []

        hit1 = 1 if any(r in gold_ids for r in retrieved[:1]) else 0
        hit3 = 1 if any(r in gold_ids for r in retrieved[:3]) else 0
        hit5 = 1 if any(r in gold_ids for r in retrieved[:5]) else 0
        hit10 = 1 if any(r in gold_ids for r in retrieved[:10]) else 0

        # MRR
        rr = 0.0
        for rank, r in enumerate(retrieved, 1):
            if r in gold_ids:
                rr = 1.0 / rank
                break

        print(f"\n[{qid}] {g['question'][:60]}")
        print(f"  Gold chunks ({len(gold_ids)}): {gold_ids}")
        print(f"  Retrieved top10: {retrieved[:10]}")
        print(f"  Hit@1={hit1} Hit@3={hit3} Hit@5={hit5} Hit@10={hit10}  RR={rr:.4f}")

        # 检查匹配问题
        hit_ids = set(retrieved[:10]) & set(gold_ids)
        if not hit_ids:
            print(f"  ⚠️  完全未命中！")
            # 打印类型信息
            for gc in g["gold_chunks"]:
                cid = gc.get("chunk_id")
                print(f"     Gold chunk_id type: {type(cid).__name__}, value: {repr(cid)}")
            for rid in retrieved[:3]:
                print(f"     Retrieved id type: {type(rid).__name__}, value: {repr(rid)}")

    print("\n" + "=" * 80)


# ============================================================
# 9. 主入口
# ============================================================

def main():
    print("=" * 60)
    print("RAG Chunk-Level Ground Truth 检索质量评测")
    print("=" * 60)

    # 1. 加载
    print("\n[1] 加载 ground truth ...")
    gt_dataset = load_ground_truth(str(DATASET_PATH))
    print(f"  Ground truth: {len(gt_dataset)} 条")
    multi = sum(1 for d in gt_dataset if len(d.get("gold_chunks", [])) >= 2)
    single = sum(1 for d in gt_dataset if len(d.get("gold_chunks", [])) == 1)
    manual = sum(1 for d in gt_dataset if d.get("needs_manual_review"))
    print(f"  多 chunk (>=2): {multi}, 单 chunk (=1): {single}, 需人工复核: {manual}")

    print("\n[2] 加载检索结果 ...")
    ret_results = load_retrieval_results(str(RESULTS_CSV))
    print(f"  检索结果: {len(ret_results)} 条")

    # 2. 匹配 + 计算指标
    print("\n[3] 匹配 GT 与检索结果，计算指标 ...")
    matched = match_results(gt_dataset, ret_results)
    print(f"  匹配结果: {len(matched)} 条")

    # 3. Spot check
    print("\n[4] Spot check（抽查 10 题）...")
    spot_check(matched, gt_dataset)

    # 4. 写 detailed CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    detailed_path = OUTPUT_DIR / "detailed_results.csv"
    print(f"\n[5] 写逐题结果: {detailed_path}")
    write_detailed_csv(matched, str(detailed_path))

    # 5. 写 summary CSV
    summary_path = OUTPUT_DIR / "summary.csv"
    print(f"[6] 写 summary: {summary_path}")
    summary = generate_summary(matched, str(summary_path))
    for s in summary:
        print(f"  {s['method']}@K{s['top_k']}: Hit@5={s['hit_at_5']:.4f} MRR={s['mrr']:.4f} Recall@5={s['recall_at_5']:.4f} Latency={s['avg_latency_ms']:.0f}ms")

    # 6. 图表
    print(f"\n[7] 生成图表 ...")
    generate_charts(summary, str(FIGURES_DIR))

    # 7. 报告
    report_path = OUTPUT_DIR / "report.md"
    print(f"[8] 生成报告: {report_path}")
    generate_report(matched, summary, gt_dataset, str(report_path))

    # 8. 最终结论
    print("\n" + "=" * 60)
    print("最终结论")
    print("=" * 60)

    methods = ["hybrid", "hybrid_rrf", "hybrid_rrf_rerank", "hyde_hybrid_rrf_rerank"]
    method_labels = ["A. hybrid", "B. hybrid+RRF", "C. hybrid+RRF+Rerank", "D. HyDE+RRF+Rerank"]

    def gv(m, k, met):
        for s in summary:
            if s["method"] == m and s["top_k"] == k:
                return s.get(met, 0.0)
        return 0.0

    print(f"\n{'方案':<28} {'Hit@5':>8} {'MRR':>8} {'Recall@5':>8} {'Lat(ms)':>10}")
    print("-" * 65)
    for m, ml in zip(methods, method_labels):
        print(f"{ml:<28} {gv(m,5,'hit_at_5'):>8.4f} {gv(m,5,'mrr'):>8.4f} {gv(m,5,'recall_at_5'):>8.4f} {gv(m,5,'avg_latency_ms'):>10.0f}")

    best_hit = max(methods, key=lambda m: gv(m, 5, "hit_at_5"))
    best_mrr = max(methods, key=lambda m: gv(m, 5, "mrr"))
    best_recall = max(methods, key=lambda m: gv(m, 5, "recall_at_5"))

    print(f"\n★ 最佳方案（K=5）:")
    print(f"  Hit@5:   {method_labels[methods.index(best_hit)]}  ({gv(best_hit,5,'hit_at_5'):.4f})")
    print(f"  MRR:     {method_labels[methods.index(best_mrr)]}  ({gv(best_mrr,5,'mrr'):.4f})")
    print(f"  Recall@5:{method_labels[methods.index(best_recall)]}  ({gv(best_recall,5,'recall_at_5'):.4f})")

    # 消融结论
    a_h, b_h, c_h, d_h = gv("hybrid",5,"hit_at_5"), gv("hybrid_rrf",5,"hit_at_5"), gv("hybrid_rrf_rerank",5,"hit_at_5"), gv("hyde_hybrid_rrf_rerank",5,"hit_at_5")
    a_r, b_r, c_r, d_r = gv("hybrid",5,"mrr"), gv("hybrid_rrf",5,"mrr"), gv("hybrid_rrf_rerank",5,"mrr"), gv("hyde_hybrid_rrf_rerank",5,"mrr")
    a_rec, b_rec, c_rec, d_rec = gv("hybrid",5,"recall_at_5"), gv("hybrid_rrf",5,"recall_at_5"), gv("hybrid_rrf_rerank",5,"recall_at_5"), gv("hyde_hybrid_rrf_rerank",5,"recall_at_5")
    a_lat, b_lat, c_lat, d_lat = gv("hybrid",5,"avg_latency_ms"), gv("hybrid_rrf",5,"avg_latency_ms"), gv("hybrid_rrf_rerank",5,"avg_latency_ms"), gv("hyde_hybrid_rrf_rerank",5,"avg_latency_ms")

    print(f"\n★ 消融结论:")
    print(f"  RRF 效果 (A→B): Hit@5 {'↑' if b_h>a_h else '↓'} {abs(b_h-a_h):.4f}, MRR {'↑' if b_r>a_r else '↓'} {abs(b_r-a_r):.4f}")
    print(f"  Rerank 效果(B→C): Hit@5 {'↑' if c_h>b_h else '↓'} {abs(c_h-b_h):.4f}, MRR {'↑' if c_r>b_r else '↓'} {abs(c_r-b_r):.4f}")
    print(f"  HyDE 效果 (C→D): Hit@5 {'↑' if d_h>c_h else '↓'} {abs(d_h-c_h):.4f}, MRR {'↑' if d_r>c_r else '↓'} {abs(d_r-c_r):.4f}, 延迟 +{d_lat-c_lat:.0f}ms")

    # 最佳 Top-K
    print(f"\n★ 最佳 Top-K（hybrid_rrf_rerank）:")
    for k in [2, 3, 5, 10]:
        print(f"  K={k}: Hit@{k}={gv('hybrid_rrf_rerank',k,f'hit_at_{k}'):.4f} MRR={gv('hybrid_rrf_rerank',k,'mrr'):.4f} Lat={gv('hybrid_rrf_rerank',k,'avg_latency_ms'):.0f}ms")

    print(f"\n★ 输出文件:")
    print(f"  - {detailed_path}")
    print(f"  - {summary_path}")
    print(f"  - {report_path}")
    for f in FIGURES_DIR.glob("*.png"):
        print(f"  - {FIGURES_DIR}/{f.name}")

    print("\n" + "=" * 60)
    print("完成。")
    print("=" * 60)


if __name__ == "__main__":
    main()
