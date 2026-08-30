"""
RAG 评测报告生成器

从CSV结果生成Markdown报告和matplotlib图表。
"""

import csv
import json
import os
from pathlib import Path
from typing import List, Dict, Any


def load_results(csv_path: str) -> List[Dict]:
    """从CSV加载结果"""
    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def generate_summary(results: List[Dict]) -> Dict[str, Any]:
    """生成汇总统计"""
    from collections import defaultdict

    # 按 method + top_k 分组
    groups = defaultdict(list)
    for r in results:
        key = (r["method"], int(r["top_k"]))
        groups[key].append(r)

    summary = {}
    for (method, top_k), rows in groups.items():
        summary[f"{method}@K{top_k}"] = {
            "count": len(rows),
            "avg_retrieved": sum(float(r["retrieved_count"]) for r in rows) / len(rows),
            "avg_elapsed": sum(float(r["elapsed_sec"]) for r in rows) / len(rows),
            "avg_pre_rerank": sum(float(r.get("pre_rerank", 0) or 0) for r in rows) / len(rows) if "pre_rerank" in rows[0] else "N/A",
            "avg_post_rerank": sum(float(r.get("post_rerank", 0) or 0) for r in rows) / len(rows) if "post_rerank" in rows[0] else "N/A",
            "avg_cliff_cutoff": sum(float(r.get("cliff_cutoff", 0) or 0) for r in rows) / len(rows) if "cliff_cutoff" in rows[0] else "N/A",
        }
    return summary


def generate_report(
    results: List[Dict],
    summary: Dict,
    errors: List[Dict],
    dataset_size: int,
    output_path: str,
):
    """生成Markdown报告"""
    lines = [
        "# RAG 检索评测报告",
        "",
        f"**生成时间**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 测试集规模",
        f"- 总测试用例数: {dataset_size}",
        f"- 用于检索评测: {len(results) // max(len(set(r['method'] for r in results)) if results else 1, 1)} 条（去重后）",
        "",
        "## 2. 测试环境",
        "- Embedding: BGE-M3 (本地CPU)",
        "- 向量库: Milvus",
        "- Rerank: DashScope Qwen3-Rerank API",
        "- RRF: Reciprocal Rank Fusion (k=60)",
        "- Cliff Cutoff: 绝对阈值0.3 / 相对阈值0.25",
        "",
        "## 3. 测试方法",
        "| 方法 | 说明 |",
        "|------|------|",
        "| hybrid | Dense + Sparse 混合检索，无融合 |",
        "| hybrid_rrf | 混合检索 + RRF融合 |",
        "| hybrid_rrf_rerank | 混合检索 + RRF + Cross-Encoder重排 + 断崖截断 |",
        "| hyde_hybrid_rrf_rerank | HyDE假设文档 + 混合检索 + RRF + Rerank + 断崖截断 |",
        "",
        "## 4. Top-K 参数",
        "- 测试值: 2, 3, 5, 10",
        "",
        "## 5. 检索结果汇总",
        "",
    ]

    # 按方法分组展示
    methods = sorted(set(r["method"] for r in results))
    for method in methods:
        method_results = [r for r in results if r["method"] == method]
        lines.append(f"### {method}")
        lines.append("")
        lines.append("| Top-K | 评测数 | 平均召回数 | 平均耗时(s) | 平均预截断数 | 平均重排后数 | 平均断崖截断数 |")
        lines.append("|-------|--------|-----------|------------|-------------|-------------|---------------|")

        for top_k in [2, 3, 5, 10]:
            tk_results = [r for r in method_results if int(r["top_k"]) == top_k]
            if tk_results:
                avg_retrieved = sum(float(r["retrieved_count"]) for r in tk_results) / len(tk_results)
                avg_elapsed = sum(float(r["elapsed_sec"]) for r in tk_results) / len(tk_results)
                pre = tk_results[0].get("pre_rerank", "N/A")
                post = tk_results[0].get("post_rerank", "N/A")
                cutoff = tk_results[0].get("cliff_cutoff", "N/A")
                lines.append(
                    f"| {top_k} | {len(tk_results)} | {avg_retrieved:.1f} | "
                    f"{avg_elapsed:.3f} | {pre} | {post} | {cutoff} |"
                )
        lines.append("")

    # 错误统计
    lines.append("## 6. 错误统计")
    lines.append(f"- 错误总数: {len(errors)}")
    if errors:
        lines.append("")
        lines.append("### 错误案例")
        lines.append("| ID | 方法 | Top-K | 错误信息 |")
        lines.append("|----|------|-------|---------|")
        for e in errors[:10]:
            lines.append(f"| {e['id']} | {e['method']} | {e['top_k']} | {e['error'][:80]} |")
    lines.append("")

    # 结论
    lines.append("## 7. 结论")
    lines.append("")
    lines.append("### 关键发现")
    lines.append("")

    # 分析断崖截断效果
    rerank_results = [r for r in results if r["method"] == "hybrid_rrf_rerank"]
    if rerank_results:
        avg_pre = sum(float(r.get("pre_rerank", 0) or 0) for r in rerank_results) / len(rerank_results)
        avg_cutoff = sum(float(r.get("cliff_cutoff", 0) or 0) for r in rerank_results) / len(rerank_results)
        lines.append(f"- **断崖截断效果**: 预截断平均 {avg_pre:.1f} 条 → 最终保留平均 {avg_cutoff:.1f} 条")
        if avg_pre > 0:
            reduction = (1 - avg_cutoff / avg_pre) * 100
            lines.append(f"- **Token节省**: 约 {reduction:.1f}% 的上下文被截断")
    lines.append("")

    lines.append("---")
    lines.append("*本报告由 eval/run_eval.py 自动生成*")

    report = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report


def generate_charts(results: List[Dict], output_dir: str):
    """生成matplotlib图表"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    os.makedirs(output_dir, exist_ok=True)

    # 1. 各方法不同Top-K的平均召回数对比
    methods = sorted(set(r["method"] for r in results))
    top_ks = [2, 3, 5, 10]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    colors = cm.tab10.colors

    # Chart 1: 平均召回数对比
    ax = axes[0]
    x = range(len(methods))
    width = 0.2
    for i, tk in enumerate(top_ks):
        values = []
        for m in methods:
            subset = [r for r in results if r["method"] == m and int(r["top_k"]) == tk]
            avg = sum(float(r["retrieved_count"]) for r in subset) / len(subset) if subset else 0
            values.append(avg)
        ax.bar([xi + i * width for xi in x], values, width, label=f"Top-K={tk}", color=colors[i])
    ax.set_xlabel("Method")
    ax.set_ylabel("Avg Retrieved Count")
    ax.set_title("Average Retrieved Count by Method & Top-K")
    ax.set_xticks([xi + 1.5 * width for xi in range(len(methods))])
    ax.set_xticklabels([m.replace("_", "\n") for m in methods], fontsize=8)
    ax.legend(fontsize=8)

    # Chart 2: 耗时对比
    ax = axes[1]
    for i, tk in enumerate(top_ks):
        values = []
        for m in methods:
            subset = [r for r in results if r["method"] == m and int(r["top_k"]) == tk]
            avg = sum(float(r["elapsed_sec"]) for r in subset) / len(subset) if subset else 0
            values.append(avg)
        ax.bar([xi + i * width for xi in x], values, width, label=f"Top-K={tk}", color=colors[i])
    ax.set_xlabel("Method")
    ax.set_ylabel("Avg Latency (seconds)")
    ax.set_title("Average Latency by Method & Top-K")
    ax.set_xticks([xi + 1.5 * width for xi in range(len(methods))])
    ax.set_xticklabels([m.replace("_", "\n") for m in methods], fontsize=8)
    ax.legend(fontsize=8)

    # Chart 3: 断崖截断效果（仅hybrid_rrf_rerank）
    ax = axes[3]
    rerank_methods = [m for m in methods if "rerank" in m and "hyde" not in m]
    if rerank_methods:
        m = rerank_methods[0]
        pre_vals, cutoff_vals = [], []
        for tk in top_ks:
            subset = [r for r in results if r["method"] == m and int(r["top_k"]) == tk]
            if subset:
                pre_vals.append(sum(float(r.get("pre_rerank", 0) or 0) for r in subset) / len(subset))
                cutoff_vals.append(sum(float(r.get("cliff_cutoff", 0) or 0) for r in subset) / len(subset))
            else:
                pre_vals.append(0)
                cutoff_vals.append(0)
        x_pos = range(len(top_ks))
        ax.bar(x_pos, pre_vals, label="Pre-cutoff")
        ax.bar(x_pos, cutoff_vals, label="After cliff cutoff", bottom=cutoff_vals)
        # Actually show both as separate bars
        ax2 = ax.twinx()
        ax.bar([xi - 0.2 for xi in x_pos], pre_vals, 0.4, label="Pre-cutoff", color="lightblue")
        ax2.bar([xi + 0.2 for xi in x_pos], cutoff_vals, 0.4, label="After cutoff", color="orange")
        ax.set_xlabel("Top-K")
        ax.set_ylabel("Count")
        ax2.set_ylabel("Cutoff Count")
        ax.set_title(f"Cliff Cutoff Effect ({m})")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(t) for t in top_ks])
    else:
        ax.text(0.5, 0.5, "No rerank data", ha="center")
        ax.set_title("Cliff Cutoff Effect")

    # Chart 4: 各方法Top-K对比（召回数）
    ax = axes[3]
    for i, m in enumerate(methods):
        values = []
        for tk in top_ks:
            subset = [r for r in results if r["method"] == m and int(r["top_k"]) == tk]
            avg = sum(float(r["retrieved_count"]) for r in subset) / len(subset) if subset else 0
            values.append(avg)
        ax.plot(top_ks, values, marker="o", label=m.replace("_", " "), linewidth=2)
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Avg Retrieved Count")
    ax.set_title("Retrieved Count vs Top-K")
    ax.legend(fontsize=8)
    ax.set_xticks(top_ks)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/retrieval_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"图表已保存: {output_dir}/retrieval_comparison.png")


def generate_summary_csv(results: List[Dict], output_path: str):
    """生成汇总CSV"""
    summary = generate_summary(results)

    rows = []
    for key, stats in summary.items():
        method, top_k = key.split("@")
        rows.append({
            "method": method,
            "top_k": top_k,
            "test_count": stats["count"],
            "avg_retrieved_count": round(stats["avg_retrieved"], 2),
            "avg_elapsed_sec": round(stats["avg_elapsed"], 3),
            "avg_pre_rerank": round(stats["avg_pre_rerank"], 2) if stats["avg_pre_rerank"] != "N/A" else "N/A",
            "avg_post_rerank": round(stats["avg_post_rerank"], 2) if stats["avg_post_rerank"] != "N/A" else "N/A",
            "avg_cliff_cutoff": round(stats["avg_cliff_cutoff"], 2) if stats["avg_cliff_cutoff"] != "N/A" else "N/A",
        })

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"汇总CSV已保存: {output_path}")
    return rows


if __name__ == "__main__":
    import sys
    results = load_results("eval/results/retrieval_results.csv")
    if results:
        summary = generate_summary(results)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        generate_charts(results, "eval/results/figures")
        generate_summary_csv(results, "eval/results/summary.csv")
        generate_report(results, summary, [], len(results), "eval/results/report.md")
        print("报告已生成: eval/results/report.md")
