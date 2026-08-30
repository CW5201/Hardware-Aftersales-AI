#!/usr/bin/env python3
"""
RAG 检索评测主入口

运行方式:
    python eval/run_eval.py
"""

import sys
import os
import time
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.retriever import RetrieverEvaluator, load_test_dataset, run_evaluation
from eval.report import generate_report, generate_charts, generate_summary_csv
from eval.metrics import compute_all_metrics, aggregate_metrics
from tool.logger import logger
from config.milvus_config import milvus_config

import csv


def main():
    print("=" * 60)
    print("RAG 检索评测系统")
    print("=" * 60)

    # 配置
    dataset_path = str(project_root / "eval" / "dataset" / "rag_eval.jsonl")
    output_dir = str(project_root / "eval" / "results")
    top_k_list = [2, 3, 5, 10]
    methods = ["hybrid", "hybrid_rrf", "hybrid_rrf_rerank", "hyde_hybrid_rrf_rerank"]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)

    # 1. 加载测试集
    print("\n[1/5] 加载测试数据集...")
    dataset = load_test_dataset(dataset_path)
    eval_dataset = [d for d in dataset if d.get("type") != "out_of_scope"]
    print(f"  总用例: {len(dataset)}, 有效评测: {len(eval_dataset)}")

    # 2. 检查Milvus连接
    print("\n[2/5] 检查Milvus连接...")
    try:
        evaluator = RetrieverEvaluator()
        client = evaluator._get_client()
        collections = client.list_collections()
        print(f"  Milvus连接成功, 集合: {collections}")
        if milvus_config.chunks_collection not in collections:
            print(f"  ⚠️ 警告: 集合 '{milvus_config.chunks_collection}' 不存在!")
            print(f"  请确保已导入文档到Milvus")
    except Exception as e:
        print(f"  ⚠️ Milvus连接失败: {e}")
        print("  将跳过Milvus相关测试，仅运行单元测试")
        evaluator = None

    # 3. 运行评测
    print("\n[3/5] 执行检索评测...")
    if evaluator is None:
        print("  Milvus不可用，跳过实际检索评测")
        all_results = []
        errors = []
    else:
        all_results, errors = run_evaluation(
            dataset_path=dataset_path,
            output_dir=output_dir,
            top_k_list=top_k_list,
            methods=methods,
        )
        print(f"  完成 {len(all_results)} 条评测结果, {len(errors)} 条错误")

    # 4. 计算指标
    print("\n[4/5] 计算检索指标...")
    if all_results:
        # 由于当前测试集没有chunk级别的ground truth，
        # 我们用file_title级别的近似评估
        # 实际Hit@K需要在有chunk级标注的数据集上计算
        print("  注意: 当前测试集缺少chunk级ground truth，无法计算Hit@K/MRR")
        print("  将输出召回数量统计和性能数据")

        # 保存原始结果 - 收集所有可能的字段
        all_fields = set()
        for r in all_results:
            all_fields.update(r.keys())
        all_fields = sorted(all_fields)

        if all_results:
            with open(f"{output_dir}/retrieval_results.csv", "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_fields)
                writer.writeheader()
                for row in all_results:
                    filtered_row = {k: row.get(k, "") for k in all_fields}
                    writer.writerow(filtered_row)

        if errors:
            with open(f"{output_dir}/errors.csv", "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=errors[0].keys())
                writer.writeheader()
                writer.writerows(errors)

        # 生成汇总
        summary_rows = generate_summary_csv(all_results, f"{output_dir}/summary.csv")
        print(f"  汇总已保存: {output_dir}/summary.csv")

        # 生成图表
        generate_charts(all_results, f"{output_dir}/figures")

        # 生成报告
        report = generate_report(all_results, {}, errors, len(eval_dataset), f"{output_dir}/report.md")
        print(f"  报告已保存: {output_dir}/report.md")
    else:
        print("  无结果数据，跳过指标计算")
        # 创建空的结果文件
        with open(f"{output_dir}/retrieval_results.csv", "w", encoding="utf-8", newline="") as f:
            f.write("id,question,method,top_k,retrieved_count,elapsed_sec\n")
        with open(f"{output_dir}/summary.csv", "w", encoding="utf-8", newline="") as f:
            f.write("method,top_k,count,avg_retrieved,avg_elapsed\n")

    # 5. 总结
    print("\n[5/5] 评测完成")
    print(f"\n输出文件:")
    print(f"  - {output_dir}/retrieval_results.csv")
    print(f"  - {output_dir}/summary.csv")
    print(f"  - {output_dir}/errors.csv")
    print(f"  - {output_dir}/report.md")
    print(f"  - {output_dir}/figures/retrieval_comparison.png")

    print("\n" + "=" * 60)
    print("下一步:")
    print("  1. 确保Milvus已启动: docker-compose up -d")
    print("  2. 确保文档已导入Milvus")
    print("  3. 构建chunk级ground truth标注")
    print("  4. 重新运行评测获取Hit@K/MRR指标")
    print("=" * 60)


if __name__ == "__main__":
    main()
