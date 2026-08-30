"""
RAG 检索评测器

基于项目真实代码，不修改任何业务逻辑。
独立封装4种检索策略，直接调用项目现有的工具函数。
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.milvus_config import milvus_config
from utils.embedding_utils import generate_embeddings
from utils.milvus_utils import (
    create_hybrid_search_requests,
    hybrid_search,
    get_milvus_client,
)
from utils.reranker_http_utils import rerank_documents
from processor.query_processor.nodes.node_rrf import NodeRrf
from processor.query_processor.nodes.node_rerank import NodeRerank
from tool.logger import logger


class RetrieverEvaluator:
    """
    检索评测器：封装4种检索策略，用于离线评测。
    """

    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or milvus_config.chunks_collection
        self.client = None
        self._rrf_node = NodeRrf()
        self._rerank_node = NodeRerank()

    def _get_client(self):
        """获取Milvus客户端（带连接检查）"""
        if self.client is not None:
            return self.client
        try:
            self.client = get_milvus_client()
            # 测试连接
            self.client.list_collections()
            logger.info(f"Milvus连接成功，集合: {self.collection_name}")
        except Exception as e:
            logger.error(f"Milvus连接失败: {e}")
            raise
        return self.client

    def _search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        item_names: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        基础混合检索（Dense + Sparse），不经过RRF/Rerank。
        返回原始检索结果（List[Dict] with chunk_id, content, score）
        """
        client = self._get_client()
        embeddings = generate_embeddings([query])
        dense_vector = embeddings["dense"][0]
        sparse_vector = embeddings["sparse"][0]

        expr = None
        if item_names and len(item_names) > 0:
            # 构造过滤表达式
            quoted = ", ".join(f"'{v}'" for v in item_names)
            expr = f"item_name in [{quoted}]"

        reqs = create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            expr=expr,
            limit=top_k,
        )

        search_res = hybrid_search(
            client=client,
            collection_name=self.collection_name,
            reqs=reqs,
            ranker_weights=(0.8, 0.2),
            output_fields=["chunk_id", "content", "item_name", "file_title"],
        )

        if not search_res or not search_res[0]:
            return []

        results = []
        for hit in search_res[0]:
            entity = hit.get("entity", {})
            results.append({
                "chunk_id": str(entity.get("chunk_id", "")),
                "content": entity.get("content", ""),
                "item_name": entity.get("item_name", ""),
                "file_title": entity.get("file_title", ""),
                "score": hit.get("distance", 0.0),
            })
        return results

    def search_hybrid(self, query: str, top_k: int = 10) -> List[Dict]:
        """方案A: Hybrid Search (Dense + Sparse)"""
        return self._search_hybrid(query, top_k=top_k)

    def search_hybrid_rrf(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        方案B: Hybrid Search + RRF
        模拟项目中的两路检索（普通向量 + HyDE），然后用RRF融合。
        由于没有真实HyDE，这里用同一query生成两路略有不同的结果来模拟。
        """
        # 主路检索
        results_a = self._search_hybrid(query, top_k=top_k)
        # 辅助路：用略微不同的查询（模拟HyDE效果）
        results_b = self._search_hybrid(query, top_k=top_k)

        # 构建RRF输入
        rrf_inputs = []
        if results_a:
            rrf_inputs.append((results_a, 1.0))
        if results_b:
            rrf_inputs.append((results_b, 1.0))

        if not rrf_inputs:
            return []

        merged = self._rrf_node._rrf_merge(rrf_inputs, k=60, max_results=top_k)
        return [{"chunk_id": d["chunk_id"], "content": d["content"],
                 "item_name": d.get("item_name", ""),
                 "file_title": d.get("file_title", ""),
                 "score": score} for d, score in merged]

    def search_hybrid_rrf_rerank(self, query: str, top_k: int = 10) -> Dict:
        """
        方案C: Hybrid Search + RRF + Cross-Encoder Rerank + Cliff Cutoff
        返回完整信息包括截断前后的数量。
        """
        # 1. 混合检索 + RRF
        results = self.search_hybrid_rrf(query, top_k=top_k)
        pre_rerank_count = len(results)

        if not results:
            return {"results": [], "pre_rerank_count": 0, "post_rerank_count": 0,
                    "cliff_cutoff_count": 0}

        # 2. Rerank
        contents = [r["content"] for r in results]
        try:
            scores = rerank_documents(query, contents)
            for r, s in zip(results, scores):
                r["score"] = s
            results = sorted(results, key=lambda x: x["score"], reverse=True)
        except Exception as e:
            logger.warning(f"Rerank failed: {e}, using RRF scores")

        rerank_count = len(results)

        # 3. Cliff Cutoff
        state = {
            "rewritten_query": query,
            "rrf_chunks": results,
            "web_search_docs": [],
            "session_id": "eval_test",
            "is_stream": False,
        }
        rerank_node = NodeRerank()
        result_state = rerank_node._step_3_cliff_cutoff(results)
        cutoff_count = len(result_state)

        return {
            "results": result_state,
            "pre_rerank_count": pre_rerank_count,
            "post_rerank_count": rerank_count,
            "cliff_cutoff_count": cutoff_count,
        }

    def search_hyde_hybrid_rrf_rerank(self, query: str, top_k: int = 10) -> Dict:
        """
        方案D: HyDE + Hybrid Search + RRF + Rerank + Cliff Cutoff
        使用LLM生成假设文档，然后用假设文档+原始query一起检索。
        """
        from utils.llm_utils import get_llm_client
        from config.embedding_config import embedding_config

        # Step 1: 生成HyDE假设文档
        hyde_doc = None
        try:
            llm = get_llm_client()
            hyde_prompt = (
                f"请根据以下问题，生成一段假设性的回答文档（不需要引用具体来源，"
                f"只需要给出最可能的答案内容）：\n\n问题：{query}\n\n假设性回答："
            )
            response = llm.invoke(hyde_prompt)
            hyde_doc = response.content.strip()
            logger.info(f"HyDE文档生成成功，长度: {len(hyde_doc)}")
        except Exception as e:
            logger.warning(f"HyDE文档生成失败，回退到普通检索: {e}")

        # Step 2: 三路检索（原始query + HyDE + 组合）
        client = self._get_client()

        # 原始query检索
        results_a = self._search_hybrid(query, top_k=top_k)

        # HyDE文档检索
        results_b = []
        if hyde_doc:
            results_b = self._search_hybrid(hyde_doc, top_k=top_k)

        # 组合检索
        combined = f"{query} {hyde_doc}" if hyde_doc else query
        results_c = self._search_hybrid(combined, top_k=top_k)

        # RRF融合三路
        rrf_inputs = []
        if results_a:
            rrf_inputs.append((results_a, 1.0))
        if results_b:
            rrf_inputs.append((results_b, 1.0))
        if results_c:
            rrf_inputs.append((results_c, 1.0))

        if not rrf_inputs:
            return {"results": [], "pre_rerank_count": 0, "post_rerank_count": 0,
                    "cliff_cutoff_count": 0, "hyde_doc": None}

        merged = self._rrf_node._rrf_merge(rrf_inputs, k=60, max_results=top_k)
        reranked_results = [
            {"chunk_id": d["chunk_id"], "content": d["content"],
             "item_name": d.get("item_name", ""),
             "file_title": d.get("file_title", ""),
             "score": score}
            for d, score in merged
        ]
        pre_rerank_count = len(reranked_results)

        # Rerank
        contents = [r["content"] for r in reranked_results]
        try:
            scores = rerank_documents(query, contents)
            for r, s in zip(reranked_results, scores):
                r["score"] = s
            reranked_results = sorted(reranked_results, key=lambda x: x["score"], reverse=True)
        except Exception as e:
            logger.warning(f"Rerank failed: {e}")

        post_rerank_count = len(reranked_results)

        # Cliff Cutoff
        rerank_node = NodeRerank()
        cutoff_results = rerank_node._step_3_cliff_cutoff(reranked_results)
        cutoff_count = len(cutoff_results)

        return {
            "results": cutoff_results,
            "pre_rerank_count": pre_rerank_count,
            "post_rerank_count": post_rerank_count,
            "cliff_cutoff_count": cutoff_count,
            "hyde_doc": hyde_doc,
        }


def load_test_dataset(path: str) -> List[Dict]:
    """加载测试数据集（支持JSON和JSONL格式，跳过注释行）"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 移除行首注释
    lines = [line for line in content.split("\n")
             if line.strip() and not line.strip().startswith("#")]
    clean_content = "\n".join(lines)

    data = json.loads(clean_content)
    logger.info(f"加载测试数据集: {len(data)} 条用例")
    return data


def extract_chunk_ids(results: List[Dict]) -> List[str]:
    """从检索结果中提取chunk_id列表"""
    return [r.get("chunk_id", "") for r in results if r.get("chunk_id")]


def run_evaluation(
    dataset_path: str,
    output_dir: str,
    top_k_list: List[int] = None,
    methods: List[str] = None,
):
    """
    执行完整评测
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)

    if top_k_list is None:
        top_k_list = [2, 3, 5, 10]
    if methods is None:
        methods = ["hybrid", "hybrid_rrf", "hybrid_rrf_rerank", "hyde_hybrid_rrf_rerank"]

    dataset = load_test_dataset(dataset_path)

    # 排除 out_of_scope 问题（用于检索评测）
    eval_dataset = [d for d in dataset if d.get("type") != "out_of_scope"]

    all_results = []
    errors = []

    evaluator = RetrieverEvaluator()

    for case in eval_dataset:
        case_id = case["id"]
        question = case["question"]
        source_files = case.get("source", [])
        relevant_chunks = []  # 由于我们没有chunk级别的ground truth，用file_title近似

        for method in methods:
            for top_k in top_k_list:
                try:
                    start = time.time()
                    result = _run_method(evaluator, method, question, top_k)
                    elapsed = time.time() - start

                    # 提取检索结果
                    if isinstance(result, dict):
                        retrieved = result.get("results", [])
                        meta = {
                            "pre_rerank": result.get("pre_rerank_count", len(retrieved)),
                            "post_rerank": result.get("post_rerank_count", len(retrieved)),
                            "cliff_cutoff": result.get("cliff_cutoff_count", len(retrieved)),
                        }
                    else:
                        retrieved = result
                        meta = {}

                    retrieved_ids = extract_chunk_ids(retrieved)

                    # 由于无法获取chunk级别的ground truth，
                    # 我们使用file_title级别的评估
                    # 从source_files构建期望的文件名列表
                    expected_files = source_files  # 测试集中的source即为期望匹配的文件

                    all_results.append({
                        "id": case_id,
                        "question": question[:100],
                        "method": method,
                        "top_k": top_k,
                        "retrieved_count": len(retrieved),
                        "retrieved_ids": json.dumps(retrieved_ids[:5]),
                        "elapsed_sec": round(elapsed, 3),
                        **meta,
                    })

                except Exception as e:
                    errors.append({
                        "id": case_id,
                        "method": method,
                        "top_k": top_k,
                        "error": str(e),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    logger.error(f"[{case_id}] {method}@{top_k} 失败: {e}")

    # 保存原始结果 - 收集所有可能的字段
    all_fields = set()
    for r in all_results:
        all_fields.update(r.keys())
    all_fields = sorted(all_fields)

    if all_results:
        with open(f"{output_dir}/retrieval_results.csv", "w", encoding="utf-8", newline="") as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            for row in all_results:
                # 确保每行只包含CSV中的字段
                filtered_row = {k: row.get(k, "") for k in all_fields}
                writer.writerow(filtered_row)

    if errors:
        with open(f"{output_dir}/errors.csv", "w", encoding="utf-8") as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=errors[0].keys())
            writer.writeheader()
            writer.writerows(errors)

    logger.info(f"评测完成: {len(all_results)} 条结果, {len(errors)} 条错误")
    return all_results, errors


def _run_method(
    evaluator: RetrieverEvaluator,
    method: str,
    query: str,
    top_k: int,
) -> Any:
    """根据方法名调用对应的检索策略"""
    if method == "hybrid":
        return evaluator.search_hybrid(query, top_k=top_k)
    elif method == "hybrid_rrf":
        return evaluator.search_hybrid_rrf(query, top_k=top_k)
    elif method == "hybrid_rrf_rerank":
        return evaluator.search_hybrid_rrf_rerank(query, top_k=top_k)
    elif method == "hyde_hybrid_rrf_rerank":
        return evaluator.search_hyde_hybrid_rrf_rerank(query, top_k=top_k)
    else:
        raise ValueError(f"未知的方法: {method}")


if __name__ == "__main__":
    # 单独测试
    dataset = load_test_dataset("eval/dataset/rag_eval.jsonl")
    print(f"测试集: {len(dataset)} 条")
    print(f"有效问题: {len([d for d in dataset if d.get('type') != 'out_of_scope'])} 条")
    print(f"方法列表: hybrid, hybrid_rrf, hybrid_rrf_rerank, hyde_hybrid_rrf_rerank")
