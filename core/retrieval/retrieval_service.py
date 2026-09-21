"""
core/retrieval/retrieval_service.py

RetrievalService —— 把 v1 已验证的检索能力抽象成 v1/v2 共同调用的 Facade。

设计约束（本阶段必须遵守）：
  1. 只做【编排 / 门面】，不重写任何检索算法。
     底层能力全部委托给现有实现：
       - 向量化:       utils.embedding_utils.generate_embeddings
       - 混合检索:     utils.milvus_utils.create_hybrid_search_requests + hybrid_search
       - RRF 融合:     processor.query_processor.nodes.node_rrf.NodeRrf._rrf_merge
       - Rerank 打分:  utils.reranker_http_utils.rerank_documents
       - 断崖截断:     processor.query_processor.nodes.node_rerank.NodeRerank._step_3_cliff_cutoff
       - HyDE 假设文档: utils.llm_utils.get_llm_client + processor.../prompt/search_embedding_hyde.HYDE_PROMPT
  2. strategy 命名严格对齐 eval/retriever.py 的 4 个方法名，
     避免 "eval 一套 / v1 一套 / v2 一套" 的词汇分裂：
       - "hybrid"
       - "hybrid_rrf"
       - "hybrid_rrf_rerank"
       - "hyde_hybrid_rrf_rerank"
  3. 返回结构化 RetrievalResult（evidence），不返回 LLM 答案、不拼 Prompt。
     供未来 Agent / Diagnosis 直接消费。

本阶段 RetrievalService 是【新增能力】，v1 的 query_processor 仍走原流程，
不切换到 Service（等回归锚点通过后再决定）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from config.milvus_config import milvus_config
from tool.logger import logger

# ---------------- 顶层只保留“纯 Python 配置”导入 ----------------
# 说明：
#   utils.embedding_utils 顶层会 `from pymilvus.model.hybrid import BGEM3EmbeddingFunction`
#   （间接拉入 FlagEmbedding 的 C 扩展）。在 pytest 的 assertion-rewriting 环境下，
#   若同一 C 扩展模块被二次 exec（例如 web.api.query_service 的 import 链再次触发
#   embedding_utils），Python 3.14 会因“同一进程 C 扩展二次初始化”而 segfault。
#   因此把会触发 C 扩展的底层工具全部改为“调用时 lazy import”，
#   使本模块顶层 import 不触碰任何 C 扩展，保证 pytest 可安全二次 exec。
#
# 检索算法本身（NodeRrf / NodeRerank / rerank / milvus / embedding）均在下方
# 各 primitive 方法内按需 import，行为与 v1 完全一致，只是延迟到真正调用时。

# 与 eval/retriever.py、node_search_embedding*.py 保持一致的底层参数
_HYBRID_RANKER_WEIGHTS = (0.8, 0.2)   # dense 权重更高（现有实现固定值）
_MILVUS_LIMIT = 10                    # 单路混合检索召回条数（现有实现固定值）
_RRF_K = 60                           # RRF 平滑参数（现有实现固定值）
_HYDE_TOP_K = 5                       # HyDE 融合后送入 Rerank 的条数（现有实现固定值）

SUPPORTED_STRATEGIES = (
    "hybrid",
    "hybrid_rrf",
    "hybrid_rrf_rerank",
    "hyde_hybrid_rrf_rerank",
)


@dataclass
class RetrievalDocument:
    """统一证据文档。字段以 v1 现有 chunk 结构为准（Milvus chunks_collection 的 output_fields）。"""
    chunk_id: Optional[str]
    content: str
    item_name: str = ""
    file_title: str = ""
    title: str = ""
    score: float = 0.0
    source: str = "local"
    url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    """RetrievalService.search 的返回结构。Agent / Diagnosis 可直接消费 evidence。"""
    documents: List[RetrievalDocument]
    strategy: str
    latency: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "documents": [d.to_dict() for d in self.documents],
            "strategy": self.strategy,
            "latency": self.latency,
            "metadata": self.metadata,
        }


class RetrievalService:
    """
    共享检索门面（Facade / Orchestrator）。

    用法（同步）：
        svc = RetrievalService()
        result = svc.search("H3C LA2608 如何配置无线控制器", product_model="H3CLA2608室内无线网关",
                            strategy="hybrid_rrf_rerank", top_k=5)
        for doc in result.documents:
            print(doc.chunk_id, doc.score, doc.content[:50])

    说明：
      - 当前实现为同步（与 v1 节点、eval/retriever.py 保持一致）。
        未来 Agent Tool 若需 async，可在 Service 上层包一层 async 门面，
        而本类内部仍复用同步底层，避免伪 async。
      - search_knowledge_base（Agent Tool）将直接调用本类，不 import utils/milvus_utils。
    """

    def __init__(self, collection_name: Optional[str] = None, client=None):
        # 默认走 v1 的 chunks 集合；client 默认复用全局 MilvusClient 单例
        self.collection_name = collection_name or milvus_config.chunks_collection
        self._client = client
        # RRF / Rerank 节点（纯 Python 算法，无 C 扩展）在 __init__ 构造一次
        from processor.query_processor.nodes.node_rrf import NodeRrf
        from processor.query_processor.nodes.node_rerank import NodeRerank
        self._rrf_node = NodeRrf()
        self._rerank_node = NodeRerank()

    # ------------------------------------------------------------------
    # 统一入口
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        product_model: Optional[str] = None,
        strategy: str = "hybrid_rrf_rerank",
        top_k: int = 5,
    ) -> RetrievalResult:
        """
        按 strategy 执行检索，返回结构化证据（不含 LLM 答案）。

        参数：
            query: 用户问题（或 v1 的 rewritten_query）。
            product_model: 商品/产品名（用于 item_name 过滤）。可传单个名字符串；
                           为保持与 v1 行为一致，内部按单值构造过滤表达式。
            strategy: 见 SUPPORTED_STRATEGIES，命名与 eval/retriever.py 对齐。
            top_k: 最终返回的最大文档数。

        异常：
            未知 strategy 直接 raise ValueError（不静默降级，便于 Agent 感知错误）。
        """
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f"未知检索策略: {strategy!r}，可选 {SUPPORTED_STRATEGIES}")

        start = time.time()
        if strategy == "hybrid":
            docs = self._search_hybrid(query, product_model, top_k)
        elif strategy == "hybrid_rrf":
            docs = self._search_hybrid_rrf(query, product_model, top_k)
        elif strategy == "hybrid_rrf_rerank":
            docs = self._search_hybrid_rrf_rerank(query, product_model, top_k)
        else:  # hyde_hybrid_rrf_rerank
            docs, hyde_doc = self._search_hyde_hybrid_rrf_rerank(query, product_model, top_k)
            metadata_hyde = {"hyde_doc": hyde_doc}
        metadata = {"product_model": product_model}
        if strategy == "hyde_hybrid_rrf_rerank":
            metadata.update(metadata_hyde)

        latency = time.time() - start
        return RetrievalResult(
            documents=docs[:top_k],
            strategy=strategy,
            latency=latency,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # 各策略实现（全部复用底层 primitive，不重写算法）
    # ------------------------------------------------------------------
    def _search_hybrid(self, query: str, product_model: Optional[str], top_k: int) -> List[RetrievalDocument]:
        """方案A：Dense + Sparse 混合检索，无融合/重排。"""
        hits = self._run_hybrid(query, product_model, top_k)
        return [self._to_doc(hit, source="local") for hit in hits]

    def _search_hybrid_rrf(self, query: str, product_model: Optional[str], top_k: int) -> List[RetrievalDocument]:
        """方案B：两路混合检索 + RRF 融合。"""
        a = self._run_hybrid(query, product_model, top_k)
        b = self._run_hybrid(query, product_model, top_k)  # 与 eval 一致：同 query 模拟两路
        merged = self._rrf_merge(a, b, max_results=top_k)
        return [self._to_doc(d, source="local") for d, _score in merged]

    def _search_hybrid_rrf_rerank(self, query: str, product_model: Optional[str], top_k: int) -> List[RetrievalDocument]:
        """方案C：两路混合检索 + RRF + Cross-Encoder Rerank + 断崖截断。"""
        a = self._run_hybrid(query, product_model, top_k)
        b = self._run_hybrid(query, product_model, top_k)
        merged = self._rrf_merge(a, b, max_results=_HYDE_TOP_K)  # Rerank 前收敛到 top5（现有实现行为）
        reranked = self._rerank_and_cutoff(query, merged, top_k)
        return reranked

    def _search_hyde_hybrid_rrf_rerank(
        self, query: str, product_model: Optional[str], top_k: int
    ) -> tuple[List[RetrievalDocument], Optional[str]]:
        """方案D：HyDE 假设文档 + 混合检索 + RRF + Rerank + 断崖截断。返回 (docs, hyde_doc)。"""
        hyde_doc = self._generate_hyde_doc(query)
        # HyDE 路：用 "query + 假设文档" 拼接后混合检索（现有实现行为）
        combined = f"{query} {hyde_doc}" if hyde_doc else query
        a = self._run_hybrid(query, product_model, top_k)
        b = self._run_hybrid(combined, product_model, top_k)
        merged = self._rrf_merge(a, b, max_results=_HYDE_TOP_K)
        reranked = self._rerank_and_cutoff(query, merged, top_k)
        return reranked, hyde_doc

    # ------------------------------------------------------------------
    # 底层 primitive 封装（全部委托给现有实现）
    # ------------------------------------------------------------------
    def _run_hybrid(self, text: str, product_model: Optional[str], limit: int) -> List[Dict[str, Any]]:
        """混合检索：向量化 → 构造双路请求 → Milvus hybrid_search。复用 utils 现有函数。"""
        from utils.embedding_utils import generate_embeddings
        from utils.milvus_utils import create_hybrid_search_requests, hybrid_search

        client = self._get_client()
        embeddings = generate_embeddings([text])
        dense_vector = embeddings.get("dense")[0]
        sparse_vector = embeddings.get("sparse")[0]

        expr = self._build_item_name_expr(product_model)
        reqs = create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            expr=expr,
            limit=limit,
        )
        res = hybrid_search(
            client=client,
            collection_name=self.collection_name,
            reqs=reqs,
            ranker_weights=_HYBRID_RANKER_WEIGHTS,
            output_fields=["chunk_id", "content", "item_name", "file_title", "title"],
        )
        if not res or not res[0]:
            return []
        # 现有 hybrid_search 返回 [[hit,...]]，每个 hit 形如 {"entity": {...}, "distance": ...}
        return [
            {
                "chunk_id": hit.get("entity", {}).get("chunk_id"),
                "content": hit.get("entity", {}).get("content", ""),
                "item_name": hit.get("entity", {}).get("item_name", ""),
                "file_title": hit.get("entity", {}).get("file_title", ""),
                "title": hit.get("entity", {}).get("title", ""),
                "score": hit.get("distance", 0.0),
            }
            for hit in res[0]
        ]

    def _rrf_merge(self, list_a: List[Dict], list_b: List[Dict], max_results: Optional[int]) -> List:
        """复用 NodeRrf 的 RRF 融合算法（不重写）。输入为上面 _run_hybrid 产出的 dict 列表。"""
        rrf_inputs = [(list_a, 1.0), (list_b, 1.0)]
        return self._rrf_node._rrf_merge(rrf_inputs, k=_RRF_K, max_results=max_results)

    def _rerank_and_cutoff(
        self, query: str, rrf_mergered: List, top_k: int
    ) -> List[RetrievalDocument]:
        """复用 DashScope Rerank 打分 + NodeRerank 断崖截断（不重写算法）。"""
        docs = [dict(d) for d, _score in rrf_mergered]
        if not docs:
            return []
        contents = [d.get("content", "") for d in docs]
        try:
            from utils.reranker_http_utils import rerank_documents
            scores = rerank_documents(query, contents)
        except Exception as e:
            # 与现有 eval/retriever.py 行为一致：Rerank 失败时用 RRF 原分兜底
            logger.warning(f"RetrievalService Rerank 失败，回退 RRF 分数: {e}")
            scores = [d.get("score", 0.0) for d in docs]
        for d, s in zip(docs, scores):
            d["score"] = s
        docs.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        cutoff = self._rerank_node._step_3_cliff_cutoff(docs)
        return [self._to_doc(d, source="local") for d in cutoff]

    def _generate_hyde_doc(self, query: str) -> Optional[str]:
        """复用现有 HyDE Prompt + LLM 客户端生成假设文档。失败返回 None（与 v1 行为一致）。"""
        try:
            from utils.llm_utils import get_llm_client
            from processor.query_processor.prompt.search_embedding_hyde import HYDE_PROMPT
            llm = get_llm_client()
            prompt = HYDE_PROMPT.format(rewritten_query=query)
            return llm.invoke(prompt).content.strip()
        except Exception as e:
            logger.warning(f"RetrievalService HyDE 文档生成失败，回退普通检索: {e}")
            return None

    def _build_item_name_expr(self, product_model: Optional[str]) -> Optional[str]:
        """构造 Milvus 过滤表达式。None → 不过滤（全库）；有值 → item_name 匹配。

        注意：v1 现有节点用的是 f-string 直接拼接（未转义），这里同样保持"不改变现有行为"，
        仅支持单值 product_model；多值 / 特殊字符的转义留待 v2 统一处理（见 docs 记录）。
        """
        if not product_model:
            return None
        return f"item_name in ['{product_model}']"

    def _to_doc(self, entity: Dict[str, Any], source: str = "local") -> RetrievalDocument:
        return RetrievalDocument(
            chunk_id=entity.get("chunk_id"),
            content=entity.get("content", ""),
            item_name=entity.get("item_name", ""),
            file_title=entity.get("file_title", ""),
            title=entity.get("title", ""),
            score=entity.get("score", 0.0),
            source=source,
            url=entity.get("url"),
        )

    def _get_client(self):
        if self._client is not None:
            return self._client
        from utils.milvus_utils import get_milvus_client
        return get_milvus_client()


if __name__ == "__main__":
    # 最小自检：离线可跑（需 Milvus 可达）；仅验证 Facade 结构，不做真实 E2E
    svc = RetrievalService()
    res = svc.search("H3C LA2608 如何配置无线控制器", product_model=None, strategy="hybrid_rrf_rerank", top_k=5)
    logger.info(f"strategy={res.strategy} docs={len(res.documents)} latency={res.latency:.3f}s")
