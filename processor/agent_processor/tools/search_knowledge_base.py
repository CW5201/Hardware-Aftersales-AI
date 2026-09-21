"""
processor/agent_processor/tools/search_knowledge_base.py

把 Step 1 的 RetrievalService 包装成 Agent 可调用的 LangChain Tool。

严格契约（本阶段）：
  - Tool → RetrievalService.search() → v1 检索链路
    **禁止** Tool 直接 import utils/milvus_utils；
    **禁止** 重新实现 embedding / hybrid / HyDE / RRF / rerank / cliff cutoff。
  - 入参 `retrieval_strategy` 严格对齐 core.retrieval.SUPPORTED_STRATEGIES，
    不另造 enum。
  - 返回 `RetrievalResult`（结构化 Evidence），不返回 LLM 答案字符串。
  - 空 query：明确抛错（安全返回空 evidence 亦可，但不能"静默打到底层"）。
  - RetrievalService 异常：**不吞掉**（返回带 error 标记的 evidence，
    避免 Agent 误判"知识库无结果"）。

设计：
  - 用 StructuredTool.from_function + Pydantic args 模型，
    让 LLM 侧能感知参数 schema；
  - service 可注入（默认全局单例），便于单测 mock。
"""

from __future__ import annotations

import threading
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from core.retrieval.retrieval_service import (
    RetrievalDocument,
    RetrievalResult,
    RetrievalService,
    SUPPORTED_STRATEGIES,
)

# ---------------- 全局单例（惰性构造，避免 import 阶段连 Milvus） ----------------
_service_lock = threading.Lock()
_shared_service: Optional[RetrievalService] = None


def get_search_service() -> RetrievalService:
    """获取全局共享的 RetrievalService（可被测试 monkeypatch 覆盖）。"""
    global _shared_service
    if _shared_service is None:
        with _service_lock:
            if _shared_service is None:
                _shared_service = RetrievalService()
    return _shared_service


# ---------------- 入参 schema ----------------
class SearchKBArgs(BaseModel):
    """search_knowledge_base 的入参。retrieval_strategy 约束到 SUPPORTED_STRATEGIES。"""

    query: str = Field(..., description="用户问题 / 改写后的检索 query")
    product_model: Optional[str] = Field(
        default=None, description="商品/产品型号，用于 item_name 过滤；无则全库"
    )
    retrieval_strategy: str = Field(
        default="hybrid_rrf_rerank",
        description=f"检索策略，必须 ∈ {list(SUPPORTED_STRATEGIES)}",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="返回的最大文档数")


def _run_search(args: SearchKBArgs, service: Optional[RetrievalService] = None) -> dict:
    """
    实际执行检索，返回 KnowledgeEvidence（复用 RetrievalResult 结构 + 顶层 error 标记）。

    返回 dict 形式（方便 LangChain Tool 的 JSON 序列化）：
        {
          "documents": [...], "strategy": ..., "latency": ...,
          "metadata": {...}, "error": None | str
        }
    """
    service = service if service is not None else get_search_service()

    # 1) 空 query：明确报错，不打底层
    if not args.query or not args.query.strip():
        return {
            "documents": [],
            "strategy": args.retrieval_strategy,
            "latency": 0.0,
            "metadata": {},
            "error": "empty query",
        }

    # 2) 非法 strategy：防御性兜底（Pydantic 已约束，这里再保险一层）
    if args.retrieval_strategy not in SUPPORTED_STRATEGIES:
        return {
            "documents": [],
            "strategy": args.retrieval_strategy,
            "latency": 0.0,
            "metadata": {},
            "error": f"invalid retrieval_strategy: {args.retrieval_strategy!r}, "
                     f"expected one of {list(SUPPORTED_STRATEGIES)}",
        }

    # 3) 调 RetrievalService；异常不吞——带 error 标记返回
    try:
        result: RetrievalResult = service.search(
            query=args.query,
            product_model=args.product_model,
            strategy=args.retrieval_strategy,
            top_k=args.top_k,
        )
        ev = result.to_dict()
        ev["error"] = None
        return ev
    except Exception as e:  # noqa: BLE001 —— 有意捕获但**不静默**，把错误显式带出
        return {
            "documents": [],
            "strategy": args.retrieval_strategy,
            "latency": 0.0,
            "metadata": {},
            "error": f"retrieval_failed: {type(e).__name__}: {e}",
        }


def make_search_knowledge_base_tool(
    service: Optional[RetrievalService] = None,
) -> StructuredTool:
    """
    构造 search_knowledge_base Tool。

    :param service: 可注入的 RetrievalService；None → 用全局共享单例。
    :return: LangChain StructuredTool，可直接 `tool.invoke({...})` 或喂给 Agent。
    """

    def _invoke(**kwargs) -> dict:
        # LangChain 把 args_schema 的字段展开成关键字传入
        args = SearchKBArgs(**kwargs)
        return _run_search(args, service=service)

    return StructuredTool.from_function(
        func=_invoke,
        name="search_knowledge_base",
        description=(
            "检索本地售后知识库，返回结构化证据（chunk 列表 + 分数 + 元数据），"
            "不返回 LLM 最终答案。"
        ),
        args_schema=SearchKBArgs,
    )


# ---------------- 顶层便捷工厂 + Triage→Tool 参数映射 ----------------
search_knowledge_base: StructuredTool = make_search_knowledge_base_tool()


def build_tool_args_from_triage(
    user_query: str,
    product_model: Optional[str],
    retrieval_strategy: str,
    top_k: int = 5,
) -> dict:
    """
    把 TriageResult 的字段映射到 Tool 入参（纯映射，**不再调 LLM**）。

        TriageResult.product_model       → Tool.product_model
        TriageResult.retrieval_strategy  → Tool.retrieval_strategy
        User Query                       → Tool.query
    """
    return {
        "query": user_query,
        "product_model": product_model,
        "retrieval_strategy": retrieval_strategy,
        "top_k": top_k,
    }


if __name__ == "__main__":
    # 最小自检（需 Milvus 可达；否则看 error 标记）
    args = SearchKBArgs(
        query="H3C LA2608 如何配置无线控制器",
        product_model=None,
        retrieval_strategy="hybrid_rrf_rerank",
        top_k=5,
    )
    print(_run_search(args))
