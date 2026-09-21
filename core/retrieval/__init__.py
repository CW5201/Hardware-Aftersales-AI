"""
core/retrieval/__init__.py

v2 共享检索能力包。本阶段只包含 Facade（RetrievalService），
不重写任何 Milvus / Embedding / RRF / Rerank / 断崖截断的底层逻辑，
全部委托给 utils/ 与 processor/query_processor/ 的现有实现。
"""

from core.retrieval.retrieval_service import RetrievalService, RetrievalResult, RetrievalDocument

__all__ = ["RetrievalService", "RetrievalResult", "RetrievalDocument"]
