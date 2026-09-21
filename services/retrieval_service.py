"""
services/retrieval_service.py

v2 Agent Tool 调用的统一入口（thin re-export）。

实际实现位于 core/retrieval/retrieval_service.py：
  - core.retrieval.RetrievalService
  - core.retrieval.RetrievalResult
  - core.retrieval.RetrievalDocument

为什么不直接在这里写实现：
  - Step 2 的目标是"不大规模移动已有代码、不破坏 v1"
  - `core/` 是"被 v1/v2 共享的检索能力"语义；`services/` 是"业务服务层"语义
  - 把实现放在 `core/retrieval/` 与 V2_ROADMAP 的"core/ 是共享内核"定位一致
  - Agent Tool 调 service 时 import 本文件即可，避免跨包导入路径混乱

用法（v2 未来的 search_knowledge_base Tool）：
    from services.retrieval_service import RetrievalService
    svc = RetrievalService()
    res = svc.search(query="...", product_model="H3CLA2608室内无线网关",
                     strategy="hybrid_rrf_rerank", top_k=5)
"""

from core.retrieval import RetrievalService, RetrievalResult, RetrievalDocument
from core.retrieval.retrieval_service import SUPPORTED_STRATEGIES

__all__ = [
    "RetrievalService",
    "RetrievalResult",
    "RetrievalDocument",
    "SUPPORTED_STRATEGIES",
]
