"""
core/retrieval/adaptive.py

Adaptive Retrieval 策略选择器（Step 13）。

契约（Step 13 要求）：
  - 默认（简单问题）     → hybrid
  - 复杂                 → hybrid_rrf_rerank（query 改写 / hybrid 增强）
  - 低置信度             → hyde_hybrid_rrf_rerank（rerank / HyDE）
  - 知识不足             → 由上层决定是否触发 MCP WebSearch（本层只给信号，
                            不直接接外部网络，避免无界副作用）

策略命名严格对齐 core.retrieval.retrieval_service.SUPPORTED_STRATEGIES。
纯函数，可单测（给定 triage + evidence 质量 → 选策略）。

本层只做**策略判定**；实际检索仍走 RetrievalService.search()。
"""

from dataclasses import dataclass, field
from typing import Optional

from core.retrieval.retrieval_service import SUPPORTED_STRATEGIES
from processor.agent_processor.state import TriageResult


@dataclass
class AdaptiveDecision:
    """Adaptive Retrieval 的输出：选定的 strategy + 决策理由 + 是否建议 web 搜索。"""
    strategy: str
    reason: str
    suggest_web_search: bool = False
    confidence: str = "normal"  # high / normal / low

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "reason": self.reason,
            "suggest_web_search": self.suggest_web_search,
            "confidence": self.confidence,
        }


def _is_complex_intent(triage: Optional[TriageResult]) -> bool:
    """复杂问题判定：故障诊断 / 带型号或错误码。"""
    if triage is None:
        return False
    if triage.intent == "fault_diagnosis":
        return True
    if triage.error_code and triage.product_model:
        return True
    if len((triage.missing_information or [])) >= 2:
        return True
    return False


def _low_confidence_evidence(retrieval: Optional[dict]) -> bool:
    """低置信度判定：无文档 / 最高分过低 / 检索带 error。"""
    if not retrieval:
        return False
    docs = retrieval.get("documents") or []
    if retrieval.get("error"):
        return True
    if not docs:
        return True
    scores = [d.get("score") for d in docs if isinstance(d.get("score"), (int, float))]
    if not scores:
        return True
    # 最高分低于阈值 → 证据质量低（0.3 为经验阈值，可配置）
    return max(scores) < 0.3


def _knowledge_insufficient(triage: Optional[TriageResult], retrieval: Optional[dict]) -> bool:
    """知识不足判定：仅当**跑过检索且确实无证据**（retrieval 已产出）才算。"""
    complex_q = _is_complex_intent(triage)
    has_retrieval = retrieval is not None
    no_docs = not (retrieval or {}).get("documents")
    return complex_q and has_retrieval and no_docs


def pick_retrieval_strategy(
    triage: Optional[TriageResult],
    retrieval: Optional[dict] = None,
) -> AdaptiveDecision:
    """
    根据 triage（意图/复杂度）+ 已有 retrieval 证据质量，选检索策略。

    优先级（高→低）：
      1. 检索已跑过且无证据（知识不足）→ hyde_hybrid_rrf_rerank + 建议 web 搜索兜底
      2. 低置信度（已有证据质量低 / 检索 error）→ hyde_hybrid_rrf_rerank（rerank / HyDE 二次检索）
      3. 复杂问题（首轮 / 证据充分）→ hybrid_rrf_rerank
      4. 简单问题（默认）→ hybrid
    """
    # 1) 知识不足（复杂 + 已检索 + 无证据）→ HyDE + 建议 web 搜索
    if _knowledge_insufficient(triage, retrieval):
        return AdaptiveDecision(
            strategy="hyde_hybrid_rrf_rerank",
            reason="复杂问题且本地知识库无证据，启用 HyDE 二次检索并建议 web 搜索兜底",
            suggest_web_search=True,
            confidence="low",
        )

    # 2) 低置信度 → HyDE（rerank / HyDE 增强）
    if _low_confidence_evidence(retrieval):
        return AdaptiveDecision(
            strategy="hyde_hybrid_rrf_rerank",
            reason="已有证据置信度低，启用 HyDE + Rerank 提升相关性",
            confidence="low",
        )

    # 3) 复杂问题 → 完整链路
    if _is_complex_intent(triage):
        return AdaptiveDecision(
            strategy="hybrid_rrf_rerank",
            reason="复杂故障诊断，启用完整混合检索 + RRF + Rerank",
            confidence="normal",
        )

    # 4) 默认简单 → 最简 hybrid
    return AdaptiveDecision(
        strategy="hybrid",
        reason="简单知识问题，默认最简 hybrid 检索",
        confidence="high",
    )


def choose_first_strategy(triage: Optional[TriageResult]) -> AdaptiveDecision:
    """首轮检索（尚无证据）的策略选择。"""
    return pick_retrieval_strategy(triage, retrieval=None)


__all__ = [
    "AdaptiveDecision",
    "pick_retrieval_strategy",
    "choose_first_strategy",
]
