"""
processor/agent_processor/state.py

Agent 工作流状态定义。

本阶段（Step 2）只引入最小字段：
    user_query + triage

后续阶段（Memory / Tool / Diagnosis / Ticket / HITL）会按需扩展，
不在此处一次性堆叠。
"""

from typing import Optional, TypedDict

from pydantic import BaseModel, Field


class TriageResult(BaseModel):
    """
    Triage Agent 的结构化输出。

    设计约束：
      - 只描述"意图识别"结果，不涉及检索 / 诊断 / 工单。
      - retrieval_strategy 必须严格对齐 core.retrieval 的 SUPPORTED_STRATEGIES，
        禁止另造词汇（避免 eval / v1 / v2 三套命名）。
      - 缺失字段返回 None / []，不允许 LLM 凭空猜型号 / 错误码。
    """

    intent: str = Field(
        description=(
            "用户意图分类，可选值："
            "knowledge_question / fault_diagnosis / device_query / ticket_query / unknown"
        )
    )
    product_model: Optional[str] = Field(
        default=None,
        description="用户明确提到的产品型号；未明确提到时为 None，不得凭空猜测",
    )
    device_id: Optional[str] = Field(
        default=None,
        description="设备序列号 / 设备 ID；用户未提供时为 None",
    )
    error_code: Optional[str] = Field(
        default=None,
        description="用户提到的错误码；未提供时为 None，不得凭空编造",
    )
    symptom: Optional[str] = Field(
        default=None,
        description="故障现象的简短描述；用户未描述时为 None",
    )
    urgency: str = Field(
        default="low",
        description="紧急程度：low / medium / high。用户未明确表达紧急程度时默认 low",
    )
    missing_information: list = Field(
        default_factory=list,
        description="完成诊断所缺的信息项列表，如 ['firmware_version', 'last_maintenance_date']；"
                    "无缺失时为空列表",
    )
    retrieval_strategy: str = Field(
        default="hybrid_rrf_rerank",
        description=(
            "推荐的检索策略，必须从 core.retrieval.retrieval_service.SUPPORTED_STRATEGIES 中选择："
            "hybrid / hybrid_rrf / hybrid_rrf_rerank / hyde_hybrid_rrf_rerank。"
            "未识别时默认 hybrid_rrf_rerank。"
        ),
    )


class AgentState(TypedDict, total=False):
    """
    LangGraph 状态（最小集）。

    本阶段只包含 user_query 与 triage；后续阶段按需追加字段
    （memory / tool_results / diagnosis / ticket / approval 等）。
    """

    user_query: str
    triage: TriageResult
