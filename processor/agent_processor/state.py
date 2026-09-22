"""
processor/agent_processor/state.py

Agent 工作流状态定义。

字段随 Step 增量扩展（不一次性堆叠）：
  Step 2: user_query + triage
  Step 4: + retrieval（结构化 Evidence）
  Step 5: + diagnosis（DiagnosisResult）

后续阶段（Memory / Tool / Ticket / HITL）继续按需追加。
"""

from typing import Optional, TypedDict

from pydantic import BaseModel, Field


class Hypothesis(BaseModel):
    """一条诊断假设，必须引用证据。"""
    text: str = Field(description="假设内容，如'电源模块供电不足导致电压异常'")
    evidence_indices: list = Field(
        default_factory=list,
        description="支持该假设的 retrieval.documents 下标列表（0-based）；"
                    "无证据时为空（此时 confidence 应为 low）",
    )
    confidence: str = Field(
        default="low", description="low / medium / high，证据充分才允许 medium/high"
    )


class DiagnosisResult(BaseModel):
    """
    Diagnosis Agent 的结构化输出（Step 5）。

    设计约束：
      - diagnosis_status: diagnosed / insufficient_evidence / retrieval_failed
      - 只有 diagnosed 时 hypotheses / diagnosis_result 才允许有实质内容
      - need_more_information: 列出还缺什么信息
      - need_human_review: 检索失败或高置信不足时置 True
    """
    diagnosis_status: str = Field(
        description="diagnosed / insufficient_evidence / retrieval_failed"
    )
    hypotheses: list = Field(
        default_factory=list, description="Hypothesis 列表（Step 5 起结构化）"
    )
    evidence: list = Field(
        default_factory=list,
        description="支持诊断的证据摘要（来自 retrieval.documents 的内容摘录）",
    )
    diagnosis_result: Optional[str] = Field(
        default=None, description="最终诊断结论；仅 diagnosed 时非空"
    )
    recommended_actions: list = Field(
        default_factory=list, description="建议的处置动作列表"
    )
    need_more_information: list = Field(
        default_factory=list, description="还需补充的信息项"
    )
    need_human_review: bool = Field(
        default=False, description="是否需要人工介入（检索失败 / 高置信不足）"
    )


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
    LangGraph 状态（Step 5）。

    字段：
      - user_query:  原始用户输入
      - triage:      TriageResult（Step 2 产出）
      - retrieval:   结构化 Evidence（Step 4 产出）
      - diagnosis:   DiagnosisResult（Step 5 产出）

    后续按需追加（memory / ticket / approval 等）。
    """

    user_query: str
    triage: TriageResult
    retrieval: dict
    diagnosis: DiagnosisResult
