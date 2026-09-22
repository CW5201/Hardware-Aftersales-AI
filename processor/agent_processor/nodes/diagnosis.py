"""
processor/agent_processor/nodes/diagnosis.py

Diagnosis Agent：基于 Triage + Evidence 做结构化故障诊断。

职责（Step 5）：
  - 读 state.triage / state.retrieval
  - 有证据 → 基于证据生成 hypotheses / diagnosis_result / recommended_actions
  - 无证据（documents=[] 且 error=None）→ insufficient_evidence，
    need_more_information=True，**不编造答案**
  - 检索故障（error!=None）→ 不做正常诊断，need_human_review=True
  - 输出 DiagnosisResult（结构化 Pydantic），写回 state.diagnosis

严格契约：
  - 不能无证据编造诊断结论
  - 每条 hypothesis 必须引用 evidence_indices（对应 retrieval.documents 下标）
  - retrieval error 时跳过诊断，标记 need_human_review
"""

import json
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage

from processor.agent_processor.state import AgentState, DiagnosisResult, Hypothesis
from tool.logger import logger


# ---------------- Prompt ----------------

DIAGNOSIS_SYSTEM_PROMPT = (
    "你是工业硬件售后场景的 Diagnosis Agent。你只能基于提供的【证据】做诊断。\n"
    "规则：\n"
    "1. 每条 hypothesis 必须引用 evidence_indices（证据下标），不得凭空编造。\n"
    "2. 证据不足时 diagnosis_status=insufficient_evidence，diagnosis_result=null，"
    "并在 need_more_information 列出缺失项。\n"
    "3. 检索系统故障时 diagnosis_status=retrieval_failed，不做诊断，need_human_review=true。\n"
    "4. 只输出 JSON，不输出多余文字。"
)

DIAGNOSIS_USER_TEMPLATE = """【用户问题】
{query}

【Triage 结构化结果】
{triage_json}

【检索证据】（下标从 0 开始）
{evidence_json}

请输出 JSON：
{{
  "diagnosis_status": "diagnosed / insufficient_evidence / retrieval_failed",
  "hypotheses": [
    {{"text": "...", "evidence_indices": [0,2], "confidence": "low|medium|high"}},
  ],
  "evidence": ["证据摘要1", "证据摘要2"],
  "diagnosis_result": "最终结论" 或 null,
  "recommended_actions": ["动作1"],
  "need_more_information": ["缺失信息1"],
  "need_human_review": true/false
}}"""


def _format_evidence_for_prompt(evidence: dict, max_docs: int = 8) -> str:
    """把 retrieval evidence 里的 documents 截断成 prompt 友好的 JSON 列表。"""
    docs = (evidence or {}).get("documents") or []
    slim = []
    for i, d in enumerate(docs[:max_docs]):
        slim.append({
            "index": i,
            "content": (d.get("content") or "")[:500],
            "score": d.get("score"),
            "source": d.get("source"),
        })
    return json.dumps(slim, ensure_ascii=False, indent=2) if slim else "[]"


def _format_triage(triage) -> str:
    if triage is None:
        return "null"
    if hasattr(triage, "model_dump"):
        return json.dumps(triage.model_dump(), ensure_ascii=False)
    return json.dumps(dict(triage), ensure_ascii=False)


def build_diagnosis_prompt(
    query: str, triage, evidence: dict
) -> tuple:
    """构造 Diagnosis Prompt（system / user）。"""
    user_prompt = DIAGNOSIS_USER_TEMPLATE.format(
        query=query or "",
        triage_json=_format_triage(triage),
        evidence_json=_format_evidence_for_prompt(evidence),
    )
    return DIAGNOSIS_SYSTEM_PROMPT, user_prompt


def parse_diagnosis_response(content: str) -> DiagnosisResult:
    """把 LLM 原始文本解析为 DiagnosisResult（健壮 JSON 清洗 + 兜底）。"""
    cleaned = (content or "").strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[: -3]
    cleaned = cleaned.strip()

    data = {}
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Diagnosis JSON 解析失败，回退 insufficient_evidence: {content[:200]}")

    # 兜底：解析失败或字段缺失时保守输出
    status = data.get("diagnosis_status") or "insufficient_evidence"
    hypotheses_raw = data.get("hypotheses") or []
    hypotheses = []
    for h in hypotheses_raw:
        if not isinstance(h, dict):
            continue
        idxs = h.get("evidence_indices") or []
        if not isinstance(idxs, list):
            idxs = []
        hypotheses.append(
            Hypothesis(
                text=str(h.get("text") or ""),
                evidence_indices=[int(x) for x in idxs if str(x).lstrip("-").isdigit()],
                confidence=h.get("confidence") or "low",
            )
        )

    def _as_list(x):
        if isinstance(x, list):
            return [str(v) for v in x]
        return []

    return DiagnosisResult(
        diagnosis_status=status if status in ("diagnosed", "insufficient_evidence", "retrieval_failed") else "insufficient_evidence",
        hypotheses=hypotheses,
        evidence=_as_list(data.get("evidence")),
        diagnosis_result=data.get("diagnosis_result"),
        recommended_actions=_as_list(data.get("recommended_actions")),
        need_more_information=_as_list(data.get("need_more_information")),
        need_human_review=bool(data.get("need_human_review", False)),
    )


class DiagnosisNode:
    """LangGraph 节点：triage + retrieval → diagnosis。

    缺省 LLM 在 __call__ 时懒加载 get_llm_client（保持与 TriageNode 相同的延迟约定）。

    三种输入态：
      A. retrieval error!=None          → retrieval_failed，不做诊断
      B. documents=[] 且 error=None     → insufficient_evidence（不编造）
      C. documents 非空                 → 调 LLM 做诊断
    """

    def __init__(self, llm=None):
        self._llm = llm

    def __call__(self, state: AgentState) -> AgentState:
        triage = state.get("triage")
        evidence = state.get("retrieval") or {}

        # --- A. 检索故障：不做正常诊断 ---
        if evidence.get("error"):
            diagnosis = DiagnosisResult(
                diagnosis_status="retrieval_failed",
                need_human_review=True,
                need_more_information=["retrieval_error_recovery"],
            )
            logger.warning(
                f"DiagnosisNode: retrieval error={evidence.get('error')!r} → "
                f"跳过诊断，need_human_review=True"
            )
            return {**state, "diagnosis": diagnosis}

        # --- B. 无证据：insufficient_evidence，不编造 ---
        documents = evidence.get("documents") or []
        if not documents:
            diagnosis = DiagnosisResult(
                diagnosis_status="insufficient_evidence",
                need_more_information=triage.missing_information if triage else [],
                need_human_review=False,
            )
            logger.info("DiagnosisNode: 检索正常但无证据 → insufficient_evidence")
            return {**state, "diagnosis": diagnosis}

        # --- C. 有证据：调 LLM ---
        system_prompt, user_prompt = build_diagnosis_prompt(
            state.get("user_query", ""), triage, evidence
        )
        llm = self._llm if self._llm is not None else None
        if llm is None:
            from utils.llm_utils import get_llm_client
            llm = get_llm_client(json_mode=True)

        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            response = llm.invoke(messages)
            diagnosis = parse_diagnosis_response(response.content)
        except Exception as e:
            logger.exception(f"DiagnosisNode: LLM 调用失败: {e}")
            diagnosis = DiagnosisResult(
                diagnosis_status="insufficient_evidence",
                need_human_review=True,
                need_more_information=["llm_unavailable"],
            )

        # 防御：LLM 若声称 diagnosed 但 hypotheses 全空 → 降级
        if diagnosis.diagnosis_status == "diagnosed" and not diagnosis.hypotheses:
            diagnosis.diagnosis_status = "insufficient_evidence"
            diagnosis.diagnosis_result = None
            diagnosis.need_human_review = True

        logger.info(
            f"DiagnosisNode: status={diagnosis.diagnosis_status}, "
            f"hypotheses={len(diagnosis.hypotheses)}, need_human_review={diagnosis.need_human_review}"
        )
        return {**state, "diagnosis": diagnosis}
