"""
processor/agent_processor/nodes/triage.py

Triage Agent：把用户自然语言报修/提问结构化，产出 TriageResult。

职责边界（本阶段）：
  - 只做【意图识别 + 信息抽取 + 紧急度判断 + 策略推荐】
  - 不检索、不调 Tool、不接 Memory、不出答案
  - 缺失字段一律 None / []，不凭空猜型号 / 错误码
  - retrieval_strategy 严格对齐 SUPPORTED_STRATEGIES
"""

import json
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage
from tool.logger import logger
from utils.llm_utils import get_llm_client

from processor.agent_processor.state import AgentState, TriageResult
from core.retrieval.retrieval_service import SUPPORTED_STRATEGIES

# ---------------- Prompt ----------------

TRIAGE_SYSTEM_PROMPT = (
    "你是工业硬件售后场景的 Triage Agent，负责把用户的报修/提问结构化。\n"
    "你必须：\n"
    "1. 严格基于用户原文，禁止凭空猜测产品型号、设备 ID、错误码。\n"
    "2. 用户未提到的字段一律返回 null；missing_information 用列表表示'还需要补充什么'。\n"
    "3. urgency 只有用户明确表达紧急程度时才设为 medium/high，否则默认 low。\n"
    "4. retrieval_strategy 必须从给定列表中选择，不要创造新值。\n"
)

TRIAGE_USER_TEMPLATE = """用户问题：
{query}

可选的 retrieval_strategy 列表（只能从中选一个）：
{strategies}

intent 可选值：knowledge_question / fault_diagnosis / device_query / ticket_query / unknown
urgency 可选值：low / medium / high

请按以下 JSON 输出（缺失字段用 null，missing_information 为数组）：
{{
  "intent": "...",
  "product_model": "..." 或 null,
  "device_id": "..." 或 null,
  "error_code": "..." 或 null,
  "symptom": "..." 或 null,
  "urgency": "...",
  "missing_information": [...],
  "retrieval_strategy": "..."
}}"""


def _default_strategy() -> str:
    """LLM 输出未落到 SUPPORTED_STRATEGIES 时的兜底：默认最简 hybrid。"""
    return "hybrid_rrf_rerank"


def build_triage_prompt(query: str, strategies: Optional[list] = None) -> tuple:
    """构造 Triage Prompt（拆成 system/user 便于单测与 LLM 调用）。"""
    strategies = strategies if strategies else list(SUPPORTED_STRATEGIES)
    user_prompt = TRIAGE_USER_TEMPLATE.format(
        query=query, strategies=json.dumps(strategies, ensure_ascii=False)
    )
    return TRIAGE_SYSTEM_PROMPT, user_prompt


def parse_triage_response(content: str) -> TriageResult:
    """把 LLM 原始文本解析为 TriageResult（健壮的 JSON 清洗 + 枚举兜底）。

    不抛异常：解析失败时返回一个保守的 TriageResult（intent=unknown），
    让上层 Graph 能稳定走完，而不是整条链挂掉。
    """
    cleaned = (content or "").strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.endswith("```"):
        cleaned = cleaned[: -3]
    cleaned = cleaned.strip()

    data = {}
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(f"Triage JSON 解析失败，回退 unknown: {content[:200]}")

    # 归一化：只保留 TriageResult 认识的字段，缺省值由 Pydantic 兜底
    result = TriageResult(
        intent=data.get("intent") or "unknown",
        product_model=data.get("product_model"),
        device_id=data.get("device_id"),
        error_code=data.get("error_code"),
        symptom=data.get("symptom"),
        urgency=data.get("urgency") or "low",
        missing_information=data.get("missing_information") or [],
        retrieval_strategy=data.get("retrieval_strategy") or _default_strategy(),
    )
    # 强制约束：retrieval_strategy 必须落在 SUPPORTED_STRATEGIES 内
    if result.retrieval_strategy not in SUPPORTED_STRATEGIES:
        logger.warning(
            f"Triage 输出了未知 strategy={result.retrieval_strategy!r}，"
            f"回退默认 {_default_strategy()!r}"
        )
        result.retrieval_strategy = _default_strategy()
    return result


class TriageNode:
    """LangGraph 节点：User Query → TriageResult。

    用法（作为 LangGraph 节点）：
        graph.add_node("triage", TriageNode())

    也可脱离 Graph 单测：
        node = TriageNode()
        state = {"user_query": "..."}
        state = node(state)
        assert state["triage"].intent == "fault_diagnosis"
    """

    def __init__(self, llm=None):
        # llm 可注入，便于测试时 mock；默认用项目统一的 get_llm_client(json_mode=True)
        self._llm = llm

    def __call__(self, state: AgentState) -> AgentState:
        query = state.get("user_query") or ""
        system_prompt, user_prompt = build_triage_prompt(query)

        llm = self._llm if self._llm is not None else get_llm_client(json_mode=True)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        response = llm.invoke(messages)
        triage = parse_triage_response(response.content)

        logger.info(
            f"Triage: intent={triage.intent}, model={triage.product_model}, "
            f"strategy={triage.retrieval_strategy}, urgency={triage.urgency}"
        )
        return {**state, "triage": triage}
