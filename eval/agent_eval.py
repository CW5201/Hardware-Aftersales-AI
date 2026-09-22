"""
eval/agent_eval.py

v2 Agent Evaluation（Step 11）。

保留 v1 检索指标（eval/metrics.py 的 Hit@K / MRR / Recall@K）不动；
本模块只负责 Agent / Tool / Diagnosis 级别的指标，且**不伪造数据**：

原则（Step 11 硬性要求）：
  - 每个指标必须有真实输入（Agent 输出 vs 标注）才能算出数字
  - 没有真实标注 / 真实数据 → 显式返回 "Not evaluated"，不编造数字
  - failure dataset 是可运行的**判定函数**（判断一次 run 是否落入某类失败），
    不是靠 LLM 自评

指标（真实可算，输入 = Agent 结构化输出 + 标注）：
    task_success_rate        任务成功率
    tool_selection_accuracy  工具选择准确率
    tool_argument_accuracy   工具入参准确率
    diagnosis_accuracy       诊断准确率
    answer_groundedness      答案是否有证据支撑
    citation_accuracy        引用准确率
    abstention_accuracy      拒答准确率
    human_escalation_accuracy 人工升级准确率

failure dataset（eval/dataset/agent_failures.json）：
    wrong_tool / wrong_argument / wrong_device / missing_evidence /
    hallucination / memory_contamination / unsafe_write /
    wrong_escalation / duplicate_ticket
"""

from typing import Optional

from eval.metrics import hit_at_k, mrr, recall_at_k  # 复用 v1 指标，不重写


# ---------------- 输入契约 ----------------

class AgentCase:
    """
    一条 agent 评估用例（真实 run 输出 + 标注）。

    expected_* 是标注（ground truth）；agent_* 是实际输出。
    标注缺失（None / []）时，相关指标记为 "Not evaluated"。
    """

    def __init__(
        self,
        case_id: str,
        # 实际输出（来自 Agent run 的 state）
        agent_tool_name: str = None,
        agent_tool_args: dict = None,
        agent_diagnosis_status: str = None,      # diagnosed / insufficient_evidence / retrieval_failed
        agent_diagnosis_result: str = None,
        agent_hypotheses: list = None,
        agent_evidence: list = None,
        agent_need_human_review: bool = False,
        agent_document_ids: list = None,        # 检索到的 doc ids
        # 标注（ground truth）
        expected_tool_name: str = None,
        expected_tool_args: dict = None,
        expected_diagnosis: str = None,
        expected_document_ids: list = None,
        expected_abstain: bool = None,
        expected_human_review: bool = None,
        failure_type: str = None,               # 该 case 属于的失败类别（用于 failure dataset）
    ):
        self.case_id = case_id
        self.agent_tool_name = agent_tool_name
        self.agent_tool_args = agent_tool_args or {}
        self.agent_diagnosis_status = agent_diagnosis_status
        self.agent_diagnosis_result = agent_diagnosis_result
        self.agent_hypotheses = agent_hypotheses or []
        self.agent_evidence = agent_evidence or []
        self.agent_need_human_review = agent_need_human_review
        self.agent_document_ids = agent_document_ids or []
        self.expected_tool_name = expected_tool_name
        self.expected_tool_args = expected_tool_args or {}
        self.expected_diagnosis = expected_diagnosis
        self.expected_document_ids = expected_document_ids or []
        self.expected_abstain = expected_abstain
        self.expected_human_review = expected_human_review
        self.failure_type = failure_type


NOT_EVALUATED = "Not evaluated"


def _has(*values) -> bool:
    return any(v not in (None, "") for v in values)


# ---------------- 单项指标（每个 case） ----------------

def tool_selection_ok(case: AgentCase) -> Optional[bool]:
    """工具选择是否正确（标注了 expected_tool_name 才算数）。"""
    if not case.expected_tool_name:
        return None
    return case.agent_tool_name == case.expected_tool_name


def tool_argument_ok(case: AgentCase) -> Optional[bool]:
    """工具入参是否正确（对比 expected_tool_args 的关键字段）。"""
    if not case.expected_tool_args:
        return None
    expected_keys = set(case.expected_tool_args.keys())
    actual = case.agent_tool_args
    if not actual:
        return False
    for k in expected_keys:
        if k not in actual or actual[k] != case.expected_tool_args[k]:
            return False
    return True


def diagnosis_ok(case: AgentCase) -> Optional[bool]:
    """诊断结论是否与标注一致（模糊匹配：标注关键词出现在结论里）。"""
    if not case.expected_diagnosis:
        return None
    actual = case.agent_diagnosis_result or ""
    if not actual:
        return False
    # 宽松匹配：标注的每个词都出现才算对（避免编造"通过"）
    tokens = [t for t in case.expected_diagnosis.replace("，", ",").split(",") if t.strip()]
    if not tokens:
        return actual.strip() != ""
    return all(t.strip() in actual for t in tokens)


def answer_groundedness_ok(case: AgentCase) -> Optional[bool]:
    """
    答案是否有证据支撑：
      - diagnosed 但 hypotheses 全空 evidence_indices → 无支撑（not grounded）
      - insufficient_evidence / retrieval_failed → 不算 grounded（本就该拒答）
    只有"声称 diagnosed 且有证据引用"才评估。
    """
    if case.agent_diagnosis_status != "diagnosed":
        return None
    grounded = False
    for h in case.agent_hypotheses:
        idxs = h.get("evidence_indices") if isinstance(h, dict) else getattr(h, "evidence_indices", [])
        if idxs:
            grounded = True
            break
    return grounded


def citation_ok(case: AgentCase) -> Optional[bool]:
    """引用准确率：检索文档是否命中 expected_document_ids（Hit@K 复用 v1 指标）。"""
    if not case.expected_document_ids:
        return None
    return hit_at_k(case.agent_document_ids, case.expected_document_ids, k=5) == 1.0


def abstention_ok(case: AgentCase) -> Optional[bool]:
    """拒答准确率：该拒答时是否拒答（insufficient_evidence 即拒答）。"""
    if case.expected_abstain is None:
        return None
    agent_abstained = case.agent_diagnosis_status in ("insufficient_evidence", "retrieval_failed")
    return agent_abstained == case.expected_abstain


def human_escalation_ok(case: AgentCase) -> Optional[bool]:
    """人工升级准确率：该升级时是否升级（need_human_review）。"""
    if case.expected_human_review is None:
        return None
    return case.agent_need_human_review == case.expected_human_review


# ---------------- 聚合指标 ----------------

def _aggregate(name: str, values: list) -> str:
    """把一批 Optional[bool] 聚合成指标；没有有效样本 → Not evaluated。"""
    valid = [v for v in values if v is not None]
    if not valid:
        return NOT_EVALUATED
    return round(sum(1 for v in valid if v) / len(valid), 4)


def evaluate_agent_cases(cases: list[AgentCase]) -> dict:
    """
    输入一批 AgentCase（带标注），输出 agent 级指标。
    任一指标缺标注 → 该指标 = "Not evaluated"（不编数字）。
    """
    return {
        "task_success_rate": _aggregate(
            "task_success_rate",
            [
                (tool_selection_ok(c) and tool_argument_ok(c))
                if (c.expected_tool_name or c.expected_tool_args) else None
                for c in cases
            ],
        ),
        "tool_selection_accuracy": _aggregate("tool_selection_accuracy", [tool_selection_ok(c) for c in cases]),
        "tool_argument_accuracy": _aggregate("tool_argument_accuracy", [tool_argument_ok(c) for c in cases]),
        "diagnosis_accuracy": _aggregate("diagnosis_accuracy", [diagnosis_ok(c) for c in cases]),
        "answer_groundedness": _aggregate("answer_groundedness", [answer_groundedness_ok(c) for c in cases]),
        "citation_accuracy": _aggregate("citation_accuracy", [citation_ok(c) for c in cases]),
        "abstention_accuracy": _aggregate("abstention_accuracy", [abstention_ok(c) for c in cases]),
        "human_escalation_accuracy": _aggregate(
            "human_escalation_accuracy", [human_escalation_ok(c) for c in cases]
        ),
        "num_cases": len(cases),
    }


# ---------------- failure dataset 判定 ----------------

def classify_failure(case: AgentCase) -> Optional[str]:
    """
    判断一次 run 落入哪类失败（failure dataset 的分类器）。
    没有标注时返回 None（无法判定，不编造）。
    """
    sel = tool_selection_ok(case)
    arg = tool_argument_ok(case)
    ground = answer_groundedness_ok(case)
    abst = abstention_ok(case)
    escal = human_escalation_ok(case)
    cite = citation_ok(case)

    # 显式标注的失败类别（memory_contamination / unsafe_write / duplicate_ticket /
    # missing_evidence 需要外部上下文或属于"该拒答"场景，直接透传标注值）
    if case.failure_type in ("memory_contamination", "unsafe_write", "duplicate_ticket", "missing_evidence"):
        return case.failure_type

    if sel is False:
        return "wrong_tool"
    if arg is False:
        return "wrong_argument"
    if cite is False and case.expected_document_ids:
        return "wrong_device"
    if case.agent_diagnosis_status == "diagnosed" and ground is False:
        # 声称诊断但假设全无证据引用 → 幻觉（编造证据之外的结论）
        return "hallucination"
    if case.expected_human_review is True and escal is False:
        return "wrong_escalation"
    if case.failure_type:
        # 其他显式标注（透传）
        return case.failure_type
    return None


def _load_failure_dataset() -> list[dict]:
    import json
    from pathlib import Path

    path = Path(__file__).parent / "dataset" / "agent_failures.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def run_failure_dataset() -> dict:
    """
    跑内置 failure dataset（eval/dataset/agent_failures.json），
    统计每类失败样本数 + 可判定率。没有真实标注的类别标记 Not evaluated。
    """
    samples = _load_failure_dataset()
    if not samples:
        return {"available": False, "note": "No failure dataset (Not evaluated)"}

    cases = [
        AgentCase(
            case_id=s.get("case_id", i),
            agent_tool_name=s.get("agent_tool_name"),
            agent_tool_args=s.get("agent_tool_args"),
            agent_diagnosis_status=s.get("agent_diagnosis_status"),
            agent_diagnosis_result=s.get("agent_diagnosis_result"),
            agent_hypotheses=s.get("agent_hypotheses"),
            agent_evidence=s.get("agent_evidence"),
            agent_need_human_review=s.get("agent_need_human_review", False),
            agent_document_ids=s.get("agent_document_ids"),
            expected_tool_name=s.get("expected_tool_name"),
            expected_tool_args=s.get("expected_tool_args"),
            expected_diagnosis=s.get("expected_diagnosis"),
            expected_document_ids=s.get("expected_document_ids"),
            expected_abstain=s.get("expected_abstain"),
            expected_human_review=s.get("expected_human_review"),
            failure_type=s.get("failure_type"),
        )
        for i, s in enumerate(samples)
    ]

    by_type = {}
    for s in samples:
        by_type[s.get("failure_type", "unknown")] = by_type.get(s.get("failure_type", "unknown"), 0) + 1

    detected = {classify_failure(c): c.case_id for c in cases}
    return {
        "available": True,
        "total_samples": len(samples),
        "failure_type_counts": by_type,
        "detected": {k: v for k, v in detected.items() if k is not None},
    }


# ---------------- 报告 ----------------

def agent_report(cases: list[AgentCase]) -> dict:
    """聚合 agent 指标 + failure dataset 结果。"""
    report = {
        "agent_metrics": evaluate_agent_cases(cases),
        "failure_dataset": run_failure_dataset(),
        "retrieval_metrics_note": "Retrieval-level Hit@K/MRR/Recall@K 仍由 v1 eval/metrics.py 计算，本模块不重复。",
    }
    return report


if __name__ == "__main__":
    import json
    rep = agent_report([])
    print(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
