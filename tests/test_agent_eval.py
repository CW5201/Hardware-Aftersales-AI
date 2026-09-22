"""
tests/test_agent_eval.py

Step 11 Agent Evaluation 测试（不依赖真实 LLM / Milvus / DB）：

  单项指标（有标注 → 真算；无标注 → Not evaluated）：
    1. tool_selection_ok 对/错/未标注
    2. tool_argument_ok 对/错/未标注
    3. diagnosis_ok 命中/未命中/未标注
    4. answer_groundedness（diagnosed 无证据引用 → False；非 diagnosed → None）
    5. citation_ok（复用 v1 Hit@K）
    6. abstention_ok / human_escalation_ok

  聚合：
    7. evaluate_agent_cases 混合标注 → 各指标数字或 "Not evaluated"
    8. 全无标注 → 全 "Not evaluated"（不编数字）

  failure dataset：
    9. 加载 eval/dataset/agent_failures.json（9 类失败齐全）
   10. classify_failure 对可判定的 case 给出正确类别
   11. run_failure_dataset 统计每类样本数
"""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from eval.agent_eval import (
    AgentCase,
    NOT_EVALUATED,
    answer_groundedness_ok,
    abstention_ok,
    citation_ok,
    classify_failure,
    diagnosis_ok,
    evaluate_agent_cases,
    human_escalation_ok,
    run_failure_dataset,
    tool_argument_ok,
    tool_selection_ok,
)

REQUIRED_FAILURE_TYPES = {
    "wrong_tool",
    "wrong_argument",
    "wrong_device",
    "missing_evidence",
    "hallucination",
    "memory_contamination",
    "unsafe_write",
    "wrong_escalation",
    "duplicate_ticket",
}


# ================= 单项指标 =================

def test_tool_selection_ok():
    c = AgentCase("c1", agent_tool_name="create_service_ticket", expected_tool_name="create_service_ticket")
    assert tool_selection_ok(c) is True
    c2 = AgentCase("c2", agent_tool_name="get_repair_history", expected_tool_name="create_service_ticket")
    assert tool_selection_ok(c2) is False
    c3 = AgentCase("c3", agent_tool_name="create_service_ticket")  # 未标注
    assert tool_selection_ok(c3) is None


def test_tool_argument_ok():
    c = AgentCase("c1", agent_tool_args={"customer_id": "C1"}, expected_tool_args={"customer_id": "C1"})
    assert tool_argument_ok(c) is True
    c2 = AgentCase("c2", agent_tool_args={"customer_id": "X"}, expected_tool_args={"customer_id": "C1"})
    assert tool_argument_ok(c2) is False
    c3 = AgentCase("c3", agent_tool_args={"customer_id": "C1"})  # 未标注
    assert tool_argument_ok(c3) is None


def test_diagnosis_ok():
    c = AgentCase("c1", agent_diagnosis_result="电源模块供电不足，建议更换", expected_diagnosis="电源模块,供电不足")
    assert diagnosis_ok(c) is True
    c2 = AgentCase("c2", agent_diagnosis_result="", expected_diagnosis="电源模块")
    assert diagnosis_ok(c2) is False
    c3 = AgentCase("c3", agent_diagnosis_result="任意")  # 未标注
    assert diagnosis_ok(c3) is None


def test_answer_groundedness():
    # diagnosed 但假设全无证据引用 → False
    c = AgentCase(
        "c1",
        agent_diagnosis_status="diagnosed",
        agent_hypotheses=[{"text": "h", "evidence_indices": []}],
    )
    assert answer_groundedness_ok(c) is False
    # diagnosed 且有证据引用 → True
    c2 = AgentCase(
        "c2",
        agent_diagnosis_status="diagnosed",
        agent_hypotheses=[{"text": "h", "evidence_indices": [0, 1]}],
    )
    assert answer_groundedness_ok(c2) is True
    # 非 diagnosed → 不评估（None）
    c3 = AgentCase("c3", agent_diagnosis_status="insufficient_evidence")
    assert answer_groundedness_ok(c3) is None


def test_citation_ok_reuses_v1_hit_at_k():
    c = AgentCase("c1", agent_document_ids=["d1", "d2"], expected_document_ids=["d2", "d3"])
    assert citation_ok(c) is True  # d2 在 top 命中
    c2 = AgentCase("c2", agent_document_ids=["x", "y"], expected_document_ids=["d1"])
    assert citation_ok(c2) is False
    c3 = AgentCase("c3", agent_document_ids=["d1"])  # 未标注
    assert citation_ok(c3) is None


def test_abstention_ok_and_escalation_ok():
    c = AgentCase("c1", agent_diagnosis_status="insufficient_evidence", expected_abstain=True)
    assert abstention_ok(c) is True
    c2 = AgentCase("c2", agent_diagnosis_status="diagnosed", expected_abstain=True)
    assert abstention_ok(c2) is False

    e = AgentCase("e1", agent_need_human_review=True, expected_human_review=True)
    assert human_escalation_ok(e) is True
    e2 = AgentCase("e2", agent_need_human_review=False, expected_human_review=True)
    assert human_escalation_ok(e2) is False


# ================= 聚合 =================

def test_evaluate_mixed_cases():
    cases = [
        AgentCase("a1", agent_tool_name="create_service_ticket", agent_tool_args={"customer_id": "C1"},
                  expected_tool_name="create_service_ticket", expected_tool_args={"customer_id": "C1"}),
        AgentCase("a2", agent_tool_name="get_repair_history",
                  expected_tool_name="create_service_ticket"),  # 选错工具
    ]
    rep = evaluate_agent_cases(cases)
    assert rep["tool_selection_accuracy"] == 0.5
    assert rep["tool_argument_accuracy"] == 1.0  # 只有 a1 有标注
    assert rep["num_cases"] == 2
    # 未标注的指标 → Not evaluated
    assert rep["diagnosis_accuracy"] == NOT_EVALUATED
    assert rep["citation_accuracy"] == NOT_EVALUATED


def test_evaluate_no_labels_all_not_evaluated():
    cases = [AgentCase("x1", agent_tool_name="create_service_ticket")]
    rep = evaluate_agent_cases(cases)
    assert rep["tool_selection_accuracy"] == NOT_EVALUATED
    assert rep["diagnosis_accuracy"] == NOT_EVALUATED
    assert rep["task_success_rate"] == NOT_EVALUATED
    # num_cases 仍真实
    assert rep["num_cases"] == 1


# ================= failure dataset =================

def test_failure_dataset_has_all_nine_types():
    res = run_failure_dataset()
    assert res["available"] is True
    assert set(REQUIRED_FAILURE_TYPES) <= set(res["failure_type_counts"].keys())
    assert res["total_samples"] >= 9


def test_classify_failure_detects_known_types():
    res = run_failure_dataset()
    detected = res["detected"]
    # 可自动判定的类别必须被检测到
    for t in ("wrong_tool", "wrong_argument", "wrong_device", "missing_evidence", "wrong_escalation",
              "hallucination"):
        assert t in detected, f"{t} 未被判定"
    # 外部上下文透传的类别
    for t in ("memory_contamination", "unsafe_write", "duplicate_ticket"):
        assert t in detected, f"{t} 未判定（应为透传标注）"
    # 总数 = 全部样本（每样本都判定出一个类别）
    assert len(detected) == res["total_samples"]


def test_classify_failure_on_individual_case():
    c = AgentCase(
        "x",
        agent_tool_name="get_repair_history",
        expected_tool_name="create_service_ticket",
    )
    assert classify_failure(c) == "wrong_tool"
    c2 = AgentCase("y", agent_tool_name="a", expected_tool_name="a")  # 全对
    assert classify_failure(c2) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
