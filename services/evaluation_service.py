"""
services/evaluation_service.py

Agent Evaluation API 服务（Step 13 / 供 /api/v2/eval/agent）。

原则（Step 11 契约）：
  - 有带标注的真实 run 样本 → 真算 agent 级指标
  - 没有 → 返回 "Not evaluated"，不编数字
"""

from typing import Optional

from eval.agent_eval import AgentCase, agent_report
from services.trace.trace_store import get_trace_store


class EvalService:
    """从 trace store 收集 run，组装 agent 指标（缺标注 → Not evaluated）。"""

    def __init__(self, trace_store=None):
        self._trace = trace_store or get_trace_store()

    def collect_cases_from_traces(self, thread_id: Optional[str] = None) -> list[AgentCase]:
        """
        把 trace run 转成 AgentCase（用于有标注时真算指标）。
        本地模拟环境没有 ground truth 标注 → 产出的 case 各 expected_* 为空，
        evaluate 后指标 = Not evaluated（不伪造）。
        """
        runs = self._trace.get_thread_runs(thread_id) if thread_id else list(self._trace._runs.values())
        cases = []
        for run in runs:
            triage_evt = next((e for e in run.events if e.node == "triage"), None)
            diag_evt = next((e for e in run.events if e.node == "diagnosis"), None)
            tool_evt = next((e for e in run.events if e.tool_name), None)
            retrieval_evt = next((e for e in run.events if e.node == "retrieval"), None)

            out = diag_evt.output if diag_evt else {}
            cases.append(
                AgentCase(
                    case_id=run.run_id,
                    agent_tool_name=tool_evt.tool_name if tool_evt else None,
                    agent_tool_args=tool_evt.tool_args if tool_evt else None,
                    agent_diagnosis_status=out.get("status"),
                    agent_diagnosis_result=None,  # trace 摘要不存全文结论
                    agent_need_human_review=bool(out.get("need_human_review")),
                    agent_document_ids=(
                        list(range((retrieval_evt.retrieved_docs_count or 0)))
                        if retrieval_evt else None
                    ),
                    # 本地模拟无标注 → 以下 expected_* 留空（指标 = Not evaluated）
                )
            )
        return cases

    def report(self, thread_id: Optional[str] = None) -> dict:
        """返回 agent 指标 + failure dataset（无标注时 agent 指标 Not evaluated）。"""
        cases = self.collect_cases_from_traces(thread_id)
        rep = agent_report(cases)
        rep["note"] = (
            "本地模拟环境无 ground truth 标注：agent 级指标 = Not evaluated；"
            "failure dataset 为内置判定样本（真实可跑的判定函数）。"
        )
        return rep


if __name__ == "__main__":
    import json
    print(json.dumps(EvalService().report(), ensure_ascii=False, indent=2, default=str))
