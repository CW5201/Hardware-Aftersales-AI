"""
processor/agent_processor/nodes/memory.py

Memory 节点（Step 7）：Graph 里的 memory retrieve / persist。

两个节点：
  - memory_retrieve: 读 thread short-term + customer/device long-term，
    写回 state["memory"]（短期槽 + 长期上下文）
  - memory_persist: 把本轮 triage/retrieval/diagnosis/tool 结果写回短期槽，
    关键事件（fault/ticket）写长期记忆

state 新增字段：
  memory: {short_term, long_term}
"""

from processor.agent_processor.state import AgentState
from services.memory.memory_service import MemoryService
from tool.logger import logger


def _extract_ids(state: AgentState) -> tuple:
    """从 state 里抠 customer_id / device_id（triage 或显式字段）。"""
    triage = state.get("triage")
    device_id = None
    customer_id = None
    if triage is not None:
        device_id = getattr(triage, "device_id", None)
    # 显式 state 字段优先
    device_id = state.get("device_id") or device_id
    customer_id = state.get("customer_id")
    return customer_id, device_id


class MemoryRetrieveNode:
    """Graph 节点：把 thread 短期 + customer/device 长期记忆注入 state。"""

    def __init__(self, memory: MemoryService):
        self._mem = memory

    def __call__(self, state: AgentState) -> AgentState:
        thread_id = state.get("thread_id") or "default"
        customer_id, device_id = _extract_ids(state)
        ctx = self._mem.build_memory_context(thread_id, customer_id, device_id)
        logger.info(
            f"MemoryRetrieve: thread={thread_id}, customer={customer_id}, "
            f"devices={len(ctx['long_term']['devices'])}, faults={len(ctx['long_term']['faults'])}"
        )
        return {**state, "memory": ctx}


class MemoryPersistNode:
    """Graph 节点：把本轮中间态写回短期；故障/工单事件写长期。"""

    def __init__(self, memory: MemoryService):
        self._mem = memory

    def __call__(self, state: AgentState) -> AgentState:
        thread_id = state.get("thread_id") or "default"
        customer_id, device_id = _extract_ids(state)

        # 短期：写回本轮中间产物
        self._mem.short_update(
            thread_id,
            user_query=state.get("user_query"),
            triage=state.get("triage"),
            retrieval=state.get("retrieval"),
            diagnosis=state.get("diagnosis"),
        )

        # 长期：有 diagnosis 且非 retrieval_failed → 记一次故障关联
        diagnosis = state.get("diagnosis")
        if diagnosis is not None and customer_id and device_id:
            if getattr(diagnosis, "diagnosis_status", None) == "diagnosed":
                self._mem.long.record_fault(
                    customer_id,
                    device_id,
                    {
                        "device_id": device_id,
                        "diagnosis": getattr(diagnosis, "diagnosis_result", None),
                        "hypotheses": [h.model_dump() for h in diagnosis.hypotheses],
                    },
                )

        # 重新拉一份 memory 上下文写回 state（含本轮更新）
        ctx = self._mem.build_memory_context(thread_id, customer_id, device_id)
        logger.info(f"MemoryPersist: thread={thread_id} 已写回短期 + 长期")
        return {**state, "memory": ctx}
