"""
services/memory/memory_service.py

v2 Memory 层实现（Step 7）。

Short-term（进程内）：
  - ShortTermMemory: 按 thread_id 存 ShortTermSlot（query/triage/retrieval/
    diagnosis/tool_calls/tool_results）
  - TTL 过期清理（默认 30 分钟），避免多租户下无界增长

Long-term（customer / device 维度）：
  - 抽象 MemoryBackend：get_fault_history / get_repair_history / get_tickets / record_ticket
  - InMemoryLongTermStore：默认实现（进程内，测试用）
  - LongTermMemory 直接复用 Step 6 的 BusinessService（历史故障/维修/工单都
    是业务数据，不另造一套）

数据隔离（硬性契约）：
  - 每个 memory API 必须带 customer_id 或 device_id；按 customer_id 查询时
    只返回该 customer 的设备/工单，跨 customer 访问返回空（不是报错），
    绝不泄露 customer B 的数据。
"""

import time
from typing import Optional

from tool.logger import logger

from services.business.business_service import BusinessService


# ---------------- Short-term ----------------

class ShortTermSlot(dict):
    """
    单个 thread 的短期记忆槽（就是 dict，便于 LangGraph state 存取）。

    字段：
      user_query / triage / retrieval / diagnosis /
      tool_calls / tool_results / customer_id / device_id /
      created_at / last_access
    """

    def __init__(self, data: Optional[dict] = None):
        super().__init__(data or {})
        self.setdefault("user_query", None)
        self.setdefault("triage", None)
        self.setdefault("retrieval", None)
        self.setdefault("diagnosis", None)
        self.setdefault("tool_calls", [])
        self.setdefault("tool_results", {})
        self.setdefault("customer_id", None)
        self.setdefault("device_id", None)
        self.setdefault("created_at", time.time())
        self.setdefault("last_access", time.time())


class ShortTermMemory:
    """进程内短期记忆：thread_id → ShortTermSlot，TTL 清理。"""

    def __init__(self, ttl_seconds: float = 30 * 60):
        self._slots: dict[str, ShortTermSlot] = {}
        self._ttl = ttl_seconds

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [tid for tid, s in self._slots.items() if now - s["last_access"] > self._ttl]
        for tid in expired:
            del self._slots[tid]
        if expired:
            logger.info(f"ShortTermMemory: 清理过期 thread {len(expired)} 个")

    def get(self, thread_id: str) -> Optional[ShortTermSlot]:
        self._purge_expired()
        slot = self._slots.get(thread_id)
        if slot is not None:
            slot["last_access"] = time.time()
        return slot

    def put(self, thread_id: str, slot: ShortTermSlot) -> None:
        self._slots[thread_id] = slot

    def update(self, thread_id: str, **fields) -> ShortTermSlot:
        slot = self.get(thread_id)
        if slot is None:
            slot = ShortTermSlot()
            self.put(thread_id, slot)
        slot.update(fields)
        slot["last_access"] = time.time()
        return slot

    def record_tool_call(self, thread_id: str, tool_name: str, args: dict, result: dict) -> None:
        slot = self.update(thread_id)
        slot["tool_calls"].append({"tool_name": tool_name, "args": args, "ts": time.time()})
        slot["tool_results"][tool_name] = result
        self.put(thread_id, slot)

    def clear(self, thread_id: str) -> None:
        self._slots.pop(thread_id, None)

    def __len__(self) -> int:
        self._purge_expired()
        return len(self._slots)


# ---------------- Long-term ----------------

class InMemoryLongTermStore:
    """进程内长期记忆：customer_id → {device_id → {faults, repairs}}。"""

    def __init__(self):
        self._data: dict[str, dict[str, dict]] = {}

    def add_fault(self, customer_id: str, device_id: str, fault: dict) -> None:
        d = self._data.setdefault(customer_id, {}).setdefault(device_id, {"faults": [], "repairs": []})
        d["faults"].append(fault)

    def add_repair(self, customer_id: str, device_id: str, repair: dict) -> None:
        d = self._data.setdefault(customer_id, {}).setdefault(device_id, {"faults": [], "repairs": []})
        d["repairs"].append(repair)

    def get_fault_history(self, customer_id: str) -> list:
        return [f for dev in self._data.get(customer_id, {}).values() for f in dev["faults"]]

    def get_repair_history(self, customer_id: str) -> list:
        return [r for dev in self._data.get(customer_id, {}).values() for r in dev["repairs"]]


class MemoryBackend:
    """
    长期记忆后端抽象。默认 InMemoryLongTermStore；
    生产可替换为 MongoDB / PostgreSQL（同接口）。
    """

    def __init__(self, business_service: Optional[BusinessService] = None, store=None):
        # 复用 Step 6 业务数据（历史故障/维修/工单）作为 long-term 关联
        self._biz = business_service or BusinessService()
        self._store = store or InMemoryLongTermStore()

    # ---- 结构化关联（customer / device 维度，严格按 customer 隔离） ----

    def get_device_list(self, customer_id: str) -> list:
        """该 customer 名下设备（隔离：不返回其他 customer 的设备）。"""
        return [d.model_dump() for d in self._biz.store.devices if d.customer_id == customer_id]

    def get_fault_history(self, customer_id: str) -> list:
        """该 customer 历史故障（跨其名下所有设备）。"""
        out = list(self._store.get_fault_history(customer_id))
        # 业务库里的维修记录也是故障历史
        for rec in self._biz.store.repairs:
            dev = self._biz._find_device(rec.device_id)
            if dev and dev.customer_id == customer_id:
                out.append({
                    "record_id": rec.record_id,
                    "device_id": rec.device_id,
                    "fault_description": rec.fault_description,
                    "repair_action": rec.repair_action,
                    "repaired_at": rec.repaired_at,
                })
        return out

    def get_repair_history(self, customer_id: str) -> list:
        """该 customer 历史维修记录。"""
        out = list(self._store.get_repair_history(customer_id))
        for rec in self._biz.store.repairs:
            dev = self._biz._find_device(rec.device_id)
            if dev and dev.customer_id == customer_id:
                out.append({
                    "record_id": rec.record_id,
                    "device_id": rec.device_id,
                    "fault_description": rec.fault_description,
                    "repair_action": rec.repair_action,
                    "repaired_at": rec.repaired_at,
                })
        return out

    def get_tickets(self, customer_id: str) -> list:
        """该 customer 工单（隔离）。"""
        return [t.model_dump() for t in self._biz.store.tickets if t.customer_id == customer_id]

    def record_fault(self, customer_id: str, device_id: str, fault: dict) -> None:
        self._store.add_fault(customer_id, device_id, fault)

    def record_repair(self, customer_id: str, device_id: str, repair: dict) -> None:
        self._store.add_repair(customer_id, device_id, repair)

    def record_ticket(self, ticket: dict) -> None:
        """工单落库（经 BusinessService，保持单一写入口）。"""
        self._biz.create_service_ticket(
            customer_id=ticket.get("customer_id", ""),
            device_id=ticket.get("device_id"),
            problem=ticket.get("problem", ""),
            diagnosis=ticket.get("diagnosis"),
            evidence=ticket.get("evidence"),
            priority=ticket.get("priority", "low"),
            idempotency_key=ticket.get("idempotency_key"),
        )


# ---------------- 门面 ----------------

class MemoryService:
    """短期 + 长期记忆统一门面（可注入到 Graph 节点）。"""

    def __init__(
        self,
        short: Optional[ShortTermMemory] = None,
        backend: Optional[MemoryBackend] = None,
        ttl_seconds: float = 30 * 60,
    ):
        self.short = short or ShortTermMemory(ttl_seconds=ttl_seconds)
        self.long = backend or MemoryBackend()

    # ---- short-term API ----
    def short_get(self, thread_id: str) -> Optional[ShortTermSlot]:
        return self.short.get(thread_id)

    def short_update(self, thread_id: str, **fields) -> ShortTermSlot:
        return self.short.update(thread_id, **fields)

    def short_record_tool(self, thread_id: str, tool_name: str, args: dict, result: dict) -> None:
        self.short.record_tool_call(thread_id, tool_name, args, result)

    # ---- long-term API（严格 customer 隔离） ----
    def long_devices(self, customer_id: str) -> list:
        return self.long.get_device_list(customer_id)

    def long_faults(self, customer_id: str) -> list:
        return self.long.get_fault_history(customer_id)

    def long_repairs(self, customer_id: str) -> list:
        return self.long.get_repair_history(customer_id)

    def long_tickets(self, customer_id: str) -> list:
        return self.long.get_tickets(customer_id)

    # ---- 组装"带记忆的 state 片段"（供 Graph 节点写回） ----
    def build_memory_context(
        self, thread_id: str, customer_id: Optional[str] = None, device_id: Optional[str] = None
    ) -> dict:
        """
        返回可放入 state 的 memory 上下文：
          {
            "short_term": {...},       # 本 thread 已有中间态（首次为空）
            "long_term": {             # customer/device 维度历史
              "customer_id": ...,
              "device_id": ...,
              "devices": [...],
              "faults": [...],
              "repairs": [...],
              "tickets": [...],
            }
          }
        """
        short = self.short.get(thread_id)
        long_ctx = {
            "customer_id": customer_id,
            "device_id": device_id,
            "devices": self.long_devices(customer_id) if customer_id else [],
            "faults": self.long_faults(customer_id) if customer_id else [],
            "repairs": self.long_repairs(customer_id) if customer_id else [],
            "tickets": self.long_tickets(customer_id) if customer_id else [],
        }
        # 记住 thread 关联的 customer/device（数据隔离锚点）
        if short is not None:
            short.update(customer_id=customer_id, device_id=device_id)
            self.short.put(thread_id, short)
        return {"short_term": short or {}, "long_term": long_ctx}


# ---------------- 全局单例 ----------------
import threading

_memory_lock = threading.Lock()
_shared_memory: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    global _shared_memory
    if _shared_memory is None:
        with _memory_lock:
            if _shared_memory is None:
                _shared_memory = MemoryService()
    return _shared_memory


__all__ = [
    "ShortTermMemory",
    "ShortTermSlot",
    "InMemoryLongTermStore",
    "MemoryBackend",
    "MemoryService",
    "get_memory_service",
]
