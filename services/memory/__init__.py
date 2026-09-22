"""
services/memory/__init__.py

v2 Memory 层（Step 7）：

  Short-term Memory（进程内，per thread_id）
    - current query / triage / retrieval / diagnosis / tool state
    - TTL 清理，防无界增长
    - 严格 thread 隔离

  Long-term Memory（持久化，customer / device 维度）
    - historical faults / repair history / tickets
    - 严格数据隔离：customer A 的 memory 查询绝不返回 customer B 的数据

设计约束（Step 7 要求）：
  - 不引入复杂向量记忆系统；关联 = 按 customer_id / device_id 索引的
    结构化记录（复用 Step 6 业务数据）。
"""

from services.memory.memory_service import (
    MemoryBackend,
    InMemoryLongTermStore,
    MemoryService,
    ShortTermMemory,
    ShortTermSlot,
    get_memory_service,
)

__all__ = [
    "MemoryBackend",
    "InMemoryLongTermStore",
    "MemoryService",
    "ShortTermMemory",
    "ShortTermSlot",
    "get_memory_service",
]
