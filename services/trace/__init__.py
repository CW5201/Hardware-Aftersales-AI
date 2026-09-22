"""
services/trace/__init__.py

v2 Agent Trace 层（Step 10）：
    记录每次 Agent run 的完整生命周期（节点 / 工具 / LLM / 错误 / 耗时 / token），
    提供 GET /api/v2/threads/{thread_id}/trace 与 GET /api/v2/runs/{run_id} 查询。

设计约束：
  - 进程内 InMemory TraceStore（本地模拟；可换 PostgreSQL / MongoDB，同接口）
  - 一次 run 一条 TraceRun；run 内每个节点 / 工具调用一条 TraceEvent（有序）
  - 不接真实存储，不影响 v1
"""

from services.trace.trace_store import (
    InMemoryTraceStore,
    TraceEvent,
    TraceRun,
    get_trace_store,
    TokenUsage,
    estimate_tokens,
)

__all__ = [
    "InMemoryTraceStore",
    "TraceEvent",
    "TraceRun",
    "TokenUsage",
    "get_trace_store",
    "estimate_tokens",
]
