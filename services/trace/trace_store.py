"""
services/trace/trace_store.py

Agent Trace 存储（Step 10）：记录 run 级 + 节点级事件，支持按 thread/run 查询。

TraceRun（一次完整 run）：
    run_id / thread_id / user_query / created_at / status /
    total_latency_ms / token_usage {input, output, total} / error

TraceEvent（run 内一条节点 / 工具记录）：
    event_id / run_id / seq / node / agent / kind /
    input / output / tool_name / tool_args / tool_result /
    retrieved_docs_count / latency_ms / token_usage / error / status

设计：
  - InMemoryTraceStore：进程内 dict，append-only，按 run_id / thread_id 索引
  - trace_run / trace_event 是 Graph 节点调用（processor/agent_processor/
    traced_graph.py）的记录入口；API 查询走本 store
  - 不写真实 DB（本地模拟；生产可替换为 PostgreSQL / MongoDB）
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from tool.logger import logger


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def estimate_tokens(text: Optional[str]) -> int:
    """
    粗估 token 数（本地无 tokenizer 依赖时用字符数 / 4 近似）。
    真实 token 数由 LLM 响应 usage_metadata 提供（有则优先）。
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


class TokenUsage:
    """{input, output, total}。None 表示未知（未接真实 LLM usage 时）。"""

    def __init__(self, input_tokens: Optional[int] = None, output_tokens: Optional[int] = None):
        self.input = input_tokens
        self.output = output_tokens
        self.total = (
            None
            if (input_tokens is None or output_tokens is None)
            else input_tokens + output_tokens
        )

    def to_dict(self) -> dict:
        return {"input": self.input, "output": self.output, "total": self.total}

    @classmethod
    def from_usage_metadata(cls, usage: Optional[dict]) -> "TokenUsage":
        """从 LLM 响应的 usage_metadata 解析（langchain 标准字段）。"""
        if not usage:
            return cls()
        return cls(
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )


class TraceEvent:
    """单条节点 / 工具事件（有序，seq 单调增）。"""

    def __init__(
        self,
        run_id: str,
        seq: int,
        node: str,
        agent: str = "agent",
        kind: str = "node",
        input=None,
        output=None,
        tool_name: Optional[str] = None,
        tool_args: Optional[dict] = None,
        tool_result: Optional[dict] = None,
        retrieved_docs_count: Optional[int] = None,
        latency_ms: Optional[float] = None,
        token_usage: Optional[TokenUsage] = None,
        error: Optional[str] = None,
        status: str = "success",
    ):
        self.event_id = f"EVT-{uuid.uuid4().hex[:8]}"
        self.run_id = run_id
        self.seq = seq
        self.node = node
        self.agent = agent
        self.kind = kind  # node / tool / llm / approval
        self.input = input
        self.output = output
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.tool_result = tool_result
        self.retrieved_docs_count = retrieved_docs_count
        self.latency_ms = latency_ms
        self.token_usage = token_usage or TokenUsage()
        self.error = error
        self.status = status
        self.occurred_at = _now_iso()

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "seq": self.seq,
            "node": self.node,
            "agent": self.agent,
            "kind": self.kind,
            "input": self.input,
            "output": self.output,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tool_result": self.tool_result,
            "retrieved_docs_count": self.retrieved_docs_count,
            "latency_ms": self.latency_ms,
            "token_usage": self.token_usage.to_dict(),
            "error": self.error,
            "status": self.status,
            "occurred_at": self.occurred_at,
        }


class TraceRun:
    """一次完整 run 的摘要 + 事件序列。"""

    def __init__(self, run_id: str, thread_id: str, user_query: str):
        self.run_id = run_id
        self.thread_id = thread_id
        self.user_query = user_query
        self.created_at = _now_iso()
        self.status = "running"
        self.total_latency_ms: Optional[float] = None
        self.token_usage = TokenUsage()
        self.error: Optional[str] = None
        self.events: list[TraceEvent] = []
        self._started_at = time.time()
        self._seq = 0

    # ---------------- 记录 ----------------

    def add_event(self, event: TraceEvent) -> None:
        self._seq += 1
        event.seq = self._seq
        self.events.append(event)

    def record_node(self, node: str, input=None, output=None, **kw) -> None:
        self.add_event(TraceEvent(self.run_id, 0, node=node, kind="node", input=input, output=output, **kw))

    def record_tool(
        self,
        tool_name: str,
        tool_args: dict,
        tool_result: dict,
        retrieved_docs_count: Optional[int] = None,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        self.add_event(
            TraceEvent(
                self.run_id, 0,
                node="tool", agent="tool", kind="tool",
                tool_name=tool_name, tool_args=tool_args, tool_result=tool_result,
                retrieved_docs_count=retrieved_docs_count, latency_ms=latency_ms,
                error=error,
                status="error" if error else "success",
            )
        )

    def finish(self, status: str = "success", error: Optional[str] = None) -> None:
        self.total_latency_ms = round((time.time() - self._started_at) * 1000, 2)
        self.status = status
        self.error = error

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "user_query": self.user_query,
            "created_at": self.created_at,
            "status": self.status,
            "total_latency_ms": self.total_latency_ms,
            "token_usage": self.token_usage.to_dict(),
            "error": self.error,
            "events": [e.to_dict() for e in self.events],
        }


class InMemoryTraceStore:
    """进程内 append-only trace 存储（thread / run 双索引）。"""

    def __init__(self):
        self._runs: dict[str, TraceRun] = {}
        self._threads: dict[str, list[str]] = {}  # thread_id -> [run_id]

    def start_run(self, run_id: str, thread_id: str, user_query: str) -> TraceRun:
        run = TraceRun(run_id, thread_id, user_query)
        self._runs[run_id] = run
        self._threads.setdefault(thread_id, []).append(run_id)
        logger.info(f"TraceStore: start run={run_id} thread={thread_id}")
        return run

    def get_run(self, run_id: str) -> Optional[TraceRun]:
        return self._runs.get(run_id)

    def get_thread_runs(self, thread_id: str) -> list[TraceRun]:
        ids = self._threads.get(thread_id, [])
        return [self._runs[i] for i in ids if i in self._runs]

    def add_event(self, run_id: str, event: TraceEvent) -> None:
        run = self._runs.get(run_id)
        if run is not None:
            run.add_event(event)


# ---------------- 全局单例 ----------------

import threading

_trace_lock = threading.Lock()
_shared_trace: Optional[InMemoryTraceStore] = None


def get_trace_store() -> InMemoryTraceStore:
    global _shared_trace
    if _shared_trace is None:
        with _trace_lock:
            if _shared_trace is None:
                _shared_trace = InMemoryTraceStore()
    return _shared_trace


__all__ = [
    "InMemoryTraceStore",
    "TraceEvent",
    "TraceRun",
    "TokenUsage",
    "estimate_tokens",
    "get_trace_store",
]
