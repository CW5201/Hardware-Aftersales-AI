"""
web/api/v2_service.py

v2 Agent API（独立命名空间 /api/v2，与 v1 路由完全隔离）。

Step 5：
    POST /api/v2/agent/triage   （Step 2 保留不变）
    POST /api/v2/agent/run      （Step 4 新增，Step 5 扩展：Triage → Retrieval → Diagnosis）

挂载方式（见 query_service.py）：
    from web.api.v2_service import v2_router
    app.include_router(v2_router)
"""

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from processor.agent_processor.main_graph import AgentWorkflow
from processor.agent_processor.state import TriageResult
from tool.logger import logger

v2_router = APIRouter(prefix="/api/v2", tags=["v2-agent"])


class TriageRequest(BaseModel):
    query: str = Field(..., description="用户自然语言报修 / 提问")


@v2_router.post("/agent/triage", response_model=TriageResult)
async def agent_triage(request: TriageRequest):
    """
    v2 Triage：把用户问题结构化，返回 TriageResult。
    本阶段只到 Triage，不触发检索 / Tool / 诊断。
    """
    logger.info(f"[v2] Triage 请求: {request.query}")
    try:
        workflow = AgentWorkflow()
        state = workflow.run(request.query)
        return state.get("triage", TriageResult(intent="unknown"))
    except Exception as e:
        logger.exception(f"[v2] Triage 执行异常: {e}")
        # 保守降级：返回 unknown 意图，不让整条接口 500
        return TriageResult(
            intent="unknown",
            retrieval_strategy="hybrid_rrf_rerank",
            urgency="low",
        )


class AgentRunRequest(BaseModel):
    query: str = Field(..., description="用户自然语言报修 / 提问")
    thread_id: str = Field(default="default", description="会话线程 ID（短期记忆锚点）")
    customer_id: str = Field(default="", description="客户 ID（数据隔离 + 长期记忆）")
    device_id: str = Field(default="", description="设备 ID（长期记忆 + 业务 Tool 作用域）")


class AgentRunResponse(BaseModel):
    """
    返回 Agent 工作流的中间产物（Step 7：Triage + Memory + Retrieval + Diagnosis）。
    不包含 ticket / approval（Step 8/9 追加）。
    """
    triage: dict
    retrieval: dict
    diagnosis: dict
    memory: dict
    thread_id: str


@v2_router.post("/agent/run", response_model=AgentRunResponse)
async def agent_run(request: AgentRunRequest):
    """
    跑完整 v2 Agent Graph（当前：Triage → Memory → Retrieval → Diagnosis → Memory → END）。
    返回 triage / memory / retrieval evidence / diagnosis，不返回最终对客答案。
    """
    logger.info(f"[v2] Agent run 请求: {request.query} (thread={request.thread_id})")
    try:
        workflow = AgentWorkflow()
        state = workflow.run(
            request.query,
            thread_id=request.thread_id or "default",
            customer_id=request.customer_id or None,
            device_id=request.device_id or None,
        )
        triage: TriageResult = state.get("triage", TriageResult(intent="unknown"))
        retrieval = state.get("retrieval") or {}
        diagnosis = state.get("diagnosis")
        memory = state.get("memory") or {}
        return AgentRunResponse(
            triage=triage.model_dump(),
            retrieval=retrieval,
            diagnosis=diagnosis.model_dump() if diagnosis is not None else {},
            memory=memory,
            thread_id=state.get("thread_id") or request.thread_id or "default",
        )
    except Exception as e:
        logger.exception(f"[v2] Agent run 异常: {e}")
        # 保守降级：返回 unknown + 空 evidence + 空 diagnosis，不让接口 500
        return AgentRunResponse(
            triage=TriageResult(intent="unknown", retrieval_strategy="hybrid_rrf_rerank").model_dump(),
            retrieval={"documents": [], "scores": [], "strategy": None,
                       "metadata": {}, "latency": 0.0,
                       "error": f"agent_run_failed: {type(e).__name__}: {e}"},
            diagnosis={},
            memory={},
            thread_id=request.thread_id or "default",
        )
