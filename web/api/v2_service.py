"""
web/api/v2_service.py

v2 Agent API（独立命名空间 /api/v2，与 v1 路由完全隔离）。

本阶段（Step 2）只暴露 Triage 一个端点：
    POST /api/v2/agent/triage

后续阶段在同一 Router 上追加：
    /api/v2/agent/run、/threads/{id}/trace、/approvals/{id}/approve ...

挂载方式（见 query_service.py）：
    from web.api.v2_service import v2_router
    app.include_router(v2_router)
"""

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
