"""
web/api/v2_service.py

v2 Agent API（独立命名空间 /api/v2，与 v1 路由完全隔离）。

Step 8：
    POST /api/v2/agent/triage   （Step 2 保留不变）
    POST /api/v2/agent/run      （Step 4/5/7 扩展：完整 Agent Graph）
    Ticket Workflow（Step 8 新增）：
      POST /api/v2/tickets                          创建工单（WRITE，Step 9 起走 HITL）
      GET  /api/v2/tickets?customer_id=&status=     工单列表
      GET  /api/v2/tickets/{ticket_id}              工单详情 + 事件流
      POST /api/v2/tickets/{ticket_id}/transition   状态迁移（校验合法）

挂载方式（见 query_service.py）：
    from web.api.v2_service import v2_router
    app.include_router(v2_router)
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from processor.agent_processor.main_graph import AgentWorkflow
from processor.agent_processor.state import TriageResult
from tool.logger import logger
from services.business.business_service import BusinessService, BusinessServiceError
from services.business.ticket_workflow import TicketWorkflow, TicketWorkflowError
from services.business.approval_service import ApprovalService, ApprovalError

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


# ================= Step 8: Ticket Workflow API =================

def _ticket_workflow() -> TicketWorkflow:
    """获取全局共享的 TicketWorkflow（单例，用默认 BusinessService 演示 seed）。"""
    return TicketWorkflow()


class CreateTicketRequest(BaseModel):
    customer_id: str = Field(..., description="客户 ID")
    device_id: str = Field(default="", description="设备 ID（可选）")
    problem: str = Field(..., description="问题描述")
    diagnosis: str = Field(default="", description="诊断结论（可选）")
    evidence: list = Field(default_factory=list, description="证据摘要列表")
    priority: str = Field(default="low", description="low / medium / high")
    idempotency_key: str = Field(default="", description="幂等键（防 retry 重复建单）")


class CreateTicketResponse(BaseModel):
    created: bool
    deduplicated: bool
    ticket: dict


@v2_router.post("/tickets", response_model=CreateTicketResponse)
async def create_ticket(request: CreateTicketRequest):
    """
    创建服务工单（WRITE）。
    Step 9 起，高风险写操作会被 Policy + HITL 拦截（需审批后 resume）。
    本阶段（Step 8）直接执行，幂等键防重复。
    """
    svc = TicketWorkflow()._svc
    try:
        res = svc.create_service_ticket(
            customer_id=request.customer_id,
            device_id=request.device_id or None,
            problem=request.problem,
            diagnosis=request.diagnosis or None,
            evidence=request.evidence,
            priority=request.priority,
            idempotency_key=request.idempotency_key or None,
        )
        return CreateTicketResponse(
            created=res["created"],
            deduplicated=res.get("deduplicated", False),
            ticket=res["ticket"],
        )
    except BusinessServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))


@v2_router.get("/tickets")
async def list_tickets(customer_id: str = "", status: str = ""):
    """工单列表（按 customer / status 过滤）。"""
    wf = _ticket_workflow()
    try:
        return wf.list_tickets(customer_id=customer_id or None, status=status or None)
    except TicketWorkflowError as e:
        raise HTTPException(status_code=400, detail=str(e))


class TicketDetailResponse(BaseModel):
    found: bool
    ticket: dict
    events: list


@v2_router.get("/tickets/{ticket_id}", response_model=TicketDetailResponse)
async def get_ticket(ticket_id: str):
    """工单详情 + 事件流（ticket_events 回放）。"""
    wf = _ticket_workflow()
    res = wf.get_ticket(ticket_id)
    if not res["found"]:
        raise HTTPException(status_code=404, detail=f"ticket not found: {ticket_id}")
    events = wf.get_events(ticket_id)["events"]
    return TicketDetailResponse(found=True, ticket=res["ticket"], events=events)


class TransitionRequest(BaseModel):
    status: str = Field(..., description="目标状态：PENDING/IN_PROGRESS/WAITING_APPROVAL/COMPLETED/CLOSED")
    reason: str = Field(default="", description="迁移原因（记入事件）")


class TransitionResponse(BaseModel):
    ticket_id: str
    from_status: str
    to_status: str
    reason: str
    events: list


@v2_router.post("/tickets/{ticket_id}/transition", response_model=TransitionResponse)
async def transition_ticket(ticket_id: str, request: TransitionRequest):
    """工单状态迁移（校验合法迁移 + 写事件）。非法迁移 → 400。"""
    wf = _ticket_workflow()
    try:
        res = wf.transition(ticket_id, request.status, reason=request.reason)
        return TransitionResponse(
            ticket_id=ticket_id,
            from_status=res["from"],
            to_status=res["to"],
            reason=res["reason"],
            events=res["events"],
        )
    except TicketWorkflowError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ================= Step 9: Human-in-the-loop 审批 API =================

_approval_service: Optional[ApprovalService] = None


def _get_approval_service() -> ApprovalService:
    """全局共享 ApprovalService（复用 TicketWorkflow 的 BusinessService 单例）。"""
    global _approval_service
    if _approval_service is None:
        _approval_service = ApprovalService(_ticket_workflow()._svc)
    return _approval_service


class ApprovalDetailResponse(BaseModel):
    found: bool
    approval: dict
    error: Optional[str] = None


@v2_router.get("/approvals/{approval_id}", response_model=ApprovalDetailResponse)
async def get_approval(approval_id: str):
    """查审批记录（审批页展示：设备 / 客户 / 问题 / 诊断 / 证据 / Tool / 参数）。"""
    res = _get_approval_service().get(approval_id)
    if not res["found"]:
        raise HTTPException(status_code=404, detail=f"approval not found: {approval_id}")
    return ApprovalDetailResponse(found=True, approval=res["approval"])


class DecideResponse(BaseModel):
    approval_id: str
    status: str
    decided_by: str
    decided_at: str


@v2_router.post("/approvals/{approval_id}/approve", response_model=DecideResponse)
async def approve(approval_id: str, decided_by: str = "human"):
    """批准 → 状态转 approved（执行由 resume 时的 execute 幂等完成）。"""
    try:
        a = _get_approval_service().approve(approval_id, decided_by=decided_by)
        return DecideResponse(
            approval_id=approval_id, status=a["status"],
            decided_by=a["decided_by"], decided_at=a["decided_at"],
        )
    except ApprovalError as e:
        raise HTTPException(status_code=400, detail=str(e))


@v2_router.post("/approvals/{approval_id}/reject", response_model=DecideResponse)
async def reject(approval_id: str, decided_by: str = "human"):
    """驳回 → 状态转 rejected（resume 后不执行写操作）。"""
    try:
        a = _get_approval_service().reject(approval_id, decided_by=decided_by)
        return DecideResponse(
            approval_id=approval_id, status=a["status"],
            decided_by=a["decided_by"], decided_at=a["decided_at"],
        )
    except ApprovalError as e:
        raise HTTPException(status_code=400, detail=str(e))
