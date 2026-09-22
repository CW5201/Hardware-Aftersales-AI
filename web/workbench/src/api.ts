// web/workbench/src/api.ts
// 统一 API 客户端（axios + SSE）。v2 走 /api/v2，v1 页面走 /api/v1。
import axios from 'axios'
import type {
  AgentRunResponse, Approval, Diagnosis, EvalMetrics,
  Retrieval, Triage, Ticket, TicketEvent, TraceRun,
} from './types'

const http = axios.create({ baseURL: '/', timeout: 60000 })

export async function apiAgentRun(req: {
  query: string; thread_id?: string; customer_id?: string; device_id?: string
}): Promise<AgentRunResponse> {
  const { data } = await http.post('/api/v2/agent/run', req)
  return data
}

export async function apiAgentTriage(query: string): Promise<Triage> {
  const { data } = await http.post('/api/v2/agent/triage', { query })
  return data
}

// ---------- 工单 ----------
export async function apiCreateTicket(req: {
  customer_id: string; device_id?: string; problem: string; diagnosis?: string;
  evidence?: any[]; priority?: string; idempotency_key?: string
}): Promise<{ created: boolean; deduplicated: boolean; ticket: Ticket }> {
  const { data } = await http.post('/api/v2/tickets', req)
  return data
}

export async function apiListTickets(customer_id = '', status = ''): Promise<{
  count: number; tickets: Ticket[]; error: string | null
}> {
  const { data } = await http.get('/api/v2/tickets', {
    params: { customer_id, status },
  })
  return data
}

export async function apiGetTicket(ticket_id: string): Promise<{
  found: boolean; ticket: Ticket; events: TicketEvent[]
}> {
  const { data } = await http.get(`/api/v2/tickets/${ticket_id}`)
  return data
}

export async function apiTransitionTicket(
  ticket_id: string, status: string, reason = '',
): Promise<{ ticket_id: string; from_status: string; to_status: string; reason: string; events: TicketEvent[] }> {
  const { data } = await http.post(`/api/v2/tickets/${ticket_id}/transition`, { status, reason })
  return data
}

// ---------- 审批（HITL） ----------
export async function apiGetApproval(approval_id: string): Promise<Approval> {
  const { data } = await http.get(`/api/v2/approvals/${approval_id}`)
  return data.approval
}

export async function apiApprove(approval_id: string, decided_by = 'human'): Promise<Approval> {
  const { data } = await http.post(`/api/v2/approvals/${approval_id}/approve`, null, { params: { decided_by } })
  return data
}

export async function apiReject(approval_id: string, decided_by = 'human'): Promise<Approval> {
  const { data } = await http.post(`/api/v2/approvals/${approval_id}/reject`, null, { params: { decided_by } })
  return data
}

// ---------- Trace ----------
export async function apiThreadTrace(thread_id: string): Promise<{ thread_id: string; runs: TraceRun[] }> {
  const { data } = await http.get(`/api/v2/threads/${thread_id}/trace`)
  return data
}

export async function apiRunTrace(run_id: string): Promise<{ found: boolean; run: TraceRun }> {
  const { data } = await http.get(`/api/v2/runs/${run_id}`)
  return data
}

// ---------- 设备（v2 数据源；后端未实现该路由时页面降级显示空列表） ----------
export async function apiListDevices(): Promise<any[]> {
  // 工作台演示数据源（与后端 BusinessService seed 对齐）。
  // /api/v2/devices 在当前版本未实现 → 返回 []，页面显示"无数据"而非报错。
  try {
    const { data } = await http.get('/api/v2/devices')
    return data.devices ?? data ?? []
  } catch {
    return []
  }
}

// ---------- SSE（v1 保留的流式） ----------
export function openSSE(session_id: string, onEvent: (ev: { event: string; data: any }) => void): () => void {
  const es = new EventSource(`/sse?session_id=${encodeURIComponent(session_id)}`)
  const handler = (e: MessageEvent) => {
    try {
      onEvent({ event: (e as any).type ?? 'message', data: JSON.parse(e.data) })
    } catch {
      onEvent({ event: 'raw', data: e.data })
    }
  }
  es.onmessage = handler
  es.addEventListener('triage', handler)
  es.addEventListener('retrieval', handler)
  es.addEventListener('diagnosis', handler)
  es.addEventListener('done', handler)
  return () => es.close()
}

// ---------- Evaluation（Step 11 指标，走后端聚合接口；没有则 Not evaluated） ----------
export async function apiAgentEval(): Promise<EvalMetrics> {
  const { data } = await http.get('/api/v2/eval/agent')
  return data
}

export default {
  http,
  apiAgentRun,
  apiAgentTriage,
  apiCreateTicket,
  apiListTickets,
  apiGetTicket,
  apiTransitionTicket,
  apiGetApproval,
  apiApprove,
  apiReject,
  apiThreadTrace,
  apiRunTrace,
  apiListDevices,
  openSSE,
  apiAgentEval,
}
