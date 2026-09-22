// web/workbench/src/types.ts
// v2 Agent 工作台前端类型（与后端 Pydantic 模型对齐）

export interface Triage {
  intent: string
  product_model: string | null
  device_id: string | null
  error_code: string | null
  symptom: string | null
  urgency: string
  missing_information: string[]
  retrieval_strategy: string
}

export interface RetrievalDoc {
  chunk_id?: string
  content?: string
  source?: string
  score?: number
  [k: string]: any
}

export interface Retrieval {
  documents: RetrievalDoc[]
  scores: number[]
  strategy: string | null
  sources: string[]
  metadata: Record<string, any>
  latency: number
  error: string | null
}

export interface Hypothesis {
  text: string
  evidence_indices: number[]
  confidence: string
}

export interface Diagnosis {
  diagnosis_status: string
  hypotheses: Hypothesis[]
  evidence: string[]
  diagnosis_result: string | null
  recommended_actions: string[]
  need_more_information: string[]
  need_human_review: boolean
}

export interface Memory {
  short_term: Record<string, any>
  long_term: Record<string, any>
}

export interface AgentRunResponse {
  triage: Triage
  retrieval: Retrieval
  diagnosis: Diagnosis
  memory: Memory
  thread_id: string
}

export interface Ticket {
  ticket_id: string
  customer_id: string
  device_id: string | null
  problem: string
  diagnosis: string | null
  evidence: any[]
  priority: string
  status: string
  assigned_to: string | null
  created_at: string
  updated_at: string
}

export interface TicketEvent {
  event_id: string
  ticket_id: string
  event_type: string
  detail: string | null
  occurred_at: string
}

export interface Approval {
  approval_id: string
  tool_name: string
  tool_args: Record<string, any>
  requester: string | null
  idempotency_key: string | null
  status: string
  created_at: string
  decided_at: string | null
  decided_by: string | null
}

export interface TokenUsage {
  input: number | null
  output: number | null
  total: number | null
}

export interface TraceEvent {
  event_id: string
  run_id: string
  seq: number
  node: string
  agent: string
  kind: string
  input: any
  output: any
  tool_name: string | null
  tool_args: Record<string, any> | null
  tool_result: Record<string, any> | null
  retrieved_docs_count: number | null
  latency_ms: number | null
  token_usage: TokenUsage
  error: string | null
  status: string
  occurred_at: string
}

export interface TraceRun {
  run_id: string
  thread_id: string
  user_query: string
  created_at: string
  status: string
  total_latency_ms: number | null
  token_usage: TokenUsage
  error: string | null
  events: TraceEvent[]
}

export interface EvalMetrics {
  [metric: string]: number | string
}

// 工作台各页通用
