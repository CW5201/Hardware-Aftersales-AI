<template>
  <div class="page">
    <h2>Chat</h2>
    <div class="sub">v2 Agent 对话 · 展示 Triage → Memory → Retrieval → Diagnosis → Tool → Approval → Ticket 全链路</div>

    <el-row :gutter="16">
      <el-col :span="14">
        <el-card>
          <template #header>对话</template>
          <div class="msgs">
            <div v-for="(m, i) in messages" :key="i" class="msg" :class="m.role">
              <div class="msg-role">{{ m.role === 'user' ? '用户' : 'Agent' }}</div>
              <div class="msg-body">
                <div v-if="m.text" class="msg-text">{{ m.text }}</div>
                <el-tag v-if="m.triage" size="small" type="info" style="margin-right: 6px">
                  intent={{ m.triage.intent }}
                </el-tag>
                <el-tag v-if="m.triage?.product_model" size="small" type="primary">
                  {{ m.triage.product_model }}
                </el-tag>
                <el-tag v-if="m.diagnosis" size="small" :type="diagType(m.diagnosis.diagnosis_status)">
                  {{ m.diagnosis.diagnosis_status }}
                </el-tag>
              </div>
              <div v-if="m.triage || m.diagnosis || m.retrieval" class="msg-meta">
                <div v-if="m.triage" class="mm-row">
                  <span class="mm-k">Triage</span>
                  <span>{{ m.triage.intent }} · strategy={{ m.triage.retrieval_strategy }}</span>
                </div>
                <div v-if="m.retrieval" class="mm-row">
                  <span class="mm-k">Retrieval</span>
                  <span>{{ m.retrieval.documents?.length ?? 0 }} docs ·
                    {{ m.retrieval.adaptive?.strategy ?? m.retrieval.strategy }}</span>
                </div>
                <div v-if="m.diagnosis" class="mm-row">
                  <span class="mm-k">Diagnosis</span>
                  <span>{{ m.diagnosis.diagnosis_result || m.diagnosis.diagnosis_status }}</span>
                </div>
              </div>
            </div>
          </div>

          <div class="input-row">
            <el-input
              v-model="query"
              placeholder="输入报修 / 提问，如：我的 X200 出现 ERR-203"
              :disabled="loading"
              @keyup.enter="send"
            />
            <el-button type="primary" :loading="loading" @click="send">发送</el-button>
          </div>

          <div class="ctrl-row">
            <el-input-number v-model="topK" :min="1" :max="20" size="small" />
            <span style="margin: 0 8px; color: #6b7385">top_k</span>
            <el-switch v-model="showTrace" active-text="Trace" />
          </div>
        </el-card>

        <el-card v-if="showTrace && currentRun" class="trace-card">
          <template #header>本次 run Trace（{{ currentRun.run_id }}）</template>
          <el-timeline>
            <el-timeline-item
              v-for="ev in currentRun.events"
              :key="ev.event_id"
              :type="ev.error ? 'danger' : 'primary'"
              :timestamp="`#${ev.seq}`"
            >
              <b>{{ ev.node }}</b>
              <span class="tl-lat">· {{ ev.latency_ms }} ms</span>
              <div v-if="ev.error" class="tl-err">{{ ev.error }}</div>
              <div v-if="ev.tool_name" class="tl-tool">
                tool={{ ev.tool_name }} args={{ JSON.stringify(ev.tool_args) }}
              </div>
            </el-timeline-item>
          </el-timeline>
        </el-card>
      </el-col>

      <el-col :span="10">
        <el-card>
          <template #header>Agent 生命周期</template>
          <AgentLifecycle :steps="steps" :state="lcState" :active-index="lcActive" />
        </el-card>

        <el-card class="state-card">
          <template #header>当前 State</template>
          <el-descriptions v-if="lastResult" :column="1" border size="small">
            <el-descriptions-item label="thread_id">{{ lastResult.thread_id }}</el-descriptions-item>
            <el-descriptions-item label="diagnosis">
              {{ lastResult.diagnosis?.diagnosis_result || lastResult.diagnosis?.diagnosis_status }}
            </el-descriptions-item>
            <el-descriptions-item label="need_human_review">
              {{ lastResult.diagnosis?.need_human_review ? '是' : '否' }}
            </el-descriptions-item>
            <el-descriptions-item label="evidence">
              {{ (lastResult.retrieval?.documents?.length ?? 0) }} 条
            </el-descriptions-item>
            <el-descriptions-item label="long_term devices">
              {{ Object.keys(lastResult.memory?.long_term?.devices ?? {}).length || '—' }}
            </el-descriptions-item>
          </el-descriptions>
          <el-empty v-else description="暂无 run" :image-size="60" />
        </el-card>

        <el-card class="ev-card">
          <template #header>Evidence（检索证据）</template>
          <div v-if="lastResult?.retrieval?.documents?.length">
            <div
              v-for="(d, i) in lastResult.retrieval.documents.slice(0, 5)"
              :key="i"
              class="ev-item"
            >
              <el-tag size="small" type="info">{{ d.source ?? 'local' }}</el-tag>
              <el-tag size="small" type="warning">score={{ d.score }}</el-tag>
              <div class="ev-text">{{ d.content }}</div>
            </div>
          </div>
          <el-empty v-else description="无证据" :image-size="50" />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { computed, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import AgentLifecycle from '../components/AgentLifecycle.vue'
import { apiAgentRun } from '../api'
import type { AgentRunResponse, TraceRun } from '../types'

const steps = [
  { key: 'triage', label: 'Triage' },
  { key: 'memory', label: 'Memory' },
  { key: 'retrieval', label: 'Retrieval' },
  { key: 'diagnosis', label: 'Diagnosis' },
  { key: 'tool', label: 'Tool' },
  { key: 'approval', label: 'Approval' },
  { key: 'ticket', label: 'Ticket' },
]

interface Msg {
  role: 'user' | 'agent'
  text?: string
  triage?: any
  diagnosis?: any
  retrieval?: any
}

const messages = reactive<Msg[]>([])
const query = ref('')
const loading = ref(false)
const showTrace = ref(false)
const topK = ref(5)
const lastResult = ref<AgentRunResponse | null>(null)
const currentRun = ref<TraceRun | null>(null)
const lcActive = ref(0)

function diagType(s: string) {
  if (s === 'diagnosed') return 'success'
  if (s === 'retrieval_failed') return 'danger'
  return 'warning'
}

const lcState = computed(() => {
  const r = lastResult.value
  return {
    triage: { headline: r?.triage?.intent, status: r?.triage?.intent ? 'done' : 'wait' },
    memory: {
      headline: r?.memory?.long_term?.customer_id ?? '—',
      status: r?.memory ? 'done' : 'wait',
    },
    retrieval: {
      headline: `${r?.retrieval?.documents?.length ?? 0} docs`,
      error: !!r?.retrieval?.error,
      status: r?.retrieval ? 'done' : 'wait',
    },
    diagnosis: {
      headline: r?.diagnosis?.diagnosis_status,
      status: r?.diagnosis ? 'done' : 'wait',
    },
    tool: { headline: '—', status: 'wait' },
    approval: { headline: '—', status: 'wait' },
    ticket: { headline: '—', status: 'wait' },
  } as Record<string, { headline?: string; error?: boolean; status?: string }>
})

async function send() {
  const q = query.value.trim()
  if (!q || loading.value) return
  messages.push({ role: 'user', text: q })
  query.value = ''
  loading.value = true
  lcActive.value = 0
  try {
    const res = await apiAgentRun({ query: q, customer_id: 'CUST-0001' })
    lastResult.value = res
    lcActive.value = 3
    messages.push({
      role: 'agent',
      text: res.diagnosis?.diagnosis_result || res.diagnosis?.diagnosis_status,
      triage: res.triage,
      diagnosis: res.diagnosis,
      retrieval: res.retrieval,
    })
    // 拉取本次 run 的 trace（若有 trace_run_id 可从 state 拿；此处演示用 thread）
    if (showTrace.value) {
      const threadId = res.thread_id
      const t = await fetchThreadTrace(threadId)
      currentRun.value = t?.runs?.[t.runs.length - 1] ?? null
    }
  } catch (e: any) {
    ElMessage.error(`Agent run 失败: ${e?.message ?? e}`)
    lcState.value.retrieval = { headline: 'error', error: true, status: 'error' }
  } finally {
    loading.value = false
  }
}

async function fetchThreadTrace(thread_id: string) {
  try {
    const { apiThreadTrace } = await import('../api')
    return await apiThreadTrace(thread_id)
  } catch {
    return null
  }
}
</script>

<style scoped>
.msgs { max-height: 380px; overflow: auto; display: flex; flex-direction: column; gap: 10px; margin-bottom: 12px; }
.msg { border: 1px solid #eef1f8; border-radius: 10px; padding: 10px 12px; }
.msg.user { background: #f0f6ff; }
.msg.agent { background: #fff; }
.msg-role { font-size: 11px; color: #8b93a7; margin-bottom: 4px; }
.msg-text { font-size: 14px; margin-bottom: 6px; }
.msg-meta { margin-top: 8px; border-top: 1px dashed #e5e8f0; padding-top: 6px; }
.mm-row { display: flex; gap: 8px; font-size: 12px; margin-bottom: 3px; }
.mm-k { color: #8b93a7; min-width: 64px; }
.input-row { display: flex; gap: 8px; }
.ctrl-row { margin-top: 10px; display: flex; align-items: center; }
.trace-card, .state-card, .ev-card { margin-top: 16px; }
.ev-item { margin-bottom: 10px; }
.ev-text { margin-top: 4px; font-size: 13px; color: #333a4d; white-space: pre-wrap; }
.tl-lat { color: #8b93a7; font-size: 12px; margin-left: 6px; }
.tl-err { color: #e5484d; font-size: 12px; }
.tl-tool { color: #6b7385; font-size: 12px; }
</style>
