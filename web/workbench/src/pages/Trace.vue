<template>
  <div class="page">
    <h2>Trace</h2>
    <div class="sub">Agent 全链路 Trace · Triage → Memory → Retrieval → Diagnosis → Tool → Approval → Ticket</div>

    <el-card>
      <div class="filter-row">
        <el-input v-model="threadId" placeholder="thread_id（如 t1 / default）" style="width: 240px" clearable />
        <el-button type="primary" @click="loadThread" :loading="loading">查询 thread</el-button>
        <el-input v-model="runId" placeholder="run_id（可空，直接查单次）" style="width: 260px; margin-left: 12px" clearable />
        <el-button @click="loadRun">查询 run</el-button>
      </div>
    </el-card>

    <!-- 单次 run 全量事件 -->
    <template v-if="currentRun">
      <el-card class="run-card">
        <template #header>
          <span>Run {{ currentRun.run_id }}</span>
          <el-tag :type="runType(currentRun.status)" style="margin-left: 10px">{{ currentRun.status }}</el-tag>
          <span class="run-lat">总延迟 {{ currentRun.total_latency_ms }} ms</span>
          <span class="run-token">token
            <el-tag size="small" type="info">in={{ currentRun.token_usage.input }}</el-tag>
            <el-tag size="small" type="info">out={{ currentRun.token_usage.output }}</el-tag>
          </span>
        </template>

        <el-descriptions :column="2" border size="small" style="margin-bottom: 14px">
          <el-descriptions-item label="thread_id">{{ currentRun.thread_id }}</el-descriptions-item>
          <el-descriptions-item label="user_query">{{ currentRun.user_query }}</el-descriptions-item>
          <el-descriptions-item label="created_at">{{ currentRun.created_at }}</el-descriptions-item>
          <el-descriptions-item label="error">{{ currentRun.error || '—' }}</el-descriptions-item>
        </el-descriptions>

        <AgentLifecycle :steps="steps" :state="lc" :active-index="lcActive" />

        <el-table :data="currentRun.events" border size="small" style="margin-top: 14px">
          <el-table-column prop="seq" label="#" width="44" />
          <el-table-column prop="node" label="Node" width="140" />
          <el-table-column prop="agent" label="Agent" width="120" />
          <el-table-column label="Input" min-width="180">
            <template #default="{ row }">
              <span class="mono">{{ summarize(row.input) }}</span>
            </template>
          </el-table-column>
          <el-table-column label="Output" min-width="200">
            <template #default="{ row }">
              <span class="mono">{{ summarize(row.output) }}</span>
            </template>
          </el-table-column>
          <el-table-column label="Tool" min-width="180">
            <template #default="{ row }">
              <span v-if="row.tool_name" class="mono">
                {{ row.tool_name }}({{ summarize(row.tool_args) }})
              </span>
              <span v-else>—</span>
            </template>
          </el-table-column>
          <el-table-column label="Tool Result" min-width="160">
            <template #default="{ row }">
              <span class="mono">{{ summarize(row.tool_result) }}</span>
            </template>
          </el-table-column>
          <el-table-column label="Docs" width="70">
            <template #default="{ row }">
              <span v-if="row.retrieved_docs_count != null">{{ row.retrieved_docs_count }}</span>
              <span v-else>—</span>
            </template>
          </el-table-column>
          <el-table-column prop="latency_ms" label="Latency(ms)" width="110" />
          <el-table-column label="Tokens" width="120">
            <template #default="{ row }">
              <span class="mono">
                {{ row.token_usage.input != null ? row.token_usage.input : '?' }}
                /{{ row.token_usage.output != null ? row.token_usage.output : '?' }}
              </span>
            </template>
          </el-table-column>
          <el-table-column label="Status" width="90">
            <template #default="{ row }">
              <el-tag :type="row.error ? 'danger' : 'success'" size="small">{{ row.status }}</el-tag>
            </template>
          </el-table-column>
        </el-table>
      </el-card>
    </template>

    <!-- thread 下所有 run -->
    <el-card v-if="threadRuns.length" class="thread-card">
      <template #header>Thread {{ threadId }} 的历史 run（{{ threadRuns.length }}）</template>
      <el-table :data="threadRuns" border size="small">
        <el-table-column prop="run_id" label="run_id" width="180" />
        <el-table-column prop="status" label="状态" width="100">
          <template #default="{ row }">
            <el-tag :type="runType(row.status)" size="small">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="total_latency_ms" label="延迟(ms)" width="110" />
        <el-table-column prop="user_query" label="query" min-width="160" />
        <el-table-column label="事件数" width="80">
          <template #default="{ row }">{{ row.events?.length ?? 0 }}</template>
        </el-table-column>
        <el-table-column label="操作" width="90">
          <template #default="{ row }">
            <el-button size="small" @click="currentRun = row">查看</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-empty v-if="!currentRun && !threadRuns.length" description="查询 thread 或 run 查看 trace" />
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { ElMessage } from 'element-plus'
import AgentLifecycle from '../components/AgentLifecycle.vue'
import { apiThreadTrace, apiRunTrace } from '../api'
import type { TraceRun } from '../types'

const steps = [
  { key: 'triage', label: 'Triage' },
  { key: 'memory', label: 'Memory' },
  { key: 'retrieval', label: 'Retrieval' },
  { key: 'diagnosis', label: 'Diagnosis' },
  { key: 'tool', label: 'Tool' },
  { key: 'approval', label: 'Approval' },
  { key: 'ticket', label: 'Ticket' },
]

const threadId = ref('')
const runId = ref('')
const loading = ref(false)
const currentRun = ref<TraceRun | null>(null)
const threadRuns = ref<TraceRun[]>([])

function runType(s: string) {
  if (s === 'success') return 'success'
  if (s === 'error') return 'danger'
  return 'info'
}

function summarize(v: any): string {
  if (v == null) return '—'
  if (typeof v === 'object') {
    try {
      const s = JSON.stringify(v)
      return s.length > 80 ? s.slice(0, 77) + '…' : s
    } catch {
      return String(v)
    }
  }
  const s = String(v)
  return s.length > 80 ? s.slice(0, 77) + '…' : s
}

// 从事件里推导每个 lifecycle 步骤的状态（trace 只有 5 个节点事件，
// tool/approval/ticket 由 tool_name/approval 字段推导）
const lc = computed(() => {
  const run = currentRun.value
  if (!run) return {}
  const byNode: Record<string, any> = {}
  run.events.forEach((e) => (byNode[e.node] = e))
  const toolEvent = run.events.find((e) => e.kind === 'tool' || e.tool_name)
  const st = (e?: any) => ({
    headline: e ? (e.error ? `error: ${e.error}` : e.output?.diagnosis_status ?? e.output?.docs != null ? `${e.output.docs} docs` : e.node) : '—',
    error: !!e?.error,
  })
  return {
    triage: st(byNode.triage),
    memory: st(byNode.memory_retrieve ?? byNode.memory_persist),
    retrieval: st(byNode.retrieval),
    diagnosis: st(byNode.diagnosis),
    tool: toolEvent
      ? {
          headline: toolEvent.tool_name,
          error: !!toolEvent.error,
        }
      : undefined,
    approval: undefined,
    ticket: undefined,
  }
})

const lcActive = computed(() => {
  const run = currentRun.value
  if (!run) return 0
  const idx = (name: string) =>
    run.events.findIndex((e) => e.node === name)
  // 取已执行到的最远节点
  const seen = [idx('triage'), idx('memory_retrieve'), idx('retrieval'), idx('diagnosis')]
  const lastOk = Math.max(...seen.filter((i) => i >= 0), 0)
  return Math.min(lastOk + 1, steps.length - 1)
})

async function loadThread() {
  if (!threadId.value.trim()) return
  loading.value = true
  try {
    const res = await apiThreadTrace(threadId.value.trim())
    threadRuns.value = res.runs
    currentRun.value = res.runs?.length ? res.runs[res.runs.length - 1] : null
  } catch (e: any) {
    ElMessage.error(`trace 查询失败: ${e?.response?.data?.detail ?? e?.message ?? e}`)
  } finally {
    loading.value = false
  }
}

async function loadRun() {
  if (!runId.value.trim()) return
  loading.value = true
  try {
    const res = await apiRunTrace(runId.value.trim())
    currentRun.value = res.run
    threadRuns.value = []
  } catch (e: any) {
    ElMessage.error(`run 不存在: ${e?.response?.data?.detail ?? e?.message ?? e}`)
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.filter-row { display: flex; align-items: center; }
.run-card, .thread-card { margin-top: 16px; }
.run-lat, .run-token { margin-left: 12px; font-size: 12px; color: #6b7385; }
.mono { font-family: 'SFMono-Regular', Consolas, monospace; font-size: 12px; color: #4b5563; }
</style>
