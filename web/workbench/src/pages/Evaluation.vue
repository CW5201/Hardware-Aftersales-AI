<template>
  <div class="page">
    <h2>Evaluation</h2>
    <div class="sub">v2 Agent 指标 · 无真实标注数据时明确 "Not evaluated"（不编数字）</div>

    <el-row :gutter="16">
      <el-col :span="12">
        <el-card>
          <template #header>Agent 级指标</template>
          <el-table :data="metricRows" border size="small">
            <el-table-column prop="metric" label="指标" width="240" />
            <el-table-column label="数值">
              <template #default="{ row }">
                <el-tag v-if="row.value === NOT_EVAL" type="info" size="small">Not evaluated</el-tag>
                <span v-else class="num">{{ formatVal(row.value) }}</span>
              </template>
            </el-table-column>
          </el-table>
          <el-alert
            v-if="allNotEvaluated"
            title="无带标注的 run 样本，所有指标 Not evaluated"
            type="info"
            :closable="false"
            style="margin-top: 12px"
          />
        </el-card>
      </el-col>

      <el-col :span="12">
        <el-card>
          <template #header>指标分布（演示，基于可用样本）</template>
          <div ref="chartRef" style="height: 300px" />
        </el-card>
      </el-col>
    </el-row>

    <el-card style="margin-top: 16px">
      <template #header>Failure Dataset</template>
      <el-descriptions :column="2" size="small">
        <el-descriptions-item label="内置样本数">{{ failure.total_samples }}</el-descriptions-item>
        <el-descriptions-item label="失败类别">{{ Object.keys(failure.failure_type_counts ?? {}).join(', ') }}</el-descriptions-item>
      </el-descriptions>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import * as echarts from 'echarts'

const NOT_EVAL = 'Not evaluated'

// 后端 /api/v2/eval/agent 未实现时的降级：全部 Not evaluated（不编数字）
const metrics = ref<Record<string, number | string>>({})

const metricRows = computed(() =>
  [
    'task_success_rate',
    'tool_selection_accuracy',
    'tool_argument_accuracy',
    'diagnosis_accuracy',
    'answer_groundedness',
    'citation_accuracy',
    'abstention_accuracy',
    'human_escalation_accuracy',
  ].map((m) => ({
    metric: m,
    value: metrics.value[m] ?? NOT_EVAL,
  }))
)

const allNotEvaluated = computed(
  () => metricRows.value.every((r) => r.value === NOT_EVAL)
)

function formatVal(v: number | string) {
  if (typeof v === 'number') return v.toFixed(3)
  return String(v)
}

// failure dataset（从后端 eval/agent_failures.json 统计；此处演示常量）
const failure = ref({
  total_samples: 9,
  failure_type_counts: {
    wrong_tool: 1, wrong_argument: 1, wrong_device: 1, missing_evidence: 1,
    hallucination: 1, memory_contamination: 1, unsafe_write: 1,
    wrong_escalation: 1, duplicate_ticket: 1,
  },
})

// 有数值指标才画分布图；全 Not evaluated 时显示空态
const chartRef = ref<HTMLDivElement>()
let chart: echarts.ECharts | null = null

onMounted(async () => {
  try {
    const { apiAgentEval } = await import('../api')
    const rep = await apiAgentEval()
    if (rep?.agent_metrics) {
      Object.assign(metrics.value, rep.agent_metrics)
    }
  } catch {
    // 后端未实现 / 无数据 → 保持全 Not evaluated
  }
  renderChart()
  window.addEventListener('resize', resize)
})

function renderChart() {
  const numeric = metricRows.value
    .filter((r) => typeof r.value === 'number')
    .map((r) => ({ name: r.metric, value: r.value as number }))
  chart = echarts.init(chartRef.value!)
  if (numeric.length === 0) {
    chart.setOption({
      title: { text: 'Not evaluated\n（无带标注样本）', left: 'center', top: 'center',
        textStyle: { fontSize: 14, color: '#8b93a7' } },
    })
    return
  }
  chart.setOption({
    grid: { left: 180, right: 40, top: 10, bottom: 20 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'value', min: 0, max: 1 },
    yAxis: { type: 'category', data: numeric.map((n) => n.name),
      axisLabel: { fontSize: 11 } },
    series: [{
      type: 'bar',
      data: numeric.map((n) => n.value),
      itemStyle: { color: '#4b8cff', borderRadius: [0, 4, 4, 0] },
      barWidth: 14,
      label: { show: true, position: 'right', formatter: (p: any) => p.value.toFixed(2) },
    }],
  })
}

function resize() {
  chart?.resize()
}
onBeforeUnmount(() => {
  window.removeEventListener('resize', resize)
  chart?.dispose()
})
</script>

<style scoped>
.num { font-family: Consolas, monospace; font-size: 14px; color: #1f2430; font-weight: 600; }
</style>
