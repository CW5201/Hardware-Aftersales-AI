<template>
  <div class="page">
    <h2>Dashboard</h2>
    <div class="sub">v2 Agent 生命周期总览 · 演示数据（非生产环境）</div>

    <el-row :gutter="16">
      <el-col :span="12">
        <el-card>
          <template #header>Agent 生命周期</template>
          <AgentLifecycle
            :steps="lifecycleSteps"
            :state="state"
            :active-index="activeIndex"
          />
          <el-progress
            :percentage="percent"
            :stroke-width="8"
            status="success"
            style="margin-top: 12px"
          />
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card>
          <template #header>运行状态（演示）</template>
          <el-statistic title="当前节点" :value="lifecycleSteps[activeIndex]?.label ?? '—'" />
          <el-statistic title="累计 run" :value="runs" style="margin-top: 16px" />
          <el-statistic title="平均延迟 (ms)" :value="avgLatency" style="margin-top: 16px" />
        </el-card>
      </el-col>
    </el-row>

    <el-card style="margin-top: 16px">
      <template #header>近 10 次 run 延迟分布</template>
      <div ref="chartRef" style="height: 260px" />
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import * as echarts from 'echarts'
import AgentLifecycle from '../components/AgentLifecycle.vue'

const lifecycleSteps = [
  { key: 'triage', label: 'Triage' },
  { key: 'memory', label: 'Memory' },
  { key: 'retrieval', label: 'Retrieval' },
  { key: 'diagnosis', label: 'Diagnosis' },
  { key: 'tool', label: 'Tool' },
  { key: 'approval', label: 'Approval' },
  { key: 'ticket', label: 'Ticket' },
]

const state = ref<Record<string, { headline?: string; error?: boolean }>>({
  triage: { headline: 'fault_diagnosis' },
  memory: { headline: 'CUST-0001' },
  retrieval: { headline: 'hybrid · 2 docs' },
  diagnosis: { headline: 'diagnosed' },
  tool: { headline: 'create_service_ticket' },
  approval: { headline: 'approved' },
  ticket: { headline: 'TICK-0001' },
})
const activeIndex = ref(3)
const percent = computed(() => Math.round(((activeIndex.value + 1) / lifecycleSteps.length) * 100))
const runs = ref(128)
const avgLatency = ref(842)

const chartRef = ref<HTMLDivElement>()
let chart: echarts.ECharts | null = null

onMounted(() => {
  chart = echarts.init(chartRef.value!)
  chart.setOption({
    grid: { left: 40, right: 16, top: 20, bottom: 30 },
    xAxis: {
      type: 'category',
      data: ['Triage', 'Memory', 'Retrieval', 'Diagnosis', 'Tool', 'Approval', 'Ticket'],
    },
    yAxis: { type: 'value', name: 'ms' },
    tooltip: { trigger: 'axis' },
    series: [
      {
        type: 'bar',
        data: [640, 120, 410, 720, 180, 520, 96],
        itemStyle: { color: '#4b8cff', borderRadius: [4, 4, 0, 0] },
      },
    ],
  })
  window.addEventListener('resize', resize)
})

function resize() {
  chart?.resize()
}
onBeforeUnmount(() => {
  window.removeEventListener('resize', resize)
  chart?.dispose()
})
</script>
