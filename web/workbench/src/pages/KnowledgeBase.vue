<template>
  <div class="page">
    <h2>Knowledge Base</h2>
    <div class="sub">v1 RAG 知识库 · 混合检索 / RRF / Rerank / HyDE（本地演示数据）</div>

    <el-card>
      <template #header>最近检索证据（演示）</template>
      <el-table :data="docs" border size="small">
        <el-table-column prop="chunk_id" label="Chunk" width="110" />
        <el-table-column prop="item_name" label="商品" width="140" />
        <el-table-column prop="content" label="内容" min-width="260">
          <template #default="{ row }">
            <span class="mono">{{ row.content }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="score" label="Score" width="90">
          <template #default="{ row }">
            <el-progress :percentage="Math.round((row.score ?? 0) * 100)" :stroke-width="6" />
          </template>
        </el-table-column>
        <el-table-column prop="source" label="Source" width="90">
          <template #default="{ row }">
            <el-tag size="small" :type="row.source === 'web' ? 'warning' : 'info'">
              {{ row.source ?? 'local' }}
            </el-tag>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-card style="margin-top: 16px">
      <template #header>检索策略（SUPPORTED_STRATEGIES）</template>
      <el-space wrap>
        <el-tag v-for="s in strategies" :key="s" type="primary" effect="dark">
          {{ s }}
        </el-tag>
      </el-space>
      <p class="hint">
        Adaptive：简单 → hybrid · 复杂 → hybrid_rrf_rerank ·
        低置信/知识不足 → hyde_hybrid_rrf_rerank（+ MCP WebSearch 兜底）
      </p>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const strategies = ['hybrid', 'hybrid_rrf', 'hybrid_rrf_rerank', 'hyde_hybrid_rrf_rerank']

// 演示数据（与 RetrievalService seed 对齐；真实证据由 /api/v2/agent/run 的 retrieval 字段提供）
const docs = ref([
  {
    chunk_id: 'c1',
    item_name: 'X200',
    content: 'X200 电源模块规格：输入 220V/50Hz，输出 12V/15A，故障码 ERR-203 表示供电电压异常。',
    score: 0.91,
    source: 'local',
  },
  {
    chunk_id: 'c2',
    item_name: 'X200',
    content: 'ERR-203 处置流程：断电 → 更换电源模块 → 上电复测。建议备件 SP-PSU-X200。',
    score: 0.87,
    source: 'local',
  },
  {
    chunk_id: 'w1',
    item_name: 'H3CLA2608',
    content: '无线控制器升级注意：5.2.0 固件前需备份配置，否则 ERR-501。',
    score: 0.62,
    source: 'web',
  },
])
</script>

<style scoped>
.mono { font-family: Consolas, monospace; font-size: 12px; }
.hint { margin-top: 12px; color: #6b7385; font-size: 12px; }
</style>
