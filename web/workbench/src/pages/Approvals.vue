<template>
  <div class="page">
    <h2>Approvals</h2>
    <div class="sub">Human-in-the-loop 审批 · 设备 / 客户 / 问题 / Diagnosis / Evidence / Tool / 参数</div>

    <el-row :gutter="16">
      <el-col :span="8">
        <el-card>
          <template #header>审批 ID</template>
          <el-input
            v-model="approvalId"
            placeholder="APR-YYYYMMDDHHMMSS-NNN"
            clearable
          />
          <el-button type="primary" @click="load" :loading="loading" style="width: 100%; margin-top: 10px">
            加载审批
          </el-button>
          <el-button @click="demo" style="width: 100%; margin-top: 8px">演示：新建一条审批</el-button>
        </el-card>
      </el-col>

      <el-col :span="16">
        <el-card v-if="approval">
          <template #header>
            <span>审批 {{ approval.approval_id }}</span>
            <el-tag :type="statusType(approval.status)" style="margin-left: 10px">
              {{ approval.status }}
            </el-tag>
          </template>

          <el-descriptions :column="2" border>
            <el-descriptions-item label="Tool">{{ approval.tool_name }}</el-descriptions-item>
            <el-descriptions-item label="状态">{{ approval.status }}</el-descriptions-item>
            <el-descriptions-item label="幂等键">{{ approval.idempotency_key || '—' }}</el-descriptions-item>
            <el-descriptions-item label="requester">{{ approval.requester || '—' }}</el-descriptions-item>
            <el-descriptions-item label="创建时间">{{ approval.created_at }}</el-descriptions-item>
            <el-descriptions-item label="决议人">{{ approval.decided_by || '—' }}</el-descriptions-item>
          </el-descriptions>

          <div class="section">Tool 参数</div>
          <pre class="code">{{ JSON.stringify(approval.tool_args, null, 2) }}</pre>

          <div class="actions">
            <el-button
              type="success"
              :disabled="approval.status !== 'pending'"
              @click="decide(true)"
            >批准</el-button>
            <el-button
              type="danger"
              :disabled="approval.status !== 'pending'"
              @click="decide(false)"
            >驳回</el-button>
          </div>
        </el-card>
        <el-empty v-else description="输入审批 ID 或点击「演示」" />
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { apiGetApproval, apiApprove, apiReject } from '../api'
import type { Approval } from '../types'

const approvalId = ref('')
const approval = ref<Approval | null>(null)
const loading = ref(false)

function statusType(s: string) {
  if (s === 'approved') return 'success'
  if (s === 'rejected') return 'danger'
  if (s === 'executed') return 'primary'
  return 'warning'
}

async function load() {
  const id = approvalId.value.trim()
  if (!id) {
    ElMessage.warning('请输入审批 ID')
    return
  }
  loading.value = true
  try {
    approval.value = await apiGetApproval(id)
  } catch (e: any) {
    ElMessage.error(`审批不存在: ${e?.response?.data?.detail ?? e?.message ?? e}`)
    approval.value = null
  } finally {
    loading.value = false
  }
}

async function decide(approve: boolean) {
  if (!approval.value) return
  try {
    if (approve) {
      await apiApprove(approval.value.approval_id, 'workbench-user')
    } else {
      await apiReject(approval.value.approval_id, 'workbench-user')
    }
    ElMessage.success(approve ? '已批准（resume 时幂等执行）' : '已驳回（resume 时不执行写操作）')
    approval.value = await apiGetApproval(approval.value.approval_id)
  } catch (e: any) {
    ElMessage.error(`决策失败: ${e?.response?.data?.detail ?? e?.message ?? e}`)
  }
}

// 演示：走 HITL graph 创建一条真实审批（需后端起）
async function demo() {
  try {
    // 直接触发 agent run + write tool 会创建审批；此处演示直接查最近一条
    const resp = await fetch('/api/v2/approvals/recent')
    if (resp.ok) {
      const data = await resp.json()
      approvalId.value = data.approvals?.[0]?.approval_id ?? ''
      await load()
    } else {
      ElMessage.info('后端 /api/v2/approvals/recent 未实现，请手动输入审批 ID')
    }
  } catch {
    ElMessage.info('请手动输入审批 ID（演示数据需后端起）')
  }
}
</script>

<style scoped>
.section { font-weight: 600; margin: 14px 0 6px; }
.code {
  background: #f7f8fc;
  border-radius: 8px;
  padding: 10px;
  font-size: 12px;
  overflow: auto;
  max-height: 200px;
}
.actions { margin-top: 16px; display: flex; gap: 10px; }
</style>
