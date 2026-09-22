<template>
  <div class="page">
    <h2>Tickets</h2>
    <div class="sub">v2 工单 · 状态机 PENDING → IN_PROGRESS → WAITING_APPROVAL → COMPLETED → CLOSED</div>

    <el-card>
      <div class="filter-row">
        <el-input v-model="customerFilter" placeholder="customer_id 过滤（可空）" style="width: 220px" clearable />
        <el-select v-model="statusFilter" placeholder="状态过滤" clearable style="width: 200px; margin-left: 8px">
          <el-option v-for="s in STATUS" :key="s" :label="s" :value="s" />
        </el-select>
        <el-button type="primary" @click="load" style="margin-left: 8px">查询</el-button>
        <el-button @click="createTicket">新建工单</el-button>
      </div>

      <el-table :data="tickets" v-loading="loading" border style="margin-top: 12px">
        <el-table-column prop="ticket_id" label="工单" width="110" />
        <el-table-column prop="customer_id" label="客户" width="120" />
        <el-table-column prop="device_id" label="设备" width="150" />
        <el-table-column prop="problem" label="问题" min-width="160" />
        <el-table-column prop="priority" label="优先级" width="90">
          <template #default="{ row }">
            <el-tag :type="priType(row.priority)" size="small">{{ row.priority }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="150">
          <template #default="{ row }">
            <el-tag :type="statusType(row.status)" size="small">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="200">
          <template #default="{ row }">
            <el-button size="small" @click="showEvents(row)">事件</el-button>
            <el-button size="small" type="primary" @click="advance(row)">推进状态</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 事件流 -->
    <el-dialog v-model="showEventDialog" title="工单事件流" width="560px">
      <el-timeline v-if="currentEvents.length">
        <el-timeline-item
          v-for="ev in currentEvents"
          :key="ev.event_id"
          :type="ev.event_type === 'created' ? 'primary' : 'success'"
        >
          <b>{{ ev.event_type }}</b>
          <span v-if="ev.detail" style="color: #6b7385"> · {{ ev.detail }}</span>
          <div class="ev-time">{{ ev.occurred_at }}</div>
        </el-timeline-item>
      </el-timeline>
      <el-empty v-else description="暂无事件" :image-size="60" />
    </el-dialog>

    <!-- 新建工单 -->
    <el-dialog v-model="showCreateDialog" title="新建工单" width="480px">
      <el-form label-width="90px">
        <el-form-item label="customer">
          <el-input v-model="newTicket.customer_id" placeholder="CUST-0001" />
        </el-form-item>
        <el-form-item label="device">
          <el-input v-model="newTicket.device_id" placeholder="DEV-X200-001（可空）" />
        </el-form-item>
        <el-form-item label="problem">
          <el-input v-model="newTicket.problem" type="textarea" :rows="2" />
        </el-form-item>
        <el-form-item label="priority">
          <el-select v-model="newTicket.priority" style="width: 100%">
            <el-option label="low" value="low" />
            <el-option label="medium" value="medium" />
            <el-option label="high" value="high" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showCreateDialog = false">取消</el-button>
        <el-button type="primary" @click="submitCreate">创建</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { apiListTickets, apiGetTicket, apiTransitionTicket, apiCreateTicket } from '../api'
import type { Ticket, TicketEvent } from '../types'

const STATUS = ['PENDING', 'IN_PROGRESS', 'WAITING_APPROVAL', 'COMPLETED', 'CLOSED']
const NEXT: Record<string, string> = {
  PENDING: 'IN_PROGRESS',
  IN_PROGRESS: 'WAITING_APPROVAL',
  WAITING_APPROVAL: 'COMPLETED',
  COMPLETED: 'CLOSED',
}

const tickets = ref<Ticket[]>([])
const loading = ref(false)
const customerFilter = ref('')
const statusFilter = ref('')

const showEventDialog = ref(false)
const currentEvents = ref<TicketEvent[]>([])
const currentTicket = ref<Ticket | null>(null)

const showCreateDialog = ref(false)
const newTicket = ref({ customer_id: 'CUST-0001', device_id: 'DEV-X200-001', problem: '', priority: 'medium' })

function priType(p: string) {
  if (p === 'high') return 'danger'
  if (p === 'medium') return 'warning'
  return 'info'
}
function statusType(s: string) {
  if (s === 'CLOSED') return 'info'
  if (s === 'COMPLETED') return 'success'
  if (s === 'WAITING_APPROVAL') return 'warning'
  return 'primary'
}

async function load() {
  loading.value = true
  try {
    const res = await apiListTickets(customerFilter.value, statusFilter.value)
    tickets.value = res.tickets
  } catch (e: any) {
    ElMessage.error(`加载工单失败: ${e?.message ?? e}`)
  } finally {
    loading.value = false
  }
}

async function showEvents(row: Ticket) {
  currentTicket.value = row
  try {
    const res = await apiGetTicket(row.ticket_id)
    currentEvents.value = res.events
    showEventDialog.value = true
  } catch (e: any) {
    ElMessage.error(`事件流加载失败: ${e?.message ?? e}`)
  }
}

async function advance(row: Ticket) {
  const next = NEXT[row.status]
  if (!next) {
    ElMessage.info('已是终态 CLOSED，无法推进')
    return
  }
  try {
    const res = await apiTransitionTicket(row.ticket_id, next)
    ElMessage.success(`${res.from_status} → ${res.to_status}`)
    await load()
  } catch (e: any) {
    ElMessage.error(`迁移失败: ${e?.response?.data?.detail ?? e?.message ?? e}`)
  }
}

function createTicket() {
  newTicket.value = { customer_id: 'CUST-0001', device_id: 'DEV-X200-001', problem: '', priority: 'medium' }
  showCreateDialog.value = true
}

async function submitCreate() {
  if (!newTicket.value.problem) {
    ElMessage.warning('请填写 problem')
    return
  }
  try {
    const res = await apiCreateTicket({
      customer_id: newTicket.value.customer_id,
      device_id: newTicket.value.device_id || undefined,
      problem: newTicket.value.problem,
      priority: newTicket.value.priority,
    })
    ElMessage.success(res.deduplicated ? '已去重（同客户+设备已有 OPEN 工单）' : '工单已创建')
    showCreateDialog.value = false
    await load()
  } catch (e: any) {
    ElMessage.error(`创建失败: ${e?.response?.data?.detail ?? e?.message ?? e}`)
  }
}

onMounted(load)
</script>

<style scoped>
.filter-row { display: flex; align-items: center; }
.ev-time { font-size: 11px; color: #8b93a7; }
</style>
