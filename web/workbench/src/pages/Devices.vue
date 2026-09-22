<template>
  <div class="page">
    <h2>Devices</h2>
    <div class="sub">v2 设备台账（演示 seed 数据，非生产）</div>

    <el-table :data="devices" v-loading="loading" border>
      <el-table-column prop="device_id" label="设备 ID" width="150" />
      <el-table-column prop="model" label="型号" width="120" />
      <el-table-column prop="customer_id" label="客户" width="120" />
      <el-table-column prop="firmware_version" label="固件" width="100" />
      <el-table-column prop="status" label="状态" width="120">
        <template #default="{ row }">
          <el-tag :type="statusType(row.status)" size="small">{{ row.status }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column label="质保" width="160">
        <template #default="{ row }">
          <el-tag :type="row.warranty_until ? 'success' : 'info'" size="small">
            {{ row.warranty_until ? 'in warranty' : 'no warranty' }}
          </el-tag>
        </template>
      </el-table-column>
    </el-table>

    <el-alert
      v-if="!loading && devices.length === 0"
      title="暂无设备数据（/api/v2/devices 未实现或后端未起）"
      type="info"
      :closable="false"
      style="margin-top: 12px"
    />
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { apiListDevices } from '../api'

const devices = ref<any[]>([])
const loading = ref(false)

function statusType(s: string) {
  if (s === 'active') return 'success'
  if (s === 'in_repair') return 'warning'
  if (s === 'offline') return 'info'
  return 'danger'
}

onMounted(async () => {
  loading.value = true
  try {
    devices.value = await apiListDevices()
  } finally {
    loading.value = false
  }
})
</script>
