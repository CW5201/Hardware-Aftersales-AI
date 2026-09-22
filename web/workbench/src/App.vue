<template>
  <el-container class="workbench">
    <el-aside width="220px" class="sidebar">
      <div class="brand">
        <h1>After-sales Agent</h1>
        <span class="ver">v2.0</span>
      </div>
      <el-menu :default-active="route.path" router>
        <el-menu-item index="/dashboard">
          <el-icon><DataLine /></el-icon><span>Dashboard</span>
        </el-menu-item>
        <el-menu-item index="/chat">
          <el-icon><ChatDotRound /></el-icon><span>Chat</span>
        </el-menu-item>
        <el-menu-item index="/devices">
          <el-icon><Cpu /></el-icon><span>Devices</span>
        </el-menu-item>
        <el-menu-item index="/tickets">
          <el-icon><Tickets /></el-icon><span>Tickets</span>
        </el-menu-item>
        <el-menu-item index="/approvals">
          <el-icon><Stamp /></el-icon><span>Approvals</span>
        </el-menu-item>
        <el-menu-item index="/trace">
          <el-icon><Promotion /></el-icon><span>Trace</span>
        </el-menu-item>
        <el-menu-item index="/evaluation">
          <el-icon><Histogram /></el-icon><span>Evaluation</span>
        </el-menu-item>
        <el-menu-item index="/knowledge">
          <el-icon><Collection /></el-icon><span>Knowledge Base</span>
        </el-menu-item>
      </el-menu>
      <div class="side-foot">
        <el-tag size="small" type="info">v1 RAG · v2 Agentic</el-tag>
      </div>
    </el-aside>
    <el-main class="main">
      <router-view />
    </el-main>
  </el-container>
</template>

<script setup lang="ts">
import { useRoute } from 'vue-router'
import {
  DataLine, ChatDotRound, Cpu, Tickets, Stamp,
  Promotion, Histogram, Collection,
} from '@element-plus/icons-vue'
const route = useRoute()
</script>

<style>
html, body, #app { height: 100%; margin: 0; font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; }
.workbench { height: 100vh; }
.sidebar { background: #1f2430; color: #fff; }
.brand { padding: 18px 20px; display: flex; align-items: baseline; gap: 8px; border-bottom: 1px solid #333a4d; }
.brand h1 { font-size: 16px; margin: 0; font-weight: 600; }
.brand .ver { font-size: 11px; color: #8b93a7; }
.sidebar .el-menu { border-right: none; background: transparent; }
.sidebar .el-menu-item { color: #aab0c5; }
.sidebar .el-menu-item.is-active { background: #2b3245; color: #fff; }
.sidebar .el-menu-item:hover { background: #262d3e; color: #fff; }
.side-foot { position: absolute; bottom: 16px; left: 20px; }
.main { background: #f5f7fb; padding: 20px 28px; overflow: auto; }
.page { max-width: 1180px; }
.page h2 { margin: 0 0 4px; }
.page .sub { color: #6b7385; font-size: 13px; margin-bottom: 20px; }
</style>
