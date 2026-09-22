import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', redirect: '/dashboard' },
  { path: '/dashboard', name: 'dashboard', component: () => import('../pages/Dashboard.vue') },
  { path: '/chat', name: 'chat', component: () => import('../pages/Chat.vue') },
  { path: '/devices', name: 'devices', component: () => import('../pages/Devices.vue') },
  { path: '/tickets', name: 'tickets', component: () => import('../pages/Tickets.vue') },
  { path: '/approvals', name: 'approvals', component: () => import('../pages/Approvals.vue') },
  { path: '/trace', name: 'trace', component: () => import('../pages/Trace.vue') },
  { path: '/evaluation', name: 'evaluation', component: () => import('../pages/Evaluation.vue') },
  { path: '/knowledge', name: 'knowledge', component: () => import('../pages/KnowledgeBase.vue') },
]

export default createRouter({
  history: createWebHistory(),
  routes,
})
