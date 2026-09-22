import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  server: {
    port: 5173,
    proxy: {
      // v2 Agent API 与 v1 API 代理到 FastAPI 后端
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
      '/sse': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
