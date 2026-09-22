<template>
  <div class="lifecycle">
    <template v-for="(step, i) in steps" :key="step.key">
      <div
        class="lc-step"
        :class="{ active: i === activeIndex, done: i < activeIndex, error: i === activeIndex && stepState(step.key)?.error, wait: i > activeIndex }"
      >
        <div class="lc-bubble">
          <span v-if="i < activeIndex">✓</span>
          <span v-else>{{ i + 1 }}</span>
        </div>
        <div class="lc-label">{{ step.label }}</div>
        <div v-if="i === activeIndex && stepState(step.key)?.headline" class="lc-detail">
          {{ stepState(step.key)?.headline }}
        </div>
      </div>
      <div v-if="i < steps.length - 1" class="lc-link" :class="{ filled: i < activeIndex }"></div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  steps: { key: string; label: string }[]
  state: Record<string, { headline?: string; error?: boolean; status?: string }>
  activeIndex: number
}>()

function stepState(key: string) {
  return props.state?.[key]
}
</script>

<style scoped>
.lifecycle {
  display: flex;
  align-items: flex-start;
  padding: 14px 0;
  overflow-x: auto;
  position: relative;
}
.lc-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  min-width: 80px;
  flex-shrink: 0;
  position: relative;
  z-index: 1;
}
.lc-bubble {
  width: 30px;
  height: 30px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 600;
  background: #e6e9f2;
  color: #8b93a7;
  border: 2px solid transparent;
  transition: all 0.25s;
}
.lc-label {
  margin-top: 6px;
  font-size: 12px;
  color: #6b7385;
  text-align: center;
  white-space: nowrap;
}
.lc-detail {
  margin-top: 3px;
  font-size: 11px;
  color: #4b8cff;
  max-width: 110px;
  text-align: center;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.lc-link {
  position: relative;
  top: 14px;
  flex: 1;
  height: 2px;
  min-width: 16px;
  background: #dfe3ee;
  z-index: 0;
}
.lc-link.filled { background: #2ea96b; }

.lc-step.active .lc-bubble {
  background: #4b8cff;
  color: #fff;
  box-shadow: 0 0 0 4px rgba(75, 140, 255, 0.18);
}
.lc-step.active .lc-label { color: #4b8cff; font-weight: 600; }
.lc-step.done .lc-bubble { background: #2ea96b; color: #fff; }
.lc-step.done .lc-label { color: #2ea96b; }
.lc-step.error .lc-bubble { background: #e5484d; color: #fff; }
</style>
