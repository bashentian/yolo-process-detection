<template>
  <div class="settings-section">
    <div class="settings-section-header">
      <div class="settings-section-title">
        <i class="bi bi-bell"></i>
        通知设置
      </div>
    </div>
    <div class="settings-section-body">
      <div class="switch-container">
        <div class="switch-info">
          <div class="switch-label">启用邮件通知</div>
          <div class="switch-description">通过邮件发送检测警报和系统通知</div>
        </div>
        <label class="switch">
          <input v-model="settings.notification.enableEmail" type="checkbox">
          <span class="switch-slider"></span>
        </label>
      </div>

      <div class="form-group" v-if="settings.notification.enableEmail">
        <label class="form-label">收件人邮箱</label>
        <input
          v-model="settings.notification.emailRecipient"
          type="email"
          class="form-input"
          placeholder="请输入收件人邮箱地址"
        >
        <div class="form-help">多个邮箱地址用逗号分隔</div>
      </div>

      <div class="switch-container">
        <div class="switch-info">
          <div class="switch-label">启用Webhook通知</div>
          <div class="switch-description">通过HTTP POST发送通知到指定URL</div>
        </div>
        <label class="switch">
          <input v-model="settings.notification.enableWebhook" type="checkbox">
          <span class="switch-slider"></span>
        </label>
      </div>

      <div class="form-group" v-if="settings.notification.enableWebhook">
        <label class="form-label">Webhook URL</label>
        <input
          v-model="settings.notification.webhookUrl"
          type="url"
          class="form-input"
          placeholder="请输入Webhook URL"
        >
        <div class="form-help">接收通知的HTTP端点URL</div>
      </div>

      <div class="form-group">
        <div class="slider-control">
          <div class="slider-header">
            <span class="slider-label-text">警报阈值</span>
            <span class="slider-value">{{ (settings.notification.alertThreshold * 100).toFixed(0) }}%</span>
          </div>
          <input
            v-model.number="settings.notification.alertThreshold"
            type="range"
            min="0"
            max="1"
            step="0.05"
          >
          <div class="form-help">触发警报的最小置信度阈值</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useSettingsStore } from '@/stores/settings'

const settingsStore = useSettingsStore()

const settings = computed(() => settingsStore.settings)
</script>

<style scoped>
.settings-section {
  background: var(--bg-secondary);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  overflow: hidden;
}

.settings-section-header {
  padding: var(--space-lg);
  border-bottom: 1px solid var(--border-primary);
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.settings-section-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
  display: flex;
  align-items: center;
  gap: var(--space-sm);
}

.settings-section-body {
  padding: var(--space-lg);
}

.form-group {
  margin-bottom: var(--space-lg);
}

.form-group:last-child {
  margin-bottom: 0;
}

.form-label {
  display: block;
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary);
  margin-bottom: var(--space-sm);
}

.form-help {
  font-size: 12px;
  color: var(--text-tertiary);
  margin-top: var(--space-xs);
}

.form-input,
.form-select,
.form-textarea {
  width: 100%;
  padding: var(--space-sm) var(--space-md);
  background: var(--bg-tertiary);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  color: var(--text-primary);
  font-size: 14px;
  transition: all var(--transition-fast);
}

.form-input:focus,
.form-select:focus,
.form-textarea:focus {
  outline: none;
  border-color: var(--hk-primary);
  box-shadow: 0 0 0 3px var(--hk-primary-200);
}

.form-input::placeholder,
.form-textarea::placeholder {
  color: var(--text-tertiary);
}

.form-textarea {
  min-height: 100px;
  resize: vertical;
}

.form-row {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: var(--space-lg);
}

.switch-container {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-md) 0;
  border-bottom: 1px solid var(--border-primary);
}

.switch-container:last-child {
  border-bottom: none;
}

.switch-info {
  flex: 1;
}

.switch-label {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary);
  margin-bottom: 4px;
}

.switch-description {
  font-size: 12px;
  color: var(--text-secondary);
}

.switch {
  position: relative;
  width: 48px;
  height: 24px;
  margin-left: var(--space-md);
}

.switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.switch-slider {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: var(--bg-tertiary);
  border: 1px solid var(--border-primary);
  border-radius: 24px;
  transition: all var(--transition-fast);
}

.switch-slider:before {
  position: absolute;
  content: "";
  height: 18px;
  width: 18px;
  left: 2px;
  bottom: 2px;
  background: var(--text-secondary);
  border-radius: 50%;
  transition: all var(--transition-fast);
}

.switch input:checked + .switch-slider {
  background: var(--hk-primary);
  border-color: var(--hk-primary);
}

.switch input:checked + .switch-slider:before {
  transform: translateX(24px);
  background: white;
}

.slider-control {
  display: flex;
  flex-direction: column;
  gap: var(--space-sm);
}

.slider-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.slider-label-text {
  font-size: 14px;
  color: var(--text-primary);
}

.slider-value {
  font-size: 14px;
  font-weight: 600;
  color: var(--hk-primary);
  min-width: 50px;
  text-align: right;
}

input[type="range"] {
  width: 100%;
  height: 6px;
  background: var(--bg-tertiary);
  border-radius: 3px;
  outline: none;
  -webkit-appearance: none;
}

input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 18px;
  height: 18px;
  background: var(--hk-primary);
  border-radius: 50%;
  cursor: pointer;
  transition: all var(--transition-fast);
}

input[type="range"]::-webkit-slider-thumb:hover {
  box-shadow: 0 0 0 6px var(--hk-primary-200);
}

@media (max-width: 768px) {
  .form-row {
    grid-template-columns: 1fr;
  }
}
</style>
