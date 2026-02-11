<template>
  <div class="settings-section">
    <div class="settings-section-header">
      <div class="settings-section-title">
        <i class="bi bi-sliders"></i>
        常规设置
      </div>
    </div>
    <div class="settings-section-body">
      <div class="form-group">
        <label class="form-label">
          系统名称
          <span class="required">*</span>
        </label>
        <input
          v-model="settings.general.systemName"
          type="text"
          class="form-input"
          placeholder="请输入系统名称"
        >
        <div class="form-help">系统显示的名称，将显示在页面标题和侧边栏</div>
      </div>

      <div class="form-row">
        <div class="form-group">
          <label class="form-label">语言</label>
          <select v-model="settings.general.language" class="form-select">
            <option value="zh-CN">简体中文</option>
            <option value="zh-TW">繁體中文</option>
            <option value="en-US">English</option>
            <option value="ja-JP">日本語</option>
          </select>
        </div>

        <div class="form-group">
          <label class="form-label">时区</label>
          <select v-model="settings.general.timezone" class="form-select">
            <option value="Asia/Shanghai">Asia/Shanghai (UTC+8)</option>
            <option value="Asia/Tokyo">Asia/Tokyo (UTC+9)</option>
            <option value="America/New_York">America/New_York (UTC-5)</option>
            <option value="Europe/London">Europe/London (UTC+0)</option>
          </select>
        </div>
      </div>

      <div class="switch-container">
        <div class="switch-info">
          <div class="switch-label">自动保存</div>
          <div class="switch-description">设置更改后自动保存，无需手动点击保存按钮</div>
        </div>
        <label class="switch">
          <input v-model="settings.general.autoSave" type="checkbox">
          <span class="switch-slider"></span>
        </label>
      </div>

      <div class="switch-container">
        <div class="switch-info">
          <div class="switch-label">深色模式</div>
          <div class="switch-description">使用深色主题，减少眼睛疲劳</div>
        </div>
        <label class="switch">
          <input v-model="settings.general.darkMode" type="checkbox">
          <span class="switch-slider"></span>
        </label>
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

.form-label .required {
  color: var(--error);
  margin-left: 2px;
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

@media (max-width: 768px) {
  .form-row {
    grid-template-columns: 1fr;
  }
}
</style>
