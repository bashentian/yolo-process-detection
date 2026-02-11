<template>
  <div class="settings-section">
    <div class="settings-section-header">
      <div class="settings-section-title">
        <i class="bi bi-hdd"></i>
        系统维护
      </div>
    </div>
    <div class="settings-section-body">
      <div class="switch-container">
        <div class="switch-info">
          <div class="switch-label">自动重启</div>
          <div class="switch-description">系统崩溃时自动重启服务</div>
        </div>
        <label class="switch">
          <input v-model="settings.system.autoRestart" type="checkbox">
          <span class="switch-slider"></span>
        </label>
      </div>

      <div class="form-row">
        <div class="form-group">
          <label class="form-label">日志级别</label>
          <select v-model="settings.system.logLevel" class="form-select">
            <option value="DEBUG">DEBUG</option>
            <option value="INFO">INFO</option>
            <option value="WARNING">WARNING</option>
            <option value="ERROR">ERROR</option>
            <option value="CRITICAL">CRITICAL</option>
          </select>
        </div>

        <div class="form-group">
          <label class="form-label">最大日志大小 (MB)</label>
          <input
            v-model.number="settings.system.maxLogSize"
            type="number"
            class="form-input"
            min="10"
            max="1000"
          >
        </div>
      </div>

      <div class="switch-container">
        <div class="switch-info">
          <div class="switch-label">启用备份</div>
          <div class="switch-description">定期备份系统配置和数据</div>
        </div>
        <label class="switch">
          <input v-model="settings.system.backupEnabled" type="checkbox">
          <span class="switch-slider"></span>
        </label>
      </div>

      <div class="form-group" v-if="settings.system.backupEnabled">
        <label class="form-label">备份间隔 (小时)</label>
        <input
          v-model.number="settings.system.backupInterval"
          type="number"
          class="form-input"
          min="1"
          max="168"
        >
        <div class="form-help">自动备份的时间间隔，单位为小时</div>
      </div>

      <div class="maintenance-actions">
        <div class="action-group">
          <h4 class="action-title">数据管理</h4>
          <div class="action-buttons">
            <button class="btn btn-secondary" @click="exportData">
              <i class="bi bi-download"></i>
              导出数据
            </button>
            <button class="btn btn-secondary" @click="importData">
              <i class="bi bi-upload"></i>
              导入数据
            </button>
            <button class="btn btn-danger" @click="clearCache">
              <i class="bi bi-trash"></i>
              清除缓存
            </button>
          </div>
        </div>

        <div class="action-group">
          <h4 class="action-title">系统操作</h4>
          <div class="action-buttons">
            <button class="btn btn-secondary" @click="restartService">
              <i class="bi bi-arrow-clockwise"></i>
              重启服务
            </button>
            <button class="btn btn-danger" @click="factoryReset">
              <i class="bi bi-exclamation-triangle"></i>
              恢复出厂设置
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useSettingsStore } from '@/stores/settings'
import { useAppStore } from '@/stores/app'

const settingsStore = useSettingsStore()
const appStore = useAppStore()

const settings = computed(() => settingsStore.settings)

function exportData() {
  appStore.addNotification({
    type: 'success',
    title: '导出成功',
    message: '数据已成功导出'
  })
}

function importData() {
  appStore.addNotification({
    type: 'info',
    title: '导入数据',
    message: '请选择要导入的数据文件'
  })
}

function clearCache() {
  if (confirm('确定要清除所有缓存吗？')) {
    appStore.addNotification({
      type: 'success',
      title: '清除成功',
      message: '缓存已清除'
    })
  }
}

function restartService() {
  if (confirm('确定要重启服务吗？')) {
    appStore.addNotification({
      type: 'info',
      title: '重启中',
      message: '服务正在重启...'
    })
  }
}

function factoryReset() {
  if (confirm('警告：此操作将删除所有数据并恢复出厂设置，确定要继续吗？')) {
    settingsStore.resetSettings()
    appStore.addNotification({
      type: 'success',
      title: '重置成功',
      message: '系统已恢复出厂设置'
    })
  }
}
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

.maintenance-actions {
  margin-top: var(--space-xl);
  display: flex;
  flex-direction: column;
  gap: var(--space-xl);
}

.action-group {
  background: var(--bg-tertiary);
  border-radius: var(--radius-md);
  padding: var(--space-lg);
}

.action-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: var(--space-md);
}

.action-buttons {
  display: flex;
  gap: var(--space-md);
  flex-wrap: wrap;
}

@media (max-width: 768px) {
  .form-row {
    grid-template-columns: 1fr;
  }

  .action-buttons {
    flex-direction: column;
  }

  .action-buttons .btn {
    width: 100%;
  }
}
</style>
