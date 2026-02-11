<template>
  <div class="settings">
    <div class="page-header">
      <div>
        <h1 class="page-title">系统设置</h1>
        <p class="page-subtitle">配置系统参数和偏好设置</p>
      </div>
      <div class="page-actions">
        <button class="btn btn-secondary" @click="resetAllSettings">
          <i class="bi bi-arrow-counterclockwise"></i>
          重置
        </button>
        <button class="btn btn-primary" @click="saveAllSettings" :disabled="settingsStore.saving">
          <i class="bi bi-check-lg"></i>
          {{ settingsStore.saving ? '保存中...' : '保存设置' }}
        </button>
      </div>
    </div>

    <div class="settings-container">
      <aside class="settings-sidebar">
        <a
          v-for="section in sections"
          :key="section.id"
          class="settings-nav-item"
          :class="{ active: activeSection === section.id }"
          @click="activeSection = section.id"
        >
          <i :class="section.icon"></i>
          <span>{{ section.label }}</span>
        </a>
      </aside>

      <div class="settings-content">
        <GeneralSettings v-if="activeSection === 'general'" />
        <ModelSettings v-if="activeSection === 'model'" />
        <DetectionSettings v-if="activeSection === 'detection'" />
        <CameraSettings v-if="activeSection === 'camera'" />
        <NotificationSettings v-if="activeSection === 'notification'" />
        <SystemSettings v-if="activeSection === 'system'" />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useSettingsStore } from '@/stores/settings'
import { useAppStore } from '@/stores/app'
import GeneralSettings from '@/components/settings/GeneralSettings.vue'
import ModelSettings from '@/components/settings/ModelSettings.vue'
import DetectionSettings from '@/components/settings/DetectionSettings.vue'
import CameraSettings from '@/components/settings/CameraSettings.vue'
import NotificationSettings from '@/components/settings/NotificationSettings.vue'
import SystemSettings from '@/components/settings/SystemSettings.vue'

const settingsStore = useSettingsStore()
const appStore = useAppStore()

const activeSection = ref('general')

const sections = [
  { id: 'general', label: '常规设置', icon: 'bi bi-sliders' },
  { id: 'model', label: '模型配置', icon: 'bi bi-cpu' },
  { id: 'detection', label: '检测参数', icon: 'bi bi-eye' },
  { id: 'camera', label: '摄像头设置', icon: 'bi bi-camera-video' },
  { id: 'notification', label: '通知设置', icon: 'bi bi-bell' },
  { id: 'system', label: '系统维护', icon: 'bi bi-hdd' }
]

async function saveAllSettings() {
  const success = await settingsStore.saveSettings()
  if (success) {
    appStore.addNotification({
      type: 'success',
      title: '保存成功',
      message: '设置已成功保存'
    })
  } else {
    appStore.addNotification({
      type: 'error',
      title: '保存失败',
      message: '设置保存失败，请重试'
    })
  }
}

function resetAllSettings() {
  if (confirm('确定要重置所有设置吗？此操作不可撤销。')) {
    settingsStore.resetSettings()
    appStore.addNotification({
      type: 'info',
      title: '重置成功',
      message: '所有设置已重置为默认值'
    })
  }
}

onMounted(() => {
  settingsStore.fetchSettings()
})
</script>

<style scoped>
.page-header {
  margin-bottom: var(--space-xl);
}

.page-title {
  font-size: 28px;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: var(--space-xs);
}

.page-subtitle {
  font-size: 14px;
  color: var(--text-secondary);
}

.page-actions {
  display: flex;
  gap: var(--space-md);
}

.settings-container {
  display: grid;
  grid-template-columns: 240px 1fr;
  gap: var(--space-xl);
}

.settings-sidebar {
  background: var(--bg-secondary);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  padding: var(--space-md);
  height: fit-content;
  position: sticky;
  top: 80px;
}

.settings-nav-item {
  display: flex;
  align-items: center;
  gap: var(--space-md);
  padding: var(--space-sm) var(--space-md);
  border-radius: var(--radius-md);
  color: var(--text-secondary);
  text-decoration: none;
  cursor: pointer;
  transition: all var(--transition-fast);
  margin-bottom: 2px;
}

.settings-nav-item:hover {
  background: var(--bg-tertiary);
  color: var(--text-primary);
}

.settings-nav-item.active {
  background: var(--hk-primary-100);
  color: var(--hk-primary);
}

.settings-nav-item i {
  font-size: 18px;
  width: 20px;
  text-align: center;
}

.settings-content {
  display: flex;
  flex-direction: column;
  gap: var(--space-lg);
}

@media (max-width: 992px) {
  .settings-container {
    grid-template-columns: 1fr;
  }

  .settings-sidebar {
    position: static;
    display: flex;
    overflow-x: auto;
    gap: var(--space-sm);
  }

  .settings-nav-item {
    white-space: nowrap;
    margin-bottom: 0;
  }
}
</style>
