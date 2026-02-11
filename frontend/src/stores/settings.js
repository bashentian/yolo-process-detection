import { defineStore } from 'pinia'
import { ref } from 'vue'
import axios from 'axios'

export const useSettingsStore = defineStore('settings', () => {
  const settings = ref({
    general: {
      systemName: 'YOLO智能检测平台',
      language: 'zh-CN',
      timezone: 'Asia/Shanghai',
      autoSave: true,
      darkMode: true
    },
    model: {
      modelName: 'yolo11n.pt',
      device: 'cpu',
      batchSize: 16,
      confidenceThreshold: 0.5,
      iouThreshold: 0.45
    },
    detection: {
      enableTracking: true,
      trackMaxAge: 30,
      trackMinHits: 3,
      enableClassification: true,
      enableSegmentation: false
    },
    camera: {
      resolution: '1920x1080',
      fps: 30,
      bitrate: 4000,
      enableRecording: true,
      recordingPath: './recordings'
    },
    notification: {
      enableEmail: false,
      emailRecipient: '',
      enableWebhook: false,
      webhookUrl: '',
      alertThreshold: 0.8
    },
    system: {
      autoRestart: false,
      logLevel: 'INFO',
      maxLogSize: 100,
      backupEnabled: true,
      backupInterval: 24
    }
  })

  const loading = ref(false)
  const saving = ref(false)

  async function fetchSettings() {
    loading.value = true
    try {
      const response = await axios.get('/api/settings')
      settings.value = response.data
    } catch (error) {
      console.error('Failed to fetch settings:', error)
    } finally {
      loading.value = false
    }
  }

  async function saveSettings() {
    saving.value = true
    try {
      await axios.post('/api/settings', settings.value)
      return true
    } catch (error) {
      console.error('Failed to save settings:', error)
      return false
    } finally {
      saving.value = false
    }
  }

  function updateSettings(category, key, value) {
    if (settings.value[category]) {
      settings.value[category][key] = value
    }
  }

  function resetSettings() {
    settings.value = {
      general: {
        systemName: 'YOLO智能检测平台',
        language: 'zh-CN',
        timezone: 'Asia/Shanghai',
        autoSave: true,
        darkMode: true
      },
      model: {
        modelName: 'yolo11n.pt',
        device: 'cpu',
        batchSize: 16,
        confidenceThreshold: 0.5,
        iouThreshold: 0.45
      },
      detection: {
        enableTracking: true,
        trackMaxAge: 30,
        trackMinHits: 3,
        enableClassification: true,
        enableSegmentation: false
      },
      camera: {
        resolution: '1920x1080',
        fps: 30,
        bitrate: 4000,
        enableRecording: true,
        recordingPath: './recordings'
      },
      notification: {
        enableEmail: false,
        emailRecipient: '',
        enableWebhook: false,
        webhookUrl: '',
        alertThreshold: 0.8
      },
      system: {
        autoRestart: false,
        logLevel: 'INFO',
        maxLogSize: 100,
        backupEnabled: true,
        backupInterval: 24
      }
    }
  }

  return {
    settings,
    loading,
    saving,
    fetchSettings,
    saveSettings,
    updateSettings,
    resetSettings
  }
})
