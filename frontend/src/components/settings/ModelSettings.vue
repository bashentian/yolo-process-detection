<template>
  <div class="settings-section">
    <div class="settings-section-header">
      <div class="settings-section-title">
        <i class="bi bi-cpu"></i>
        模型配置
      </div>
    </div>
    <div class="settings-section-body">
      <div class="form-group">
        <label class="form-label">选择模型</label>
        <div class="model-grid">
          <div
            v-for="model in models"
            :key="model.id"
            class="model-card"
            :class="{ selected: settings.model.modelName === model.id }"
            @click="selectModel(model.id)"
          >
            <div class="model-card-header">
              <div class="model-name">{{ model.name }}</div>
              <span class="model-badge">{{ model.badge }}</span>
            </div>
            <div class="model-info">
              <div>大小: {{ model.size }}</div>
              <div>速度: {{ model.speed }}</div>
              <div>精度: {{ model.accuracy }}</div>
            </div>
          </div>
        </div>
      </div>

      <div class="form-row">
        <div class="form-group">
          <label class="form-label">计算设备</label>
          <select v-model="settings.model.device" class="form-select">
            <option value="cpu">CPU</option>
            <option value="cuda">CUDA (GPU)</option>
          </select>
          <div class="form-help">选择用于模型推理的计算设备</div>
        </div>

        <div class="form-group">
          <label class="form-label">批处理大小</label>
          <input
            v-model.number="settings.model.batchSize"
            type="number"
            class="form-input"
            min="1"
            max="64"
          >
          <div class="form-help">每批处理的图像数量，影响内存使用和速度</div>
        </div>
      </div>

      <div class="form-group">
        <div class="slider-control">
          <div class="slider-header">
            <span class="slider-label-text">置信度阈值</span>
            <span class="slider-value">{{ (settings.model.confidenceThreshold * 100).toFixed(0) }}%</span>
          </div>
          <input
            v-model.number="settings.model.confidenceThreshold"
            type="range"
            min="0"
            max="1"
            step="0.01"
          >
          <div class="form-help">检测目标的最小置信度，值越高检测越严格</div>
        </div>
      </div>

      <div class="form-group">
        <div class="slider-control">
          <div class="slider-header">
            <span class="slider-label-text">IoU 阈值</span>
            <span class="slider-value">{{ (settings.model.iouThreshold * 100).toFixed(0) }}%</span>
          </div>
          <input
            v-model.number="settings.model.iouThreshold"
            type="range"
            min="0"
            max="1"
            step="0.01"
          >
          <div class="form-help">非极大值抑制的IoU阈值，用于去除重复检测</div>
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

const models = [
  {
    id: 'yolo11n.pt',
    name: 'YOLO11n',
    badge: '最快',
    size: '6.3MB',
    speed: '1.5ms',
    accuracy: '63.7 mAP'
  },
  {
    id: 'yolov8n.pt',
    name: 'YOLOv8n',
    badge: '轻量',
    size: '6.2MB',
    speed: '1.6ms',
    accuracy: '63.3 mAP'
  },
  {
    id: 'yolov12n.pt',
    name: 'YOLOv12n',
    badge: '最新',
    size: '6.5MB',
    speed: '1.4ms',
    accuracy: '64.1 mAP'
  }
]

function selectModel(modelId) {
  settingsStore.updateSettings('model', 'modelName', modelId)
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

.model-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--space-md);
}

.model-card {
  background: var(--bg-tertiary);
  border: 2px solid var(--border-primary);
  border-radius: var(--radius-md);
  padding: var(--space-md);
  cursor: pointer;
  transition: all var(--transition-fast);
}

.model-card:hover {
  border-color: var(--border-secondary);
}

.model-card.selected {
  border-color: var(--hk-primary);
  background: var(--hk-primary-50);
}

.model-card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: var(--space-sm);
}

.model-name {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
}

.model-badge {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
  background: var(--hk-primary-100);
  color: var(--hk-primary);
}

.model-info {
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.6;
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

  .model-grid {
    grid-template-columns: 1fr;
  }
}
</style>
