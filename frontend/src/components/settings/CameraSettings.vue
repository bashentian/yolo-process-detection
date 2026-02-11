<template>
  <div class="settings-section">
    <div class="settings-section-header">
      <div class="settings-section-title">
        <i class="bi bi-camera-video"></i>
        摄像头设置
      </div>
    </div>
    <div class="settings-section-body">
      <div class="form-row">
        <div class="form-group">
          <label class="form-label">分辨率</label>
          <select v-model="settings.camera.resolution" class="form-select">
            <option value="640x480">640×480 (SD)</option>
            <option value="1280x720">1280×720 (HD)</option>
            <option value="1920x1080">1920×1080 (Full HD)</option>
            <option value="2560x1440">2560×1440 (2K)</option>
            <option value="3840x2160">3840×2160 (4K)</option>
          </select>
        </div>

        <div class="form-group">
          <label class="form-label">帧率 (FPS)</label>
          <input
            v-model.number="settings.camera.fps"
            type="number"
            class="form-input"
            min="1"
            max="60"
          >
        </div>
      </div>

      <div class="form-group">
        <div class="slider-control">
          <div class="slider-header">
            <span class="slider-label-text">码率 (Kbps)</span>
            <span class="slider-value">{{ settings.camera.bitrate }}</span>
          </div>
          <input
            v-model.number="settings.camera.bitrate"
            type="range"
            min="1000"
            max="10000"
            step="500"
          >
          <div class="form-help">视频流的码率，影响视频质量和带宽使用</div>
        </div>
      </div>

      <div class="switch-container">
        <div class="switch-info">
          <div class="switch-label">启用录制</div>
          <div class="switch-description">自动录制检测到的视频流</div>
        </div>
        <label class="switch">
          <input v-model="settings.camera.enableRecording" type="checkbox">
          <span class="switch-slider"></span>
        </label>
      </div>

      <div class="form-group">
        <label class="form-label">录制路径</label>
        <input
          v-model="settings.camera.recordingPath"
          type="text"
          class="form-input"
          placeholder="请输入录制文件保存路径"
        >
        <div class="form-help">录制视频文件的保存目录</div>
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
