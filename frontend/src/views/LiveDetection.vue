<template>
  <div class="live-detection">
    <div class="page-header">
      <div>
        <h1 class="page-title">实时检测</h1>
        <p class="page-subtitle">监控实时视频流和检测结果</p>
      </div>
      <div class="page-actions">
        <button class="btn btn-secondary" @click="refreshCameras">
          <i class="bi bi-arrow-clockwise"></i>
          刷新
        </button>
        <button class="btn btn-primary" @click="toggleAllDetection">
          <i class="bi bi-play-fill"></i>
          {{ allRunning ? '全部停止' : '全部开始' }}
        </button>
      </div>
    </div>

    <div class="stats-row">
      <div class="stat-box">
        <div class="stat-box-value">{{ activeCameras }}</div>
        <div class="stat-box-label">运行中</div>
      </div>
      <div class="stat-box">
        <div class="stat-box-value">{{ totalDetections }}</div>
        <div class="stat-box-label">检测对象</div>
      </div>
      <div class="stat-box">
        <div class="stat-box-value">{{ avgConfidence }}%</div>
        <div class="stat-box-label">平均置信度</div>
      </div>
      <div class="stat-box">
        <div class="stat-box-value">{{ avgFps }}</div>
        <div class="stat-box-label">平均FPS</div>
      </div>
    </div>

    <div class="control-panel">
      <div class="control-group">
        <div class="control-label">画面布局</div>
        <div class="control-buttons">
          <button
            v-for="layout in layouts"
            :key="layout.value"
            class="btn btn-secondary"
            :class="{ active: currentLayout === layout.value }"
            @click="setLayout(layout.value)"
          >
            <i :class="layout.icon"></i>
            {{ layout.label }}
          </button>
        </div>
      </div>

      <div class="control-group">
        <div class="control-label">检测控制</div>
        <div class="control-buttons">
          <button class="btn btn-secondary" @click="takeSnapshot">
            <i class="bi bi-camera"></i>
            截图
          </button>
          <button class="btn btn-secondary" @click="toggleRecording">
            <i :class="isRecording ? 'bi bi-stop-fill' : 'bi bi-record-fill'"></i>
            {{ isRecording ? '停止录制' : '开始录制' }}
          </button>
        </div>
      </div>
    </div>

    <div class="video-grid" :class="`grid-${currentLayout}`">
      <div v-for="camera in cameras" :key="camera.id" class="video-card">
        <div class="video-card-header">
          <div class="video-card-title">
            <i class="bi bi-camera-video" style="color: var(--hk-primary);"></i>
            {{ camera.name }} - {{ camera.location }}
          </div>
          <span class="status-badge" :class="camera.status">
            {{ camera.status === 'running' ? '运行中' : '已停止' }}
          </span>
        </div>
        <div class="video-container">
          <img :src="camera.streamUrl" :alt="camera.name" :id="`stream-${camera.id}`">
          <div class="video-overlay">
            <div class="video-controls">
              <button class="video-control-btn" title="全屏" @click="toggleFullscreen(camera.id)">
                <i class="bi bi-fullscreen"></i>
              </button>
              <button class="video-control-btn" title="截图" @click="takeSnapshot(camera.id)">
                <i class="bi bi-camera"></i>
              </button>
              <button class="video-control-btn" title="设置" @click="openCameraSettings(camera.id)">
                <i class="bi bi-gear"></i>
              </button>
            </div>
            <div class="video-info-bar">
              <div class="video-stats">
                <div class="video-stat">
                  <div class="video-stat-value">{{ camera.detections }}</div>
                  <div class="video-stat-label">检测对象</div>
                </div>
                <div class="video-stat">
                  <div class="video-stat-value">{{ camera.confidence }}%</div>
                  <div class="video-stat-label">置信度</div>
                </div>
                <div class="video-stat">
                  <div class="video-stat-value">{{ camera.fps }}</div>
                  <div class="video-stat-label">FPS</div>
                </div>
              </div>
              <span class="video-resolution">{{ camera.resolution }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="detection-results">
      <div class="card">
        <div class="card-header">
          <div class="card-title">
            <i class="bi bi-list-check"></i>
            实时检测结果
          </div>
          <button class="btn btn-secondary btn-sm" @click="clearResults">
            <i class="bi bi-trash"></i>
            清空
          </button>
        </div>
        <div class="card-body">
          <div class="detection-list">
            <div v-for="result in detectionResults" :key="result.id" class="detection-item">
              <div class="detection-thumbnail">
                <img :src="result.thumbnail" :alt="result.className">
              </div>
              <div class="detection-info">
                <div class="detection-class">{{ result.className }}</div>
                <div class="detection-meta">{{ result.camera }} · {{ result.time }}</div>
              </div>
              <div class="detection-confidence">
                <div class="confidence-value">{{ result.confidence }}%</div>
                <div class="confidence-bar">
                  <div class="confidence-fill" :style="{ width: result.confidence + '%' }"></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useDetectionStore } from '@/stores/detection'
import { useAppStore } from '@/stores/app'

const detectionStore = useDetectionStore()
const appStore = useAppStore()

const cameras = ref([
  {
    id: 1,
    name: '摄像头 01',
    location: '生产线A',
    status: 'running',
    streamUrl: '/api/camera/stream/0',
    detections: 24,
    confidence: 96,
    fps: 30,
    resolution: '1920×1080'
  },
  {
    id: 2,
    name: '摄像头 02',
    location: '生产线B',
    status: 'running',
    streamUrl: '/api/camera/stream/1',
    detections: 18,
    confidence: 94,
    fps: 28,
    resolution: '1920×1080'
  },
  {
    id: 3,
    name: '摄像头 03',
    location: '仓库入口',
    status: 'stopped',
    streamUrl: '/api/camera/stream/2',
    detections: 0,
    confidence: 0,
    fps: 0,
    resolution: '1920×1080'
  },
  {
    id: 4,
    name: '摄像头 04',
    location: '包装区',
    status: 'stopped',
    streamUrl: '/api/camera/stream/3',
    detections: 0,
    confidence: 0,
    fps: 0,
    resolution: '1920×1080'
  }
])

const detectionResults = ref([
  {
    id: 1,
    className: '人员',
    camera: '摄像头 01',
    time: '10:30:25',
    confidence: 96,
    thumbnail: '/api/detection/thumbnail/1'
  },
  {
    id: 2,
    className: '安全帽',
    camera: '摄像头 01',
    time: '10:30:20',
    confidence: 94,
    thumbnail: '/api/detection/thumbnail/2'
  },
  {
    id: 3,
    className: '车辆',
    camera: '摄像头 02',
    time: '10:30:15',
    confidence: 98,
    thumbnail: '/api/detection/thumbnail/3'
  },
  {
    id: 4,
    className: '人员',
    camera: '摄像头 02',
    time: '10:30:10',
    confidence: 92,
    thumbnail: '/api/detection/thumbnail/4'
  }
])

const layouts = [
  { label: '1画面', value: 1, icon: 'bi bi-grid-1x2' },
  { label: '2画面', value: 2, icon: 'bi bi-grid-2x2' },
  { label: '4画面', value: 4, icon: 'bi bi-grid-3x3' },
  { label: '6画面', value: 6, icon: 'bi bi-grid' },
  { label: '9画面', value: 9, icon: 'bi bi-grid-3x3-gap' }
]

const currentLayout = ref(2)
const isRecording = ref(false)

let updateInterval = null

const activeCameras = computed(() => cameras.value.filter(c => c.status === 'running').length)
const totalDetections = computed(() => cameras.value.reduce((sum, c) => sum + c.detections, 0))
const avgConfidence = computed(() => {
  const running = cameras.value.filter(c => c.status === 'running')
  if (running.length === 0) return 0
  return Math.round(running.reduce((sum, c) => sum + c.confidence, 0) / running.length)
})
const avgFps = computed(() => {
  const running = cameras.value.filter(c => c.status === 'running')
  if (running.length === 0) return 0
  return Math.round(running.reduce((sum, c) => sum + c.fps, 0) / running.length)
})
const allRunning = computed(() => cameras.value.every(c => c.status === 'running'))

function setLayout(value) {
  currentLayout.value = value
}

function refreshCameras() {
  detectionStore.fetchCameras()
  appStore.addNotification({
    type: 'success',
    title: '刷新成功',
    message: '摄像头列表已更新'
  })
}

function toggleAllDetection() {
  if (allRunning.value) {
    cameras.value.forEach(camera => {
      if (camera.status === 'running') {
        stopDetection(camera.id)
      }
    })
  } else {
    cameras.value.forEach(camera => {
      if (camera.status === 'stopped') {
        startDetection(camera.id)
      }
    })
  }
}

async function startDetection(cameraId) {
  await detectionStore.startDetection(cameraId)
  const camera = cameras.value.find(c => c.id === cameraId)
  if (camera) camera.status = 'running'
}

async function stopDetection(cameraId) {
  await detectionStore.stopDetection(cameraId)
  const camera = cameras.value.find(c => c.id === cameraId)
  if (camera) camera.status = 'stopped'
}

function takeSnapshot(cameraId) {
  appStore.addNotification({
    type: 'success',
    title: '截图成功',
    message: cameraId ? `摄像头 ${cameraId} 截图已保存` : '所有摄像头截图已保存'
  })
}

function toggleRecording() {
  isRecording.value = !isRecording.value
  appStore.addNotification({
    type: 'info',
    title: isRecording.value ? '开始录制' : '停止录制',
    message: isRecording.value ? '录制已开始' : '录制已停止'
  })
}

function toggleFullscreen(cameraId) {
  const element = document.getElementById(`stream-${cameraId}`)
  if (element) {
    if (document.fullscreenElement) {
      document.exitFullscreen()
    } else {
      element.requestFullscreen()
    }
  }
}

function openCameraSettings(cameraId) {
  appStore.addNotification({
    type: 'info',
    title: '摄像头设置',
    message: `正在打开摄像头 ${cameraId} 的设置`
  })
}

function clearResults() {
  detectionResults.value = []
  appStore.addNotification({
    type: 'success',
    title: '清空成功',
    message: '检测结果已清空'
  })
}

function updateDetectionData() {
  cameras.value.forEach(camera => {
    if (camera.status === 'running') {
      camera.detections = Math.floor(Math.random() * 30) + 10
      camera.confidence = Math.floor(Math.random() * 10) + 90
      camera.fps = Math.floor(Math.random() * 5) + 28
    }
  })
}

onMounted(() => {
  detectionStore.fetchCameras()
  updateInterval = setInterval(updateDetectionData, 1000)
})

onUnmounted(() => {
  if (updateInterval) {
    clearInterval(updateInterval)
  }
})
</script>

<style scoped>
.page-header {
  margin-bottom: var(--space-xl);
  display: flex;
  align-items: center;
  justify-content: space-between;
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

.stats-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: var(--space-md);
  margin-bottom: var(--space-xl);
}

.stat-box {
  background: var(--bg-tertiary);
  border-radius: var(--radius-md);
  padding: var(--space-md);
  text-align: center;
}

.stat-box-value {
  font-size: 24px;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 4px;
}

.stat-box-label {
  font-size: 12px;
  color: var(--text-secondary);
}

.control-panel {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-lg);
  margin-bottom: var(--space-xl);
}

.control-group {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
}

.control-label {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-secondary);
}

.control-buttons {
  display: flex;
  gap: var(--space-md);
  flex-wrap: wrap;
}

.btn.active {
  background: var(--hk-primary);
  color: white;
  border-color: var(--hk-primary);
}

.video-grid {
  display: grid;
  gap: var(--space-lg);
  margin-bottom: var(--space-xl);
}

.video-grid.grid-1 {
  grid-template-columns: 1fr;
}

.video-grid.grid-2 {
  grid-template-columns: repeat(2, 1fr);
}

.video-grid.grid-4 {
  grid-template-columns: repeat(2, 1fr);
}

.video-grid.grid-6 {
  grid-template-columns: repeat(3, 1fr);
}

.video-grid.grid-9 {
  grid-template-columns: repeat(3, 1fr);
}

.video-card {
  background: var(--bg-secondary);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  overflow: hidden;
}

.video-card-header {
  padding: var(--space-md);
  border-bottom: 1px solid var(--border-primary);
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.video-card-title {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  font-weight: 500;
  color: var(--text-primary);
}

.video-container {
  background: var(--bg-primary);
  aspect-ratio: 16 / 9;
  position: relative;
  overflow: hidden;
}

.video-container img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.video-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  pointer-events: none;
}

.video-controls {
  position: absolute;
  top: var(--space-md);
  right: var(--space-md);
  display: flex;
  gap: var(--space-sm);
}

.video-control-btn {
  width: 36px;
  height: 36px;
  border-radius: var(--radius-md);
  background: rgba(0, 0, 0, 0.6);
  border: none;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all var(--transition-fast);
  pointer-events: auto;
}

.video-control-btn:hover {
  background: rgba(0, 0, 0, 0.8);
}

.video-info-bar {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: var(--space-md);
  background: linear-gradient(transparent, rgba(0, 0, 0, 0.8));
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.video-stats {
  display: flex;
  gap: var(--space-lg);
}

.video-stat {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
}

.video-stat-value {
  font-size: 18px;
  font-weight: 700;
  color: white;
}

.video-stat-label {
  font-size: 11px;
  color: rgba(255, 255, 255, 0.7);
  text-transform: uppercase;
}

.video-resolution {
  color: rgba(255, 255, 255, 0.7);
  font-size: 12px;
}

.detection-results {
  margin-top: var(--space-xl);
}

.detection-list {
  max-height: 400px;
  overflow-y: auto;
}

.detection-item {
  display: flex;
  align-items: center;
  gap: var(--space-md);
  padding: var(--space-md);
  border-bottom: 1px solid var(--border-primary);
  transition: background var(--transition-fast);
}

.detection-item:hover {
  background: var(--bg-tertiary);
}

.detection-item:last-child {
  border-bottom: none;
}

.detection-thumbnail {
  width: 60px;
  height: 60px;
  border-radius: var(--radius-sm);
  background: var(--bg-tertiary);
  overflow: hidden;
  flex-shrink: 0;
}

.detection-thumbnail img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.detection-info {
  flex: 1;
}

.detection-class {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary);
  margin-bottom: 4px;
}

.detection-meta {
  font-size: 12px;
  color: var(--text-secondary);
}

.detection-confidence {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 4px;
  min-width: 80px;
}

.confidence-value {
  font-size: 16px;
  font-weight: 700;
  color: var(--hk-primary);
}

.confidence-bar {
  width: 100%;
  height: 4px;
  background: var(--bg-tertiary);
  border-radius: 2px;
  overflow: hidden;
}

.confidence-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--hk-primary) 0%, var(--hk-primary-light) 100%);
  transition: width var(--transition-normal);
}

@media (max-width: 1200px) {
  .video-grid.grid-2,
  .video-grid.grid-4 {
    grid-template-columns: 1fr;
  }

  .control-panel {
    grid-template-columns: 1fr;
  }

  .stats-row {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 768px) {
  .page-header {
    flex-direction: column;
    align-items: flex-start;
    gap: var(--space-md);
  }

  .stats-row {
    grid-template-columns: 1fr;
  }

  .video-stats {
    gap: var(--space-md);
  }

  .video-stat-value {
    font-size: 14px;
  }

  .video-stat-label {
    font-size: 10px;
  }
}
</style>
