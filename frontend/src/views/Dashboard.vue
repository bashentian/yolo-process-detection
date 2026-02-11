<template>
  <div class="dashboard">
    <div class="page-header">
      <div>
        <h1 class="page-title">控制台</h1>
        <p class="page-subtitle">系统概览和实时状态</p>
      </div>
      <div class="page-actions">
        <button class="btn btn-primary">
          <i class="bi bi-plus-lg"></i>
          添加摄像头
        </button>
      </div>
    </div>

    <div class="stats-row">
      <div class="stat-box">
        <div class="stat-box-value">{{ activeCameras }}</div>
        <div class="stat-box-label">运行中摄像头</div>
      </div>
      <div class="stat-box">
        <div class="stat-box-value">{{ totalDetections }}</div>
        <div class="stat-box-label">今日检测数</div>
      </div>
      <div class="stat-box">
        <div class="stat-box-value">{{ avgConfidence }}%</div>
        <div class="stat-box-label">平均置信度</div>
      </div>
      <div class="stat-box">
        <div class="stat-box-value">{{ systemStatus }}</div>
        <div class="stat-box-label">系统状态</div>
      </div>
    </div>

    <div class="dashboard-grid">
      <div class="card">
        <div class="card-header">
          <div class="card-title">
            <i class="bi bi-activity"></i>
            实时检测趋势
          </div>
        </div>
        <div class="card-body">
          <div class="chart-placeholder">
            <i class="bi bi-bar-chart-line"></i>
            <span>检测趋势图表</span>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <div class="card-title">
            <i class="bi bi-camera-video"></i>
            摄像头状态
          </div>
        </div>
        <div class="card-body">
          <div class="camera-list">
            <div v-for="camera in cameras" :key="camera.id" class="camera-item">
              <div class="camera-info">
                <div class="camera-name">{{ camera.name }}</div>
                <div class="camera-location">{{ camera.location }}</div>
              </div>
              <span class="status-badge" :class="camera.status">
                {{ camera.status === 'running' ? '运行中' : '已停止' }}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <div class="card-title">
            <i class="bi bi-clock-history"></i>
            最近检测
          </div>
        </div>
        <div class="card-body">
          <div class="detection-list">
            <div v-for="detection in recentDetections" :key="detection.id" class="detection-item">
              <div class="detection-info">
                <div class="detection-class">{{ detection.className }}</div>
                <div class="detection-meta">{{ detection.camera }} · {{ detection.time }}</div>
              </div>
              <div class="detection-confidence">{{ detection.confidence }}%</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useDetectionStore } from '@/stores/detection'

const detectionStore = useDetectionStore()

const cameras = ref([
  { id: 1, name: '摄像头 01', location: '生产线A', status: 'running' },
  { id: 2, name: '摄像头 02', location: '生产线B', status: 'running' },
  { id: 3, name: '摄像头 03', location: '仓库入口', status: 'stopped' },
  { id: 4, name: '摄像头 04', location: '包装区', status: 'stopped' }
])

const recentDetections = ref([
  { id: 1, className: '人员', camera: '摄像头 01', time: '10:30:25', confidence: 96 },
  { id: 2, className: '安全帽', camera: '摄像头 01', time: '10:30:20', confidence: 94 },
  { id: 3, className: '车辆', camera: '摄像头 02', time: '10:30:15', confidence: 98 },
  { id: 4, className: '人员', camera: '摄像头 02', time: '10:30:10', confidence: 92 }
])

const activeCameras = computed(() => cameras.value.filter(c => c.status === 'running').length)
const totalDetections = computed(() => 1248)
const avgConfidence = computed(() => 94)
const systemStatus = computed(() => '正常')

onMounted(() => {
  detectionStore.fetchCameras()
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

.dashboard-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: var(--space-lg);
}

.dashboard-grid .card:last-child {
  grid-column: 1 / -1;
}

.chart-placeholder {
  height: 200px;
  background: var(--bg-tertiary);
  border-radius: var(--radius-md);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--text-secondary);
  gap: var(--space-sm);
}

.chart-placeholder i {
  font-size: 48px;
}

.camera-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
}

.camera-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-md);
  background: var(--bg-tertiary);
  border-radius: var(--radius-md);
}

.camera-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.camera-name {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary);
}

.camera-location {
  font-size: 12px;
  color: var(--text-secondary);
}

.detection-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-sm);
}

.detection-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-sm) var(--space-md);
  background: var(--bg-tertiary);
  border-radius: var(--radius-md);
}

.detection-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.detection-class {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary);
}

.detection-meta {
  font-size: 12px;
  color: var(--text-secondary);
}

.detection-confidence {
  font-size: 14px;
  font-weight: 600;
  color: var(--hk-primary);
}

@media (max-width: 1200px) {
  .dashboard-grid {
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
}
</style>
