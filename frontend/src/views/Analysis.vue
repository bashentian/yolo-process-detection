<template>
  <div class="analysis">
    <div class="page-header">
      <div>
        <h1 class="page-title">数据分析</h1>
        <p class="page-subtitle">检测数据的统计分析和可视化</p>
      </div>
      <div class="page-actions">
        <button class="btn btn-secondary" @click="exportReport">
          <i class="bi bi-file-earmark-text"></i>
          导出报告
        </button>
        <button class="btn btn-primary" @click="refreshData">
          <i class="bi bi-arrow-clockwise"></i>
          刷新
        </button>
      </div>
    </div>

    <div class="stats-row">
      <div class="stat-box">
        <div class="stat-box-icon">
          <i class="bi bi-camera-video"></i>
        </div>
        <div class="stat-box-content">
          <div class="stat-box-value">{{ totalDetections }}</div>
          <div class="stat-box-label">总检测数</div>
        </div>
      </div>
      <div class="stat-box">
        <div class="stat-box-icon">
          <i class="bi bi-person"></i>
        </div>
        <div class="stat-box-content">
          <div class="stat-box-value">{{ personCount }}</div>
          <div class="stat-box-label">人员检测</div>
        </div>
      </div>
      <div class="stat-box">
        <div class="stat-box-icon">
          <i class="bi bi-car-front"></i>
        </div>
        <div class="stat-box-content">
          <div class="stat-box-value">{{ vehicleCount }}</div>
          <div class="stat-box-label">车辆检测</div>
        </div>
      </div>
      <div class="stat-box">
        <div class="stat-box-icon">
          <i class="bi bi-shield-check"></i>
        </div>
        <div class="stat-box-content">
          <div class="stat-box-value">{{ safetyCount }}</div>
          <div class="stat-box-label">安全装备</div>
        </div>
      </div>
    </div>

    <div class="analysis-grid">
      <div class="card chart-card">
        <div class="card-header">
          <div class="card-title">
            <i class="bi bi-graph-up"></i>
            检测趋势
          </div>
          <select v-model="trendPeriod" class="form-select-sm">
            <option value="day">按天</option>
            <option value="week">按周</option>
            <option value="month">按月</option>
          </select>
        </div>
        <div class="card-body">
          <div class="chart-placeholder">
            <i class="bi bi-bar-chart-line"></i>
            <span>检测趋势图表</span>
          </div>
        </div>
      </div>

      <div class="card chart-card">
        <div class="card-header">
          <div class="card-title">
            <i class="bi bi-pie-chart"></i>
            类别分布
          </div>
        </div>
        <div class="card-body">
          <div class="chart-placeholder">
            <i class="bi bi-pie-chart"></i>
            <span>类别分布饼图</span>
          </div>
        </div>
      </div>

      <div class="card chart-card">
        <div class="card-header">
          <div class="card-title">
            <i class="bi bi-clock-history"></i>
            时段分析
          </div>
        </div>
        <div class="card-body">
          <div class="chart-placeholder">
            <i class="bi bi-bar-chart"></i>
            <span>时段分布图</span>
          </div>
        </div>
      </div>

      <div class="card chart-card">
        <div class="card-header">
          <div class="card-title">
            <i class="bi bi-geo-alt"></i>
            区域分析
          </div>
        </div>
        <div class="card-body">
          <div class="chart-placeholder">
            <i class="bi bi-geo-alt"></i>
            <span>区域分布图</span>
          </div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div class="card-title">
          <i class="bi bi-table"></i>
          详细统计
        </div>
      </div>
      <div class="card-body">
        <div class="stats-table">
          <table>
            <thead>
              <tr>
                <th>指标</th>
                <th>数值</th>
                <th>占比</th>
                <th>变化</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="stat in stats" :key="stat.name">
                <td>{{ stat.name }}</td>
                <td>{{ stat.value }}</td>
                <td>{{ stat.percentage }}%</td>
                <td>
                  <span :class="stat.trend === 'up' ? 'trend-up' : 'trend-down'">
                    <i :class="stat.trend === 'up' ? 'bi bi-arrow-up' : 'bi bi-arrow-down'"></i>
                    {{ stat.change }}%
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useAppStore } from '@/stores/app'

const appStore = useAppStore()

const trendPeriod = ref('day')

const totalDetections = ref(1248)
const personCount = ref(456)
const vehicleCount = ref(328)
const safetyCount = ref(464)

const stats = ref([
  { name: '人员检测', value: 456, percentage: 36.5, trend: 'up', change: 12.3 },
  { name: '车辆检测', value: 328, percentage: 26.3, trend: 'up', change: 8.7 },
  { name: '安全帽检测', value: 234, percentage: 18.8, trend: 'down', change: -3.2 },
  { name: '安全背心检测', value: 230, percentage: 18.4, trend: 'up', change: 5.1 }
])

function exportReport() {
  appStore.addNotification({
    type: 'success',
    title: '导出成功',
    message: '分析报告已成功导出'
  })
}

function refreshData() {
  appStore.addNotification({
    type: 'success',
    title: '刷新成功',
    message: '数据已更新'
  })
}
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
  display: flex;
  align-items: center;
  gap: var(--space-md);
}

.stat-box-icon {
  width: 48px;
  height: 48px;
  border-radius: var(--radius-md);
  background: var(--hk-primary-100);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  color: var(--hk-primary);
}

.stat-box-content {
  flex: 1;
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

.analysis-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: var(--space-lg);
  margin-bottom: var(--space-xl);
}

.chart-card {
  min-height: 300px;
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.form-select-sm {
  width: auto;
  padding: var(--space-xs) var(--space-sm);
  background: var(--bg-tertiary);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  color: var(--text-primary);
  font-size: 12px;
}

.form-select-sm:focus {
  outline: none;
  border-color: var(--hk-primary);
}

.chart-placeholder {
  height: 220px;
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

.stats-table {
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
}

thead {
  background: var(--bg-tertiary);
}

th {
  padding: var(--space-md);
  text-align: left;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-secondary);
}

td {
  padding: var(--space-md);
  border-bottom: 1px solid var(--border-primary);
  color: var(--text-primary);
}

tr:hover td {
  background: var(--bg-tertiary);
}

.trend-up {
  color: var(--success);
  font-weight: 600;
}

.trend-down {
  color: var(--error);
  font-weight: 600;
}

@media (max-width: 1200px) {
  .analysis-grid {
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
