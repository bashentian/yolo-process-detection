<template>
  <div class="history">
    <div class="page-header">
      <div>
        <h1 class="page-title">历史记录</h1>
        <p class="page-subtitle">查看历史检测记录和数据</p>
      </div>
      <div class="page-actions">
        <button class="btn btn-secondary" @click="exportData">
          <i class="bi bi-download"></i>
          导出数据
        </button>
        <button class="btn btn-primary" @click="refreshData">
          <i class="bi bi-arrow-clockwise"></i>
          刷新
        </button>
      </div>
    </div>

    <div class="filter-bar">
      <div class="filter-group">
        <label class="filter-label">时间范围</label>
        <select v-model="filters.timeRange" class="form-select">
          <option value="today">今天</option>
          <option value="week">最近7天</option>
          <option value="month">最近30天</option>
          <option value="custom">自定义</option>
        </select>
      </div>

      <div class="filter-group">
        <label class="filter-label">摄像头</label>
        <select v-model="filters.camera" class="form-select">
          <option value="all">全部摄像头</option>
          <option value="1">摄像头 01</option>
          <option value="2">摄像头 02</option>
          <option value="3">摄像头 03</option>
          <option value="4">摄像头 04</option>
        </select>
      </div>

      <div class="filter-group">
        <label class="filter-label">检测类别</label>
        <select v-model="filters.category" class="form-select">
          <option value="all">全部类别</option>
          <option value="person">人员</option>
          <option value="vehicle">车辆</option>
          <option value="safety">安全装备</option>
        </select>
      </div>

      <div class="filter-group">
        <label class="filter-label">搜索</label>
        <input v-model="filters.search" type="text" class="form-input" placeholder="搜索关键词...">
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div class="card-title">
          <i class="bi bi-table"></i>
          检测记录
        </div>
        <div class="card-actions">
          <span class="record-count">共 {{ records.length }} 条记录</span>
        </div>
      </div>
      <div class="card-body">
        <div class="records-table">
          <table>
            <thead>
              <tr>
                <th>时间</th>
                <th>摄像头</th>
                <th>类别</th>
                <th>置信度</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="record in paginatedRecords" :key="record.id">
                <td>{{ record.time }}</td>
                <td>{{ record.camera }}</td>
                <td>{{ record.category }}</td>
                <td>
                  <div class="confidence-cell">
                    <span class="confidence-value">{{ record.confidence }}%</span>
                    <div class="confidence-bar">
                      <div class="confidence-fill" :style="{ width: record.confidence + '%' }"></div>
                    </div>
                  </div>
                </td>
                <td>
                  <button class="btn-icon-sm" title="查看详情" @click="viewDetail(record)">
                    <i class="bi bi-eye"></i>
                  </button>
                  <button class="btn-icon-sm" title="下载图片" @click="downloadImage(record)">
                    <i class="bi bi-download"></i>
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="pagination">
          <button
            class="btn btn-secondary btn-sm"
            :disabled="currentPage === 1"
            @click="currentPage--"
          >
            <i class="bi bi-chevron-left"></i>
            上一页
          </button>
          <span class="page-info">第 {{ currentPage }} / {{ totalPages }} 页</span>
          <button
            class="btn btn-secondary btn-sm"
            :disabled="currentPage === totalPages"
            @click="currentPage++"
          >
            下一页
            <i class="bi bi-chevron-right"></i>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useAppStore } from '@/stores/app'

const appStore = useAppStore()

const filters = ref({
  timeRange: 'today',
  camera: 'all',
  category: 'all',
  search: ''
})

const currentPage = ref(1)
const pageSize = 10

const records = ref([
  { id: 1, time: '2024-01-10 10:30:25', camera: '摄像头 01', category: '人员', confidence: 96 },
  { id: 2, time: '2024-01-10 10:30:20', camera: '摄像头 01', category: '安全帽', confidence: 94 },
  { id: 3, time: '2024-01-10 10:30:15', camera: '摄像头 02', category: '车辆', confidence: 98 },
  { id: 4, time: '2024-01-10 10:30:10', camera: '摄像头 02', category: '人员', confidence: 92 },
  { id: 5, time: '2024-01-10 10:30:05', camera: '摄像头 01', category: '人员', confidence: 95 },
  { id: 6, time: '2024-01-10 10:30:00', camera: '摄像头 03', category: '安全帽', confidence: 91 },
  { id: 7, time: '2024-01-10 10:29:55', camera: '摄像头 04', category: '车辆', confidence: 97 },
  { id: 8, time: '2024-01-10 10:29:50', camera: '摄像头 01', category: '人员', confidence: 93 },
  { id: 9, time: '2024-01-10 10:29:45', camera: '摄像头 02', category: '安全帽', confidence: 96 },
  { id: 10, time: '2024-01-10 10:29:40', camera: '摄像头 03', category: '人员', confidence: 94 },
  { id: 11, time: '2024-01-10 10:29:35', camera: '摄像头 04', category: '车辆', confidence: 98 },
  { id: 12, time: '2024-01-10 10:29:30', camera: '摄像头 01', category: '人员', confidence: 92 }
])

const totalPages = computed(() => Math.ceil(records.value.length / pageSize))

const paginatedRecords = computed(() => {
  const start = (currentPage.value - 1) * pageSize
  const end = start + pageSize
  return records.value.slice(start, end)
})

function exportData() {
  appStore.addNotification({
    type: 'success',
    title: '导出成功',
    message: '数据已成功导出'
  })
}

function refreshData() {
  appStore.addNotification({
    type: 'success',
    title: '刷新成功',
    message: '数据已更新'
  })
}

function viewDetail(record) {
  appStore.addNotification({
    type: 'info',
    title: '查看详情',
    message: `正在查看记录 ${record.id} 的详细信息`
  })
}

function downloadImage(record) {
  appStore.addNotification({
    type: 'success',
    title: '下载成功',
    message: `记录 ${record.id} 的图片已下载`
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

.filter-bar {
  display: flex;
  gap: var(--space-md);
  margin-bottom: var(--space-xl);
  flex-wrap: wrap;
}

.filter-group {
  display: flex;
  flex-direction: column;
  gap: var(--space-xs);
}

.filter-label {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-secondary);
}

.form-input,
.form-select {
  width: 200px;
  padding: var(--space-sm) var(--space-md);
  background: var(--bg-tertiary);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  color: var(--text-primary);
  font-size: 14px;
}

.form-input:focus,
.form-select:focus {
  outline: none;
  border-color: var(--hk-primary);
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.card-actions {
  display: flex;
  align-items: center;
  gap: var(--space-md);
}

.record-count {
  font-size: 14px;
  color: var(--text-secondary);
}

.records-table {
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

.confidence-cell {
  display: flex;
  align-items: center;
  gap: var(--space-md);
}

.confidence-value {
  font-size: 14px;
  font-weight: 600;
  color: var(--hk-primary);
  min-width: 40px;
}

.confidence-bar {
  width: 80px;
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

.btn-icon-sm {
  width: 28px;
  height: 28px;
  border-radius: var(--radius-sm);
  background: transparent;
  border: none;
  color: var(--text-secondary);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all var(--transition-fast);
  margin-right: var(--space-xs);
}

.btn-icon-sm:hover {
  background: var(--bg-tertiary);
  color: var(--text-primary);
}

.pagination {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-md);
  margin-top: var(--space-lg);
}

.page-info {
  font-size: 14px;
  color: var(--text-secondary);
}

.btn-sm {
  padding: var(--space-xs) var(--space-md);
  font-size: 12px;
}

@media (max-width: 768px) {
  .page-header {
    flex-direction: column;
    align-items: flex-start;
    gap: var(--space-md);
  }

  .filter-bar {
    flex-direction: column;
  }

  .filter-group {
    width: 100%;
  }

  .form-input,
  .form-select {
    width: 100%;
  }
}
</style>
