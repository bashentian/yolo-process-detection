<template>
  <div class="app-container">
    <aside class="sidebar" :class="{ collapsed: appStore.isSidebarCollapsed }">
      <div class="sidebar-header">
        <div class="logo">
          <i class="bi bi-eye-fill"></i>
        </div>
        <span class="logo-text">AI检测平台</span>
      </div>

      <nav class="sidebar-nav">
        <div class="nav-section">
          <div class="nav-section-title">主菜单</div>
          <router-link to="/dashboard" class="nav-item" :class="{ active: $route.name === 'Dashboard' }">
            <i class="bi bi-grid-1x2-fill"></i>
            <span>控制台</span>
          </router-link>
          <router-link to="/live-detection" class="nav-item" :class="{ active: $route.name === 'LiveDetection' }">
            <i class="bi bi-camera-video-fill"></i>
            <span>实时检测</span>
          </router-link>
          <router-link to="/history" class="nav-item" :class="{ active: $route.name === 'History' }">
            <i class="bi bi-clock-history"></i>
            <span>历史记录</span>
          </router-link>
          <router-link to="/analysis" class="nav-item" :class="{ active: $route.name === 'Analysis' }">
            <i class="bi bi-bar-chart-fill"></i>
            <span>数据分析</span>
          </router-link>
        </div>

        <div class="nav-section">
          <div class="nav-section-title">系统</div>
          <router-link to="/settings" class="nav-item" :class="{ active: $route.name === 'Settings' }">
            <i class="bi bi-gear-fill"></i>
            <span>系统设置</span>
          </router-link>
          <a href="#" class="nav-item">
            <i class="bi bi-question-circle-fill"></i>
            <span>帮助文档</span>
          </a>
        </div>
      </nav>
    </aside>

    <main class="main-content">
      <header class="top-header">
        <div class="header-left">
          <button class="btn-icon" @click="appStore.toggleSidebar">
            <i class="bi bi-list"></i>
          </button>
          <nav class="breadcrumb">
            <router-link to="/">首页</router-link>
            <span class="separator">/</span>
            <span class="active">{{ currentPageTitle }}</span>
          </nav>
        </div>
        <div class="header-right">
          <button class="btn-icon">
            <i class="bi bi-bell"></i>
          </button>
          <div class="user-menu">
            <div class="user-avatar">A</div>
            <span class="user-name">管理员</span>
          </div>
        </div>
      </header>

      <div class="content-wrapper">
        <router-view />
      </div>
    </main>

    <NotificationContainer />
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { useAppStore } from '@/stores/app'
import NotificationContainer from '@/components/NotificationContainer.vue'

const route = useRoute()
const appStore = useAppStore()

const currentPageTitle = computed(() => {
  const titles = {
    Dashboard: '控制台',
    LiveDetection: '实时检测',
    History: '历史记录',
    Analysis: '数据分析',
    Settings: '系统设置'
  }
  return titles[route.name] || '未知页面'
})
</script>

<style scoped>
.sidebar {
  width: 260px;
  background: var(--bg-secondary);
  border-right: 1px solid var(--border-primary);
  display: flex;
  flex-direction: column;
  position: fixed;
  height: 100vh;
  z-index: 100;
  transition: transform var(--transition-normal);
}

.sidebar.collapsed {
  transform: translateX(-260px);
}

.sidebar-header {
  padding: var(--space-lg);
  border-bottom: 1px solid var(--border-primary);
  display: flex;
  align-items: center;
  gap: var(--space-md);
}

.logo {
  width: 40px;
  height: 40px;
  background: linear-gradient(135deg, var(--hk-primary) 0%, var(--hk-primary-light) 100%);
  border-radius: var(--radius-md);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  color: white;
}

.logo-text {
  font-size: 18px;
  font-weight: 700;
  color: var(--text-primary);
}

.sidebar-nav {
  flex: 1;
  padding: var(--space-md) 0;
  overflow-y: auto;
}

.nav-section {
  margin-bottom: var(--space-lg);
}

.nav-section-title {
  padding: var(--space-sm) var(--space-lg);
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-tertiary);
}

.nav-item {
  display: flex;
  align-items: center;
  padding: var(--space-sm) var(--space-lg);
  margin: 2px var(--space-sm);
  border-radius: var(--radius-md);
  color: var(--text-secondary);
  text-decoration: none;
  transition: all var(--transition-fast);
  cursor: pointer;
  gap: var(--space-md);
}

.nav-item:hover {
  background: var(--bg-tertiary);
  color: var(--text-primary);
}

.nav-item.active {
  background: var(--hk-primary-100);
  color: var(--hk-primary);
}

.nav-item i {
  font-size: 20px;
  width: 24px;
  text-align: center;
}

.main-content {
  flex: 1;
  margin-left: 260px;
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  transition: margin-left var(--transition-normal);
}

.sidebar.collapsed + .main-content {
  margin-left: 0;
}

.top-header {
  height: 64px;
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border-primary);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 var(--space-xl);
  position: sticky;
  top: 0;
  z-index: 50;
}

.header-left {
  display: flex;
  align-items: center;
  gap: var(--space-lg);
}

.btn-icon {
  width: 36px;
  height: 36px;
  border-radius: var(--radius-md);
  background: transparent;
  border: none;
  color: var(--text-secondary);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all var(--transition-fast);
}

.btn-icon:hover {
  background: var(--bg-tertiary);
  color: var(--text-primary);
}

.breadcrumb {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  color: var(--text-secondary);
  font-size: 14px;
}

.breadcrumb a {
  color: var(--text-secondary);
  text-decoration: none;
}

.breadcrumb a:hover {
  color: var(--text-primary);
}

.breadcrumb .separator {
  color: var(--text-tertiary);
}

.breadcrumb .active {
  color: var(--text-primary);
  font-weight: 500;
}

.header-right {
  display: flex;
  align-items: center;
  gap: var(--space-md);
}

.user-menu {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  padding: var(--space-sm) var(--space-md);
  background: var(--bg-tertiary);
  border-radius: var(--radius-md);
  cursor: pointer;
}

.user-avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--hk-primary) 0%, var(--hk-primary-light) 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  font-size: 14px;
}

.user-name {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary);
}

.content-wrapper {
  flex: 1;
  padding: var(--space-xl);
}
</style>
