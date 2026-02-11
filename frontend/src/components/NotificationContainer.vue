<template>
  <div class="notification-container">
    <transition-group name="notification">
      <div
        v-for="notification in appStore.notifications"
        :key="notification.id"
        class="notification"
        :class="[notification.type]"
      >
        <div class="notification-icon">
          <i :class="notificationIcon(notification.type)"></i>
        </div>
        <div class="notification-content">
          <div class="notification-title">{{ notification.title }}</div>
          <div class="notification-message">{{ notification.message }}</div>
        </div>
        <button class="notification-close" @click="appStore.removeNotification(notification.id)">
          <i class="bi bi-x-lg"></i>
        </button>
      </div>
    </transition-group>
  </div>
</template>

<script setup>
import { useAppStore } from '@/stores/app'

const appStore = useAppStore()

function notificationIcon(type) {
  const icons = {
    success: 'bi bi-check-circle-fill',
    error: 'bi bi-exclamation-circle-fill',
    warning: 'bi bi-exclamation-triangle-fill',
    info: 'bi bi-info-circle-fill'
  }
  return icons[type] || icons.info
}
</script>

<style scoped>
.notification-container {
  position: fixed;
  top: 80px;
  right: var(--space-xl);
  z-index: 1000;
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
}

.notification {
  min-width: 320px;
  padding: var(--space-md) var(--space-lg);
  background: var(--bg-secondary);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-lg);
  display: flex;
  align-items: flex-start;
  gap: var(--space-md);
}

.notification.success {
  border-color: var(--success);
}

.notification.error {
  border-color: var(--error);
}

.notification.warning {
  border-color: var(--warning);
}

.notification.info {
  border-color: var(--info);
}

.notification-icon {
  font-size: 20px;
  flex-shrink: 0;
}

.notification.success .notification-icon {
  color: var(--success);
}

.notification.error .notification-icon {
  color: var(--error);
}

.notification.warning .notification-icon {
  color: var(--warning);
}

.notification.info .notification-icon {
  color: var(--info);
}

.notification-content {
  flex: 1;
}

.notification-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 4px;
}

.notification-message {
  font-size: 13px;
  color: var(--text-secondary);
}

.notification-close {
  width: 24px;
  height: 24px;
  border-radius: var(--radius-sm);
  background: transparent;
  border: none;
  color: var(--text-tertiary);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all var(--transition-fast);
}

.notification-close:hover {
  background: var(--bg-tertiary);
  color: var(--text-primary);
}

.notification-enter-active,
.notification-leave-active {
  transition: all var(--transition-normal);
}

.notification-enter-from {
  opacity: 0;
  transform: translateX(400px);
}

.notification-leave-to {
  opacity: 0;
  transform: translateX(400px);
}
</style>
