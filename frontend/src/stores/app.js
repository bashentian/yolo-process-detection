import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useAppStore = defineStore('app', () => {
  const sidebarCollapsed = ref(false)
  const currentPage = ref('dashboard')
  const notifications = ref([])

  const isSidebarCollapsed = computed(() => sidebarCollapsed.value)

  function toggleSidebar() {
    sidebarCollapsed.value = !sidebarCollapsed.value
  }

  function setCurrentPage(page) {
    currentPage.value = page
  }

  function addNotification(notification) {
    notifications.value.push({
      id: Date.now(),
      ...notification
    })
  }

  function removeNotification(id) {
    const index = notifications.value.findIndex(n => n.id === id)
    if (index > -1) {
      notifications.value.splice(index, 1)
    }
  }

  return {
    sidebarCollapsed,
    currentPage,
    notifications,
    isSidebarCollapsed,
    toggleSidebar,
    setCurrentPage,
    addNotification,
    removeNotification
  }
})
