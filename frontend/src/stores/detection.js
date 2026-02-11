import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import axios from 'axios'

export const useDetectionStore = defineStore('detection', () => {
  const cameras = ref([])
  const selectedCamera = ref(null)
  const detectionResults = ref([])
  const isDetecting = ref(false)
  const confidenceThreshold = ref(0.5)
  const iouThreshold = ref(0.45)
  const modelType = ref('yolo11n.pt')

  const activeCameras = computed(() => cameras.value.filter(c => c.status === 'running'))
  const selectedCameraData = computed(() => cameras.value.find(c => c.id === selectedCamera.value))

  async function fetchCameras() {
    try {
      const response = await axios.get('/api/cameras')
      cameras.value = response.data
    } catch (error) {
      console.error('Failed to fetch cameras:', error)
    }
  }

  async function startDetection(cameraId) {
    try {
      await axios.post(`/api/camera/${cameraId}/start`)
      const camera = cameras.value.find(c => c.id === cameraId)
      if (camera) camera.status = 'running'
    } catch (error) {
      console.error('Failed to start detection:', error)
    }
  }

  async function stopDetection(cameraId) {
    try {
      await axios.post(`/api/camera/${cameraId}/stop`)
      const camera = cameras.value.find(c => c.id === cameraId)
      if (camera) camera.status = 'stopped'
    } catch (error) {
      console.error('Failed to stop detection:', error)
    }
  }

  function selectCamera(cameraId) {
    selectedCamera.value = cameraId
  }

  function addDetectionResult(result) {
    detectionResults.value.unshift(result)
    if (detectionResults.value.length > 100) {
      detectionResults.value.pop()
    }
  }

  function setConfidenceThreshold(value) {
    confidenceThreshold.value = value
  }

  function setIouThreshold(value) {
    iouThreshold.value = value
  }

  function setModelType(value) {
    modelType.value = value
  }

  return {
    cameras,
    selectedCamera,
    detectionResults,
    isDetecting,
    confidenceThreshold,
    iouThreshold,
    modelType,
    activeCameras,
    selectedCameraData,
    fetchCameras,
    startDetection,
    stopDetection,
    selectCamera,
    addDetectionResult,
    setConfidenceThreshold,
    setIouThreshold,
    setModelType
  }
})
