"""模型模块

提供检测和跟踪相关的模型类。
"""
from .detector import YOLODetector, Detection, SceneUnderstanding, AnomalyDetection, EfficiencyAnalyzer
from .tracker import ObjectTracker, TrackedObject

# Alias for backward compatibility or clarity
ProcessDetector = YOLODetector

__all__ = [
    "ProcessDetector",
    "YOLODetector",
    "Detection",
    "SceneUnderstanding",
    "AnomalyDetection",
    "EfficiencyAnalyzer",
    "ObjectTracker",
    "TrackedObject"
]
