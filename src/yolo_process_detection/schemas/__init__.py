"""模式定义模块

提供Pydantic模式类，用于API请求/响应数据验证。
"""
from .detection import (
    DetectionResult,
    BoundingBox,
    DetectionResponse,
    BatchDetectionResponse
)
from .config import (
    DetectionConfigSchema,
    TrackingConfigSchema,
    AnalysisConfigSchema,
    UpdateConfigRequest
)
from .response import (
    ErrorResponse,
    HealthResponse,
    StatisticsResponse,
    EfficiencyResponse,
    TimelineResponse,
    AnomalyResponse
)

__all__ = [
    # Detection
    "DetectionResult",
    "BoundingBox",
    "DetectionResponse",
    "BatchDetectionResponse",
    
    # Config
    "DetectionConfigSchema",
    "TrackingConfigSchema",
    "AnalysisConfigSchema",
    "UpdateConfigRequest",
    
    # Response
    "ErrorResponse",
    "HealthResponse",
    "StatisticsResponse",
    "EfficiencyResponse",
    "TimelineResponse",
    "AnomalyResponse"
]
