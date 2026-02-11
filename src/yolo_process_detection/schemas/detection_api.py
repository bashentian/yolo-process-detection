"""检测相关模式定义"""
from typing import Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """边界框"""
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class DetectionResult(BaseModel):
    """检测结果"""
    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int]


class DetectionResponse(BaseModel):
    """检测响应"""
    success: bool
    detections: list[DetectionResult]
    image_width: int
    image_height: int
    inference_time: float
    model_name: str
    scene_info: Optional[dict] = None
    anomaly_info: Optional[dict] = None
    efficiency_info: Optional[dict] = None


class BatchDetectionResponse(BaseModel):
    """批量检测响应"""
    success: bool
    total_images: int
    successful_images: int
    failed_images: int
    total_detections: int
    total_time: float
    results: list[DetectionResponse]
    errors: list[dict]


class DetectionHistoryResponse(BaseModel):
    """检测历史响应"""
    success: bool
    total: int
    limit: int
    offset: int
    detections: list[dict]


class DetectionStatisticsResponse(BaseModel):
    """检测统计响应"""
    success: bool
    summary: dict
    time_range: dict
