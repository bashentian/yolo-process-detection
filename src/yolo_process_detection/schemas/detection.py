"""检测模式定义

提供检测相关的数据模式类。
"""
from typing import Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """边界框模式
    
    表示检测到的目标边界框。
    """
    x_min: float = Field(..., ge=0, description="左上角X坐标")
    y_min: float = Field(..., ge=0, description="左上角Y坐标")
    x_max: float = Field(..., ge=0, description="右下角X坐标")
    y_max: float = Field(..., ge=0, description="右下角Y坐标")
    
    @property
    def width(self) -> float:
        """计算宽度"""
        return self.x_max - self.x_min
    
    @property
    def height(self) -> float:
        """计算高度"""
        return self.y_max - self.y_min
    
    @property
    def area(self) -> float:
        """计算面积"""
        return self.width * self.height
    
    @property
    def center(self) -> tuple[float, float]:
        """计算中心点"""
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)


class DetectionResult(BaseModel):
    """单次检测结果模式"""
    bbox: BoundingBox = Field(..., description="边界框")
    confidence: float = Field(..., ge=0, le=1, description="置信度")
    class_id: int = Field(..., ge=0, description="类别ID")
    class_name: str = Field(..., description="类别名称")
    track_id: Optional[int] = Field(None, description="跟踪ID")


class DetectionResponse(BaseModel):
    """检测响应模式"""
    success: bool = Field(..., description="是否成功")
    detections: list[DetectionResult] = Field(
        default_factory=list,
        description="检测结果列表"
    )
    image_width: int = Field(..., description="图像宽度")
    image_height: int = Field(..., description="图像高度")
    inference_time: float = Field(..., ge=0, description="推理时间(秒)")
    model_name: str = Field(..., description="模型名称")
    scene_info: Optional[dict] = Field(
        None,
        description="场景信息"
    )
    anomaly_info: Optional[dict] = Field(
        None,
        description="异常检测信息"
    )
    efficiency_info: Optional[dict] = Field(
        None,
        description="效率分析信息"
    )


class BatchDetectionResponse(BaseModel):
    """批量检测响应模式"""
    success: bool = Field(..., description="是否成功")
    total_images: int = Field(..., description="总图像数")
    successful_images: int = Field(..., description="成功处理的图像数")
    failed_images: int = Field(..., description="处理失败的图像数")
    total_detections: int = Field(..., description="总检测数")
    total_time: float = Field(..., ge=0, description="总处理时间(秒)")
    results: list[DetectionResponse] = Field(
        default_factory=list,
        description="每个图像的检测结果"
    )
    errors: list[dict] = Field(
        default_factory=list,
        description="处理错误列表"
    )


class DetectionHistoryItem(BaseModel):
    """检测历史项模式"""
    id: int = Field(..., description="检测ID")
    camera_id: int = Field(..., description="摄像头ID")
    timestamp: str = Field(..., description="检测时间戳")
    class_name: str = Field(..., description="类别名称")
    confidence: float = Field(..., ge=0, le=1, description="置信度")
    bbox: list[float] = Field(..., description="边界框坐标")


class DetectionHistoryResponse(BaseModel):
    """检测历史响应模式"""
    success: bool = Field(..., description="是否成功")
    total: int = Field(..., description="总记录数")
    detections: list[DetectionHistoryItem] = Field(
        default_factory=list,
        description="检测历史列表"
    )


class ClassStatistics(BaseModel):
    """类别统计模式"""
    class_name: str = Field(..., description="类别名称")
    count: int = Field(..., ge=0, description="检测次数")


class DetectionStatisticsResponse(BaseModel):
    """检测统计响应模式"""
    success: bool = Field(..., description="是否成功")
    total_detections: int = Field(..., ge=0, description="总检测数")
    avg_confidence: float = Field(..., ge=0, le=1, description="平均置信度")
    by_class: list[ClassStatistics] = Field(
        default_factory=list,
        description="按类别统计"
    )
