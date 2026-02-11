"""配置模式定义

提供配置相关的Pydantic模式类。
"""
from typing import Literal
from pydantic import BaseModel, Field


class DetectionConfigSchema(BaseModel):
    """检测配置模式"""
    model_name: str = Field(default="yolo11n.pt", description="模型名称")
    confidence_threshold: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="置信度阈值"
    )
    iou_threshold: float = Field(
        default=0.45,
        ge=0,
        le=1,
        description="IoU阈值"
    )
    max_detections: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="最大检测数"
    )
    device: Literal["cpu", "cuda"] = Field(
        default="cpu",
        description="计算设备"
    )


class TrackingConfigSchema(BaseModel):
    """跟踪配置模式"""
    enabled: bool = Field(default=True, description="是否启用")
    max_age: int = Field(default=30, ge=1, le=1000, description="最大丢失帧数")
    min_hits: int = Field(default=3, ge=1, le=100, description="最小命中次数")
    iou_threshold: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="IoU阈值"
    )


class AnalysisConfigSchema(BaseModel):
    """分析配置模式"""
    scene_understanding_enabled: bool = Field(
        default=True,
        description="是否启用场景理解"
    )
    anomaly_threshold: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="异常检测阈值"
    )
    efficiency_window_size: int = Field(
        default=50,
        ge=10,
        le=1000,
        description="效率分析窗口大小"
    )
    use_attention: bool = Field(
        default=False,
        description="是否使用注意力机制"
    )


class UpdateConfigRequest(BaseModel):
    """更新配置请求模式"""
    detection: DetectionConfigSchema | None = Field(
        None,
        description="检测配置"
    )
    tracking: TrackingConfigSchema | None = Field(
        None,
        description="跟踪配置"
    )
    analysis: AnalysisConfigSchema | None = Field(
        None,
        description="分析配置"
    )


class ConfigStatusResponse(BaseModel):
    """配置状态响应模式"""
    success: bool = Field(..., description="是否成功")
    current_config: dict = Field(..., description="当前配置")
    updated_fields: list[str] = Field(
        default_factory=list,
        description="更新的字段"
    )
    message: str = Field(..., description="状态消息")
