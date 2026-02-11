"""响应模式定义

提供API响应相关的数据模式类。
"""
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """错误响应模式"""
    error: str = Field(..., description="错误代码")
    message: str = Field(..., description="错误消息")
    details: Optional[dict] = Field(None, description="详细信息")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="时间戳"
    )


class HealthResponse(BaseModel):
    """健康检查响应模式"""
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="版本号")
    model_loaded: bool = Field(..., description="模型是否已加载")
    gpu_available: bool = Field(..., description="GPU是否可用")
    uptime_seconds: float = Field(..., description="运行时间(秒)")


class DetectionSummary(BaseModel):
    """检测摘要模式"""
    total: int = Field(..., description="总检测数")
    by_class: dict[str, int] = Field(
        default_factory=dict,
        description="按类别的检测数"
    )
    avg_confidence: float = Field(..., description="平均置信度")


class StatisticsResponse(BaseModel):
    """统计信息响应模式"""
    success: bool = Field(..., description="是否成功")
    summary: DetectionSummary = Field(..., description="检测摘要")
    stage_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="工序阶段分布"
    )
    frame_count: int = Field(..., description="处理帧数")
    processing_time: float = Field(..., description="处理时间")
    fps: float = Field(..., description="帧率")


class EfficiencyMetrics(BaseModel):
    """效率指标模式"""
    throughput: float = Field(..., description="吞吐量(对象/秒)")
    latency: float = Field(..., description="延迟(秒)")
    accuracy: float = Field(..., description="准确度")
    efficiency_score: float = Field(..., description="效率分数")


class EfficiencyResponse(BaseModel):
    """效率分析响应模式"""
    success: bool = Field(..., description="是否成功")
    status: str = Field(..., description="状态")
    score: float = Field(..., description="效率分数")
    metrics: EfficiencyMetrics = Field(..., description="效率指标")
    trend: str = Field(..., description="趋势")
    recommendations: list[str] = Field(
        default_factory=list,
        description="优化建议"
    )
    sample_count: int = Field(..., description="样本数")


class TimelinePoint(BaseModel):
    """时间线点模式"""
    timestamp: datetime = Field(..., description="时间戳")
    frame: int = Field(..., description="帧号")
    stage: str = Field(..., description="工序阶段")
    object_count: int = Field(..., description="对象数")
    avg_confidence: float = Field(..., description="平均置信度")


class TimelineResponse(BaseModel):
    """时间线响应模式"""
    success: bool = Field(..., description="是否成功")
    total_points: int = Field(..., description="数据点数")
    stage_timeline: list[TimelinePoint] = Field(
        default_factory=list,
        description="阶段时间线"
    )
    processing_range: dict = Field(..., description="处理范围")


class AnomalyInfo(BaseModel):
    """异常信息模式"""
    is_anomaly: bool = Field(..., description="是否异常")
    score: float = Field(..., description="异常分数")
    anomaly_type: Optional[str] = Field(None, description="异常类型")
    reasons: list[str] = Field(default_factory=list, description="异常原因")


class AnomalyResponse(BaseModel):
    """异常检测响应模式"""
    success: bool = Field(..., description="是否成功")
    total_checked: int = Field(..., description="检查数")
    anomaly_count: int = Field(..., description="异常数")
    anomaly_rate: float = Field(..., description="异常率")
    anomalies: list[AnomalyInfo] = Field(
        default_factory=list,
        description="异常列表"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="时间戳"
    )


class SceneInfo(BaseModel):
    """场景信息模式"""
    stage: str = Field(..., description="当前阶段")
    confidence: float = Field(..., description="置信度")
    context: str = Field(..., description="上下文描述")
    features: dict = Field(..., description="场景特征")


class SceneResponse(BaseModel):
    """场景分析响应模式"""
    success: bool = Field(..., description="是否成功")
    scene: SceneInfo = Field(..., description="场景信息")


class ExportResponse(BaseModel):
    """导出响应模式"""
    success: bool = Field(..., description="是否成功")
    file_path: str = Field(..., description="文件路径")
    file_size: int = Field(..., description="文件大小(bytes)")
    format: str = Field(..., description="导出格式")
    record_count: int = Field(..., description="记录数")
    message: str = Field(..., description="状态消息")


class ResetResponse(BaseModel):
    """重置响应模式"""
    success: bool = Field(..., description="是否成功")
    cleared_data: list[str] = Field(
        default_factory=list,
        description="清除的数据类型"
    )
    message: str = Field(..., description="状态消息")
