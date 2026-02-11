"""配置管理模块

提供类型安全的配置管理，使用Pydantic进行验证。
"""
from pathlib import Path
from typing import Literal
from functools import lru_cache

from pydantic import (
    Field,
    field_validator,
    model_validator,
    ConfigDict
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class DetectionConfig(BaseSettings):
    """检测配置"""
    model_name: str = Field(
        default="yolo11n.pt",
        description="模型文件名"
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="置信度阈值"
    )
    iou_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="IoU阈值"
    )
    max_detections: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="最大检测数量"
    )
    device: Literal["cpu", "cuda"] = Field(
        default="cpu",
        description="计算设备"
    )
    
    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """验证模型名称"""
        valid_models = [
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
            "yolov12n.pt"
        ]
        if v not in valid_models:
            raise ValueError(
                f"无效的模型名称: {v}。\n"
                f"支持的模型: {', '.join(valid_models)}"
            )
        return v
    
    model_config = SettingsConfigDict(
        env_prefix="DETECTION_",
        validate_default=True
    )


class TrackingConfig(BaseSettings):
    """跟踪配置"""
    enabled: bool = Field(
        default=True,
        description="是否启用跟踪"
    )
    max_age: int = Field(
        default=30,
        ge=1,
        le=1000,
        description="最大跟踪丢失帧数"
    )
    min_hits: int = Field(
        default=3,
        ge=1,
        le=100,
        description="最小命中次数"
    )
    iou_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="跟踪IoU阈值"
    )
    
    model_config = SettingsConfigDict(
        env_prefix="TRACKING_",
        validate_default=True
    )


class AnalysisConfig(BaseSettings):
    """分析配置"""
    scene_understanding_enabled: bool = Field(
        default=True,
        description="是否启用场景理解"
    )
    anomaly_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="异常检测阈值"
    )
    anomaly_history_size: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="异常检测历史大小"
    )
    efficiency_window_size: int = Field(
        default=50,
        ge=10,
        le=1000,
        description="效率分析窗口大小"
    )
    use_attention: bool = Field(
        default=False,
        description="是否使用注意力机制(YOLOv12)"
    )
    
    model_config = SettingsConfigDict(
        env_prefix="ANALYSIS_",
        validate_default=True
    )


class VideoConfig(BaseSettings):
    """视频配置"""
    frame_width: int = Field(
        default=640,
        ge=32,
        le=4096,
        description="帧宽度"
    )
    frame_height: int = Field(
        default=640,
        ge=32,
        le=4096,
        description="帧高度"
    )
    fps_limit: int = Field(
        default=30,
        ge=1,
        le=120,
        description="FPS限制"
    )
    buffer_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="帧缓冲区大小"
    )
    
    model_config = SettingsConfigDict(
        env_prefix="VIDEO_",
        validate_default=True
    )


class LoggingConfig(BaseSettings):
    """日志配置"""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="日志级别"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="日志格式"
    )
    file_enabled: bool = Field(
        default=True,
        description="是否启用文件日志"
    )
    file_path: Path = Field(
        default=Path("logs/app.log"),
        description="日志文件路径"
    )
    
    @model_validator(mode="after")
    def validate_file_path(self):
        """验证文件路径"""
        if self.file_enabled:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
        return self
    
    model_config = SettingsConfigDict(
        env_prefix="LOGGING_",
        validate_default=True
    )


class Settings(
    DetectionConfig,
    TrackingConfig,
    AnalysisConfig,
    VideoConfig,
    LoggingConfig
):
    """应用设置
    
    组合所有配置类，提供统一的配置访问接口。
    """
    project_root: Path = Field(
        default=Path(__file__).parent.parent.parent,
        description="项目根目录"
    )
    data_root: Path = Field(
        default=Path("data"),
        description="数据目录"
    )
    models_root: Path = Field(
        default=Path("models"),
        description="模型目录"
    )
    outputs_root: Path = Field(
        default=Path("outputs"),
        description="输出目录"
    )
    uploads_root: Path = Field(
        default=Path("uploads"),
        description="上传目录"
    )
    cache_root: Path = Field(
        default=Path("cache"),
        description="缓存目录"
    )
    secret_key: str = Field(
        default="your-secret-key-change-this-in-production",
        description="JWT密钥"
    )
    access_token_expire_minutes: int = Field(
        default=30,
        ge=1,
        le=1440,
        description="访问令牌过期时间（分钟）"
    )
    
    @model_validator(mode="after")
    def setup_directories(self):
        """初始化目录"""
        for directory in [
            self.data_root,
            self.models_root,
            self.outputs_root,
            self.uploads_root,
            self.cache_root
        ]:
            directory.mkdir(parents=True, exist_ok=True)
        return self
    
    @property
    def class_names(self) -> dict[int, str]:
        """获取类别名称映射"""
        return {
            0: "worker",
            1: "machine",
            2: "product",
            3: "tool",
            4: "material"
        }
    
    @property
    def process_stages(self) -> list[str]:
        """获取工序阶段列表"""
        return [
            "preparation",
            "processing",
            "assembly",
            "quality_check",
            "packaging"
        ]
    
    model_config = SettingsConfigDict(
        validate_default=True,
        extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    """获取应用设置（单例模式）
    
    使用lru_cache实现单例，避免重复创建配置对象。
    """
    return Settings()


def reload_settings() -> Settings:
    """重新加载设置"""
    get_settings.cache_clear()
    return get_settings()
