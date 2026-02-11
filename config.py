import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_ROOT = PROJECT_ROOT / "models"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
LOGS_ROOT = PROJECT_ROOT / "logs"
UPLOADS_ROOT = PROJECT_ROOT / "uploads"
CACHE_ROOT = PROJECT_ROOT / "cache"


def _parse_env_bool(value: str, default: bool = False) -> bool:
    """解析布尔值环境变量"""
    return value.lower() in ("true", "1", "yes", "on") if isinstance(value, str) else default


def _parse_env_list(value: str, default: str = "") -> List[str]:
    """解析列表环境变量"""
    if not value:
        return default.split(",") if default else []
    return [item.strip() for item in value.split(",")]


@dataclass
class ProcessDetectionConfig:
    """工序检测配置类
    
    支持实例化配置和动态更新
    """
    
    # 基础配置
    model_name: str = field(default="yolo11n.pt")
    confidence_threshold: float = field(default=0.5)
    iou_threshold: float = field(default=0.45)
    max_detections: int = field(default=100)
    
    device: str = field(default="cpu")
    video_source: int = field(default=0)
    
    frame_resize: tuple = field(default=(640, 640))
    display_size: tuple = field(default=(1280, 720))
    fps_limit: int = field(default=30)
    
    save_results: bool = field(default=True)
    output_format: str = field(default="mp4")
    
    class_names: Dict[int, str] = field(default_factory=lambda: {
        0: "worker",
        1: "machine",
        2: "product",
        3: "tool",
        4: "material"
    })
    
    process_stages: List[str] = field(default_factory=lambda: [
        "preparation",
        "processing",
        "assembly",
        "quality_check",
        "packaging"
    ])
    
    # 跟踪配置
    tracking_enabled: bool = field(default=True)
    tracking_max_age: int = field(default=30)
    tracking_min_hits: int = field(default=3)
    
    # YOLOv12高级功能配置
    use_attention: bool = field(default=False)
    scene_understanding_enabled: bool = field(default=True)
    anomaly_threshold: float = field(default=0.5)
    anomaly_history_size: int = field(default=100)
    efficiency_window_size: int = field(default=50)
    
    def __post_init__(self):
        """初始化后处理"""
        # 从环境变量加载
        self.model_name = os.getenv("MODEL_NAME", self.model_name)
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", self.confidence_threshold))
        self.iou_threshold = float(os.getenv("IOU_THRESHOLD", self.iou_threshold))
        self.max_detections = int(os.getenv("MAX_DETECTIONS", self.max_detections))
        
        cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")
        self.device = "cuda" if cuda_visible else "cpu"
        
        self.use_attention = _parse_env_bool(os.getenv("USE_ATTENTION", str(self.use_attention)))
        self.scene_understanding_enabled = _parse_env_bool(
            os.getenv("SCENE_UNDERSTANDING_ENABLED", str(self.scene_understanding_enabled))
        )
        self.anomaly_threshold = float(os.getenv("ANOMALY_THRESHOLD", self.anomaly_threshold))
        self.anomaly_history_size = int(os.getenv("ANOMALY_HISTORY_SIZE", self.anomaly_history_size))
        self.efficiency_window_size = int(os.getenv("EFFICIENCY_WINDOW_SIZE", self.efficiency_window_size))
    
    def update(self, **kwargs):
        """动态更新配置
        
        Args:
            **kwargs: 配置参数名和值
        """
        valid_params = {
            'model_name', 'confidence_threshold', 'iou_threshold', 'max_detections',
            'device', 'video_source', 'frame_resize', 'display_size', 'fps_limit',
            'save_results', 'output_format', 'tracking_enabled', 'tracking_max_age',
            'tracking_min_hits', 'use_attention', 'scene_understanding_enabled',
            'anomaly_threshold', 'anomaly_history_size', 'efficiency_window_size'
        }
        
        for key, value in kwargs.items():
            if key in valid_params:
                setattr(self, key, value)
            else:
                raise ValueError(f"无效的配置参数: {key}")
    
    def validate(self) -> Dict[str, Any]:
        """验证配置有效性
        
        Returns:
            Dict: 验证结果
        """
        errors = []
        
        if not 0 < self.confidence_threshold <= 1:
            errors.append(f"confidence_threshold必须在0-1之间: {self.confidence_threshold}")
        
        if not 0 < self.iou_threshold <= 1:
            errors.append(f"iou_threshold必须在0-1之间: {self.iou_threshold}")
        
        if self.max_detections < 1:
            errors.append(f"max_detections必须大于0: {self.max_detections}")
        
        if self.fps_limit < 1:
            errors.append(f"fps_limit必须大于0: {self.fps_limit}")
        
        if not 0 < self.anomaly_threshold <= 1:
            errors.append(f"anomaly_threshold必须在0-1之间: {self.anomaly_threshold}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Dict: 配置字典
        """
        return {
            'model_name': self.model_name,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'max_detections': self.max_detections,
            'device': self.device,
            'video_source': self.video_source,
            'frame_resize': self.frame_resize,
            'display_size': self.display_size,
            'fps_limit': self.fps_limit,
            'save_results': self.save_results,
            'output_format': self.output_format,
            'class_names': self.class_names,
            'process_stages': self.process_stages,
            'tracking_enabled': self.tracking_enabled,
            'tracking_max_age': self.tracking_max_age,
            'tracking_min_hits': self.tracking_min_hits,
            'use_attention': self.use_attention,
            'scene_understanding_enabled': self.scene_understanding_enabled,
            'anomaly_threshold': self.anomaly_threshold,
            'anomaly_history_size': self.anomaly_history_size,
            'efficiency_window_size': self.efficiency_window_size
        }


class AugmentationConfig:
    ENABLED = True
    AUGMENT_RATIO = 0.8
    AUGMENTATION_FACTOR = 3
    
    BRIGHTNESS_LIMIT = 0.2
    CONTRAST_LIMIT = 0.2
    BLUR_LIMIT = 5
    NOISE_VARIANCE = (10.0, 50.0)
    ROTATION_LIMIT = 15
    SCALE_LIMIT = 0.2
    SHEAR_LIMIT = 10.0
    FLIP_PROBABILITY = 0.5
    
    MIXUP_PROBABILITY = 0.5
    CUTMIX_PROBABILITY = 0.5
    MOSAIC_PROBABILITY = 0.5


class ModelOptimizationConfig:
    ENABLED = True
    PRUNING_RATIO = 0.2
    QUANTIZATION_TYPE = "dynamic"
    KNOWLEDGE_DISTILLATION = False
    TEMPERATURE = 3.0
    ALPHA = 0.5
    
    TEACHER_MODEL = "yolov8l.pt"
    STUDENT_MODEL = "yolov8n.pt"
    
    TARGET_MODEL_SIZE_MB = 5.0
    MIN_ACCURACY_DROP = 0.02


class HyperparameterSearchConfig:
    ENABLED = True
    OPTIMIZATION_METHOD = "bayesian"
    N_TRIALS = 50
    TIMEOUT = 7200
    
    LEARNING_RATE_RANGE = (1e-5, 1e-2)
    MOMENTUM_RANGE = (0.8, 0.98)
    WEIGHT_DECAY_RANGE = (0.0, 0.001)
    
    BATCH_SIZES = [8, 16, 32]
    EPOCHS_RANGE = (10, 100)
    
    WARMUP_EPOCHS_RANGE = (0, 5)
    BOX_LOSS_RANGE = (0.02, 0.2)
    CLS_LOSS_RANGE = (0.2, 1.0)
    DFL_LOSS_RANGE = (0.5, 2.0)


class ExportConfig:
    ONNX_ENABLED = True
    TORCHSCRIPT_ENABLED = True
    TENSORRT_ENABLED = False
    
    ONNX_OPSET_VERSION = 12
    ONNX_SIMPLIFY = True
    
    TENSORRT_PRECISION = "fp16"
    TENSORRT_WORKSPACE_GB = 4
    TENSORRT_BATCH_SIZE = 1
    
    INPUT_SIZE = 640
    DYNAMIC_BATCH = False


class APIDeploymentConfig:
    HOST = "0.0.0.0"
    PORT = 8000
    WORKERS = 1
    
    MAX_UPLOAD_SIZE = 50 * 1024 * 1024
    ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    MAX_BATCH_SIZE = 10
    
    ENABLE_CORS = True
    CORS_ORIGINS = ["*"]
    
    RATE_LIMIT_ENABLED = False
    RATE_LIMIT_REQUESTS = 100
    RATE_LIMIT_PERIOD = 60
    
    LOG_LEVEL = "INFO"
    ACCESS_LOG = True


class MonitoringConfig:
    ENABLED = True
    DATA_DRIFT_DETECTION = True
    PERFORMANCE_MONITORING = True
    
    DRIFT_THRESHOLD = 0.05
    SIGNIFICANCE_LEVEL = 0.05
    WINDOW_SIZE = 100
    
    BASELINE_ACCURACY = 0.95
    PERFORMANCE_THRESHOLD = 0.05
    
    ALERT_ENABLED = True
    ALERT_EMAIL = ""
    ALERT_WEBHOOK = ""
    
    MONITORING_INTERVAL = 300
    STATE_SAVE_INTERVAL = 3600


class DockerConfig:
    IMAGE_NAME = "yolo-process-detection"
    TAG = "latest"
    
    CONTAINER_NAME = "yolo_process_detection"
    
    GPU_ENABLED = True
    GPU_MEMORY_LIMIT = "8g"
    CPU_LIMIT = "4.0"
    MEMORY_LIMIT = "8g"
    
    LOG_ROTATION = True
    MAX_LOG_SIZE = "10m"
    MAX_LOG_FILES = 3
    
    HEALTH_CHECK_ENABLED = True
    HEALTH_CHECK_INTERVAL = 30
    HEALTH_CHECK_TIMEOUT = 10
    HEALTH_CHECK_RETRIES = 3


class AdvancedTrainingConfig:
    ENABLE_ADVANCED_FEATURES = True
    
    MULTI_STAGE_TRAINING = False
    STAGE1_EPOCHS = 50
    STAGE2_EPOCHS = 100
    
    AUTO_MIXED_PRECISION = True
    GRADIENT_ACCUMULATION_STEPS = 1
    
    EARLY_STOPPING_PATIENCE = 20
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    SAVE_CHECKPOINT_INTERVAL = 10
    KEEP_BEST_ONLY = True
    
    VALIDATION_INTERVAL = 1
    LOG_INTERVAL = 10


class ProductionConfig:
    ENVIRONMENT = "production"
    DEBUG = False
    
    ENABLE_MONITORING = True
    ENABLE_METRICS = True
    ENABLE_TRACING = False
    
    MODEL_VERSION = "1.0.0"
    DEPLOYMENT_ID = ""
    
    ROLLBACK_ENABLED = True
    ROLLBACK_THRESHOLD = 0.95
    
    LOAD_BALANCING_ENABLED = False
    SCALING_ENABLED = False
    MIN_INSTANCES = 1
    MAX_INSTANCES = 10


def get_config(environment: str = "production") -> ProcessDetectionConfig:
    if environment == "development":
        config = ProcessDetectionConfig()
        config.DEVICE = "cpu"
        config.FPS_LIMIT = 10
        config.SAVE_RESULTS = False
        return config
    elif environment == "testing":
        config = ProcessDetectionConfig()
        config.DEVICE = "cpu"
        config.MAX_DETECTIONS = 50
        config.CONFIDENCE_THRESHOLD = 0.3
        return config
    else:
        return ProcessDetectionConfig()


ENV = os.getenv("ENVIRONMENT", "production")
CONFIG = get_config(ENV)