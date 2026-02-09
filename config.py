import os
from pathlib import Path
from typing import List, Dict, Optional

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_ROOT = PROJECT_ROOT / "models"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
LOGS_ROOT = PROJECT_ROOT / "logs"
UPLOADS_ROOT = PROJECT_ROOT / "uploads"
CACHE_ROOT = PROJECT_ROOT / "cache"


class ProcessDetectionConfig:
    MODEL_NAME = os.getenv("MODEL_NAME", "yolo11n.pt")
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
    MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", "100"))
    
    DEVICE = os.getenv("DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu")
    
    VIDEO_SOURCE = 0
    
    FRAME_RESIZE = (640, 640)
    DISPLAY_SIZE = (1280, 720)
    
    FPS_LIMIT = 30
    
    SAVE_RESULTS = True
    OUTPUT_FORMAT = "mp4"
    
    CLASS_NAMES: Dict[int, str] = {
        0: "worker",
        1: "machine",
        2: "product",
        3: "tool",
        4: "material"
    }
    
    PROCESS_STAGES: List[str] = [
        "preparation",
        "processing",
        "assembly",
        "quality_check",
        "packaging"
    ]
    
    TRACKING_ENABLED = True
    TRACKING_MAX_AGE = 30
    TRACKING_MIN_HITS = 3
    
    # 新增：YOLOv12高级功能参数
    USE_ATTENTION = os.getenv("USE_ATTENTION", "False").lower() in ("true", "1", "yes")
    ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "0.5"))
    ANOMALY_HISTORY_SIZE = int(os.getenv("ANOMALY_HISTORY_SIZE", "100"))
    EFFICIENCY_WINDOW_SIZE = int(os.getenv("EFFICIENCY_WINDOW_SIZE", "50"))
    SCENE_UNDERSTANDING_ENABLED = os.getenv("SCENE_UNDERSTANDING_ENABLED", "True").lower() in ("true", "1", "yes")


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