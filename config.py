import os
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_ROOT = PROJECT_ROOT / "models"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"

class ProcessDetectionConfig:
    MODEL_NAME = "yolov8n.pt"
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45
    MAX_DETECTIONS = 100
    
    DEVICE = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
    
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
