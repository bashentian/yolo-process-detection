import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from dataclasses import dataclass

from config import ProcessDetectionConfig


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    
    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class ProcessDetector:
    def __init__(self, config: ProcessDetectionConfig):
        self.config = config
        self.model = self._load_model()
        self.process_history: List[Dict] = []
        
    def _load_model(self) -> YOLO:
        model_path = Path(__file__).parent.parent / "models" / self.config.MODEL_NAME
        
        if not model_path.exists():
            model = YOLO(self.config.MODEL_NAME)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.export(format="onnx")
        else:
            model = YOLO(str(model_path))
        
        return model
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(
            frame,
            conf=self.config.CONFIDENCE_THRESHOLD,
            iou=self.config.IOU_THRESHOLD,
            max_det=self.config.MAX_DETECTIONS,
            device=self.config.DEVICE,
            verbose=False
        )
        
        detections = []
        if len(results) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.config.CLASS_NAMES.get(class_id, f"class_{class_id}")
                
                detection = Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name
                )
                detections.append(detection)
        
        return detections
    
    def detect_frame(self, frame: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        original_frame = frame.copy()
        
        if self.config.FRAME_RESIZE:
            frame = cv2.resize(frame, self.config.FRAME_RESIZE)
        
        detections = self.detect(frame)
        
        return detections, frame
    
    def analyze_process_stage(self, detections: List[Detection]) -> str:
        if not detections:
            return "idle"
        
        class_counts = {}
        for det in detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
        
        if class_counts.get("worker", 0) > 0 and class_counts.get("machine", 0) > 0:
            if class_counts.get("product", 0) > 0:
                return "processing"
            elif class_counts.get("material", 0) > 0:
                return "preparation"
        
        if class_counts.get("product", 0) > 0 and class_counts.get("tool", 0) > 0:
            return "assembly"
        
        if class_counts.get("product", 0) > 0 and "worker" in class_counts:
            return "quality_check"
        
        if class_counts.get("product", 0) > 0 and "machine" not in class_counts:
            return "packaging"
        
        return "idle"
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection], 
                       current_stage: str = "") -> np.ndarray:
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            color = self._get_color_for_class(detection.class_id)
            
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                          color, 2)
            
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.5, 1)
            
            cv2.rectangle(annotated_frame, 
                         (int(x1), int(y1) - label_size[1] - 10),
                         (int(x1) + label_size[0], int(y1)),
                         color, -1)
            
            cv2.putText(annotated_frame, label,
                       (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if current_stage:
            stage_text = f"Current Stage: {current_stage}"
            cv2.putText(annotated_frame, stage_text,
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated_frame
    
    def _get_color_for_class(self, class_id: int) -> Tuple[int, int, int]:
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]
        return colors[class_id % len(colors)]
