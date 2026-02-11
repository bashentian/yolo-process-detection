"""检测器模块

提供基于YOLO的目标检测功能。
"""
from pathlib import Path
from typing import NamedTuple
from dataclasses import dataclass
from typing import Protocol


class Detection(NamedTuple):
    """检测结果"""
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class ProcessDetector(Protocol):
    """检测器协议"""
    
    def detect(self, frame) -> list[Detection]:
        ...
    
    def __call__(self, frame) -> list[Detection]:
        ...


@dataclass
class YOLODetector:
    """YOLO检测器实现
    
    基于Ultralytics YOLO的目标检测器。
    """
    
    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self._model = None
    
    @property
    def model(self):
        """懒加载模型"""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self) -> None:
        """加载YOLO模型"""
        from ultralytics import YOLO
        
        # 检查本地模型
        models_dir = Path(__file__).parent.parent.parent / "models"
        model_path = models_dir / self.model_name
        
        if model_path.exists():
            self._model = YOLO(str(model_path))
        else:
            self._model = YOLO(self.model_name)
    
    def detect(self, frame) -> list[Detection]:
        """检测图像中的对象"""
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold
        )
        
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # 过滤低置信度
                    if conf >= self.confidence_threshold:
                        detections.append(Detection(
                            bbox=(float(x1), float(y1), float(x2), float(y2)),
                            confidence=conf,
                            class_id=cls,
                            class_name=result.names.get(cls, f"class_{cls}")
                        ))
        
        return detections
    
    def __call__(self, frame) -> list[Detection]:
        """使检测器可调用"""
        return self.detect(frame)
    
    def info(self) -> dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device
        }
