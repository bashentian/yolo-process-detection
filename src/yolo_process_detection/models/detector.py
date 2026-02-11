"""检测器模块

提供基于YOLO的目标检测功能，集成场景理解、异常检测和效率分析等高级特性。
"""
from pathlib import Path
from typing import NamedTuple, List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from typing import Protocol
from collections import deque
import time
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

from ..core.config import get_settings


logger = logging.getLogger(__name__)


class RegionalAttention(nn.Module):
    """
    区域注意力模块 (YOLOv12核心创新)
    实现区域级别的注意力机制，提升特征提取能力
    """
    
    def __init__(self, dim, num_heads=8, reduction=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.reduction = reduction
        
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj_out = nn.Conv2d(dim, dim, 1)
        
        # 区域池化
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(dim, dim // reduction)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 计算Q, K, V
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        # 多头注意力
        q = q * self.scale
        attn = torch.einsum('bchw,bciw->bhwj', q, k)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhwj,bciw->bchw', attn, v)
        
        # 区域特征
        regional_feat = self.pool(x)
        regional_feat = self.fc(regional_feat.flatten(1)).view(B, -1, 1, 1)
        regional_feat = regional_feat.expand(-1, -1, H, W)
        
        # 融合注意力和区域特征
        combined = out + regional_feat
        
        return self.proj_out(combined)


class SceneUnderstanding:
    """
    场景理解模块
    基于检测结果进行工序阶段识别和场景分析
    """
    
    def __init__(self):
        self.process_stages = [
            'idle',
            'preparation',
            'processing',
            'assembly',
            'quality_check',
            'packaging'
        ]
        
        # 工序关键词映射
        self.stage_keywords = {
            'idle': [],
            'preparation': ['material', 'worker', 'preparation'],
            'processing': ['machine', 'processing', 'worker'],
            'assembly': ['product', 'assembly', 'tool'],
            'quality_check': ['quality', 'check', 'test'],
            'packaging': ['package', 'box', 'product']
        }
    
    def identify_stage(self, detections: List['Detection'], frame: np.ndarray) -> Dict:
        """
        基于检测到的对象识别当前工序阶段
        """
        if not detections:
            return {
                'stage': 'idle',
                'confidence': 0.0,
                'context': 'No objects detected',
                'scene_features': {
                    'object_count': 0,
                    'unique_classes': 0,
                    'density': 0.0,
                    'class_distribution': {}
                }
            }
        
        # 提取类别信息
        class_names = [det.class_name for det in detections]
        
        # 统计各类别数量
        class_counts = {}
        for name in class_names:
            class_counts[name] = class_counts.get(name, 0) + 1
        
        # 场景特征
        scene_features = {
            'object_count': len(detections),
            'unique_classes': len(set(class_names)),
            'density': len(detections) / (frame.shape[0] * frame.shape[1]) if len(frame.shape) >= 2 else 0,
            'class_distribution': class_counts
        }
        
        # 识别工序阶段
        stage_scores = {}
        for stage, keywords in self.stage_keywords.items():
            score = sum(class_counts.get(kw, 0) for kw in keywords)
            stage_scores[stage] = score / len(keywords) if keywords else 0
        
        predicted_stage = max(stage_scores, key=stage_scores.get) if stage_scores else 'idle'
        confidence = stage_scores[predicted_stage] if stage_scores else 0.0
        
        # 生成上下文描述
        context = self._generate_context(class_names, scene_features)
        
        return {
            'stage': predicted_stage,
            'confidence': confidence,
            'context': context,
            'scene_features': scene_features
        }
    
    def _generate_context(self, class_names: List[str], features: Dict) -> str:
        """
        生成场景上下文描述
        """
        if not class_names:
            return "Empty scene"
        
        # 基于检测对象生成描述
        context_parts = []
        
        if 'worker' in class_names:
            context_parts.append("检测到工作人员")
        if 'machine' in class_names:
            context_parts.append("检测到生产设备")
        if 'product' in class_names:
            context_parts.append("检测到产品")
        if 'material' in class_names:
            context_parts.append("检测到原材料")
        
        if context_parts:
            return "、".join(context_parts)
        else:
            return f"检测到{len(set(class_names))}类对象"


class AnomalyDetection:
    """
    异常检测模块
    基于历史数据检测异常行为或状态
    """
    
    def __init__(self, history_size=100):
        self.history = deque(maxlen=history_size)
        self.history_size = history_size
        
        # 正常模式阈值
        self.normal_ranges = {
            'object_count': (1, 50),
            'confidence': (0.3, 1.0),
            'processing_time': (0.01, 2.0)
        }
    
    def update(self, detections: List['Detection'], processing_time: float):
        """
        更新历史数据
        """
        features = self._extract_features(detections, processing_time)
        self.history.append(features)
    
    def detect(self, current_detections: List['Detection'], processing_time: float) -> Dict:
        """
        检测异常
        """
        if len(self.history) < 10:
            return {
                'is_anomaly': False,
                'score': 0.0,
                'type': None,
                'reasons': ['Insufficient data']
            }
        
        # 提取当前特征
        current_features = self._extract_features(current_detections, processing_time)
        
        # 计算异常分数
        anomaly_score = 0.0
        anomaly_reasons = []
        
        # 检测对象数量异常
        obj_count = current_features['object_count']
        if not (self.normal_ranges['object_count'][0] <= obj_count <= self.normal_ranges['object_count'][1]):
            anomaly_score += 0.3
            anomaly_reasons.append(f"对象数量异常: {obj_count}")
        
        # 检测置信度异常
        avg_conf = current_features['avg_confidence']
        if not (self.normal_ranges['confidence'][0] <= avg_conf <= self.normal_ranges['confidence'][1]):
            anomaly_score += 0.2
            anomaly_reasons.append(f"置信度异常: {avg_conf:.2f}")
        
        # 检测处理时间异常
        if not (self.normal_ranges['processing_time'][0] <= processing_time <= self.normal_ranges['processing_time'][1]):
            anomaly_score += 0.3
            anomaly_reasons.append(f"处理时间异常: {processing_time:.3f}s")
        
        # 检测模式异常
        if len(self.history) > 0:
            historical_avg = np.mean([h['object_count'] for h in self.history])
            if abs(obj_count - historical_avg) > historical_avg * 0.5:
                anomaly_score += 0.2
                anomaly_reasons.append(f"对象数量偏离历史均值")
        
        # 判断是否异常
        is_anomaly = anomaly_score >= 0.5
        anomaly_type = self._classify_anomaly(anomaly_reasons) if is_anomaly else None
        
        return {
            'is_anomaly': is_anomaly,
            'score': anomaly_score,
            'type': anomaly_type,
            'reasons': anomaly_reasons
        }
    
    def _extract_features(self, detections: List['Detection'], processing_time: float) -> Dict:
        """
        提取特征
        """
        if not detections:
            return {
                'object_count': 0,
                'avg_confidence': 0.0,
                'processing_time': processing_time
            }
        
        confidences = [det.confidence for det in detections]
        
        return {
            'object_count': len(detections),
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'processing_time': processing_time
        }
    
    def _classify_anomaly(self, reasons: List[str]) -> Optional[str]:
        """
        分类异常类型
        """
        if not reasons:
            return None
        
        if '对象数量' in reasons[0]:
            return 'object_count'
        elif '置信度' in reasons[0]:
            return 'confidence'
        elif '处理时间' in reasons[0]:
            return 'performance'
        elif '偏离' in reasons[0]:
            return 'pattern'
        else:
            return 'unknown'


class EfficiencyAnalyzer:
    """
    效率分析模块
    分析处理效率并提供优化建议
    """
    
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.metrics = deque(maxlen=window_size)
        
        # 效率基准
        self.benchmarks = {
            'throughput': 100,  # 对象/秒
            'latency': 0.1,    # 秒
            'accuracy': 0.95      # 置信度
        }
    
    def update(self, detections: List['Detection'], processing_time: float):
        """
        更新效率指标
        """
        if not detections:
            return
        
        throughput = len(detections) / processing_time if processing_time > 0 else 0
        avg_confidence = np.mean([det.confidence for det in detections]) if detections else 0.0
        
        self.metrics.append({
            'throughput': throughput,
            'latency': processing_time / len(detections) if detections else 0,
            'accuracy': avg_confidence,
            'timestamp': time.time()
        })
    
    def analyze(self) -> Dict:
        """
        分析效率并提供建议
        """
        if len(self.metrics) < 10:
            return {
                'score': 0.5,
                'status': 'collecting',
                'recommendations': ['需要更多数据'],
                'throughput': 0,
                'latency': 0,
                'accuracy': 0,
                'trend': 'stable'
            }
        
        # 计算平均值
        recent_metrics = list(self.metrics)[-self.window_size:]
        avg_throughput = np.mean([m['throughput'] for m in recent_metrics])
        avg_latency = np.mean([m['latency'] for m in recent_metrics])
        avg_accuracy = np.mean([m['accuracy'] for m in recent_metrics])
        
        # 计算效率分数
        throughput_score = min(avg_throughput / self.benchmarks['throughput'], 1.0)
        latency_score = self.benchmarks['latency'] / avg_latency if avg_latency > 0 else 1.0
        accuracy_score = avg_accuracy / self.benchmarks['accuracy']
        
        efficiency_score = (throughput_score + latency_score + accuracy_score) / 3
        
        # 生成建议
        recommendations = []
        
        if throughput_score < 0.8:
            recommendations.append("考虑优化模型推理速度")
        if latency_score < 0.8:
            recommendations.append("考虑使用更快的硬件或量化模型")
        if accuracy_score < 0.9:
            recommendations.append("考虑调整置信度阈值或重新训练模型")
        
        # 趋势分析
        trend = self._analyze_trend(recent_metrics)
        
        return {
            'score': efficiency_score,
            'status': self._get_status(efficiency_score),
            'throughput': avg_throughput,
            'latency': avg_latency,
            'accuracy': avg_accuracy,
            'trend': trend,
            'recommendations': recommendations
        }
    
    def _analyze_trend(self, metrics: List[Dict]) -> str:
        """
        分析趋势
        """
        if len(metrics) < 5:
            return 'stable'
        
        recent = metrics[-5:]
        avg_throughput = np.mean([m['throughput'] for m in recent])
        
        # 检查趋势
        increasing = all(m['throughput'] >= avg_throughput for m in recent)
        decreasing = all(m['throughput'] <= avg_throughput for m in recent)
        
        if increasing:
            return 'improving'
        elif decreasing:
            return 'declining'
        else:
            return 'stable'
    
    def _get_status(self, score: float) -> str:
        """
        获取状态描述
        """
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.7:
            return 'good'
        elif score >= 0.5:
            return 'fair'
        else:
            return 'poor'


@dataclass
class Detection:
    """检测结果"""
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


class ProcessDetectorProtocol(Protocol):
    """检测器协议"""
    
    def detect(self, frame) -> list[Detection]:
        ...
    
    def __call__(self, frame) -> list[Detection]:
        ...


class YOLODetector:
    """YOLO检测器实现
    
    基于Ultralytics YOLO的目标检测器，集成高级分析功能。
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        device: Optional[str] = None
    ):
        settings = get_settings()
        self.model_name = model_name or settings.model_name
        self.confidence_threshold = confidence_threshold or settings.confidence_threshold
        self.iou_threshold = iou_threshold or settings.iou_threshold
        self.device = device or settings.device
        self._model = None
        
        # 初始化高级分析模块
        self.scene_understanding = SceneUnderstanding()
        self.anomaly_detector = AnomalyDetection(
            history_size=settings.anomaly_history_size
        )
        self.efficiency_analyzer = EfficiencyAnalyzer(
            window_size=settings.efficiency_window_size
        )
        
        # 初始化注意力模块 (如果启用)
        self.use_attention = settings.use_attention
        if self.use_attention:
            try:
                self.attention_module = RegionalAttention(dim=256)
                logger.info("Regional Attention module initialized")
            except Exception as e:
                logger.error(f"Failed to initialize attention module: {e}")
                self.use_attention = False
    
    @property
    def model(self):
        """懒加载模型"""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self) -> None:
        """加载YOLO模型"""
        settings = get_settings()
        
        # 检查本地模型
        models_dir = settings.models_root
        model_path = models_dir / self.model_name
        
        logger.info(f"Loading model from {model_path} or {self.model_name}")
        
        if model_path.exists():
            self._model = YOLO(str(model_path))
        else:
            self._model = YOLO(self.model_name)
            
        # 设置模型参数 (如果模型支持)
        # 注意: ultralytics YOLO模型在推理时传递参数，而不是设置属性
        # 但我们可以在这里做一些预热或配置
    
    def detect(self, frame) -> list[Detection]:
        """检测图像中的对象"""
        settings = get_settings()
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # 过滤低置信度 (虽然模型推理时已经过滤，但这里作为双重检查)
                    if conf >= self.confidence_threshold:
                        detections.append(Detection(
                            bbox=(float(x1), float(y1), float(x2), float(y2)),
                            confidence=conf,
                            class_id=cls,
                            class_name=result.names.get(cls, f"class_{cls}")
                        ))
        
        return detections
    
    def detect_advanced(self, frame: np.ndarray) -> Dict:
        """
        高级检测方法
        集成场景理解、异常检测和效率分析
        """
        start_time = time.time()
        
        # 标准检测
        detections = self.detect(frame)
        processing_time = time.time() - start_time
        
        # 场景理解
        scene_info = self.scene_understanding.identify_stage(detections, frame)
        
        # 异常检测
        anomaly_info = self.anomaly_detector.detect(detections, processing_time)
        
        # 效率分析
        self.efficiency_analyzer.update(detections, processing_time)
        efficiency_info = self.efficiency_analyzer.analyze()
        
        # 更新历史
        self.anomaly_detector.update(detections, processing_time)
        
        return {
            'detections': detections,
            'scene': scene_info,
            'anomaly': anomaly_info,
            'efficiency': efficiency_info,
            'processing_time': processing_time
        }
    
    def analyze_process_stage(self, detections: List[Detection]) -> str:
        """分析工序阶段 (简单版，保留兼容性)"""
        return self.scene_understanding.identify_stage(detections, np.array([]))['stage']
        
    def __call__(self, frame) -> list[Detection]:
        """使检测器可调用"""
        return self.detect(frame)
    
    def info(self) -> dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
            "features": {
                "scene_understanding": True,
                "anomaly_detection": True,
                "efficiency_analysis": True,
                "attention": self.use_attention
            }
        }
