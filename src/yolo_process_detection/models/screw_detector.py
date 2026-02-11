"""螺钉安装检测模块

识别螺钉位置准确性和紧固状态。
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np
from datetime import datetime


class ScrewStatus(Enum):
    """螺钉状态枚举"""
    MISSING = "missing"
    POSITIONED = "positioned"
    TIGHTENED = "tightened"
    OVERTIGHTENED = "overtightened"
    UNDETECTED = "undetected"


@dataclass
class ScrewPosition:
    """螺钉位置信息"""
    x: float
    y: float
    width: float
    height: float
    confidence: float
    is_standard: bool = True


@dataclass
class ScrewDetectionResult:
    """螺钉检测结果"""
    screw_id: Optional[int] = None
    position: Optional[ScrewPosition] = None
    status: ScrewStatus
    deviation: float = 0.0
    deviation_x: float = 0.0
    deviation_y: float = 0.0
    confidence: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TighteningStatus:
    """紧固状态信息"""
    is_tightening: bool
    torque_value: float = 0.0
    torque_target: float = 0.0
    tightening_duration: float = 0.0
    rotation_count: int = 0
    sound_detected: bool = False


class ScrewDetector:
    """螺钉检测器"""
    
    def __init__(self, standard_positions: List[Dict[str, float]]):
        self.standard_positions = standard_positions
        self.position_tolerance = 2.0
        self.confidence_threshold = 0.95
    
    def detect_screw(
        self,
        frame: np.ndarray,
        bbox: List[float]
    ) -> ScrewDetectionResult:
        """检测螺钉
        
        Args:
            frame: 输入图像帧
            bbox: 边界框 [x_min, y_min, x_max, y_max]
            
        Returns:
            ScrewDetectionResult: 检测结果
        """
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        
        # 检查是否为螺钉形状（近似圆形）
        aspect_ratio = width / height if height > 0 else 0
        is_screw_shape = 0.7 <= aspect_ratio <= 1.3
        
        # 计算中心点
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # 与标准位置对比
        position = ScrewPosition(
            x=center_x,
            y=center_y,
            width=width,
            height=height,
            confidence=0.8 if is_screw_shape else 0.6
        )
        
        # 计算偏差
        deviation = self._calculate_deviation(position)
        
        # 判断状态
        status = ScrewStatus.UNDETECTED
        if is_screw_shape and position.confidence > self.confidence_threshold:
            if deviation <= self.position_tolerance:
                status = ScrewStatus.POSITIONED
            elif deviation > self.position_tolerance * 2:
                status = ScrewStatus.OVERTIGHTENED
            else:
                status = ScrewStatus.TIGHTENED
        
        return ScrewDetectionResult(
            position=position,
            status=status,
            deviation=deviation,
            deviation_x=deviation[0],
            deviation_y=deviation[1],
            confidence=position.confidence
        )
    
    def _calculate_deviation(self, position: ScrewPosition) -> Tuple[float, float]:
        """计算与标准位置的偏差"""
        deviations = []
        
        for std_pos in self.standard_positions:
            dx = position.x - std_pos['x']
            dy = position.y - std_pos['y']
            distance = np.sqrt(dx**2 + dy**2)
            deviations.append(distance)
        
        if deviations:
            avg_deviation = np.mean(deviations)
            min_deviation = np.min(deviations)
            return avg_deviation, min_deviation
        return 0.0, 0.0


class TorqueAnalyzer:
    """扭矩分析器"""
    
    def __init__(self, target_torque: float, tolerance: float = 0.1):
        self.target_torque = target_torque
        self.tolerance = tolerance
        self.history: List[float] = []
    
    def analyze(self, torque_value: float) -> Dict[str, Any]:
        """分析扭矩值
        
        Args:
            torque_value: 当前扭矩值
            
        Returns:
            dict: 分析结果
        """
        self.history.append(torque_value)
        
        # 计算偏差
        deviation = abs(torque_value - self.target_torque)
        deviation_ratio = deviation / self.target_torque if self.target_torque > 0 else 0
        
        # 判断状态
        is_adequate = deviation <= self.tolerance * self.target_torque
        is_insufficient = deviation < -self.tolerance * self.target_torque
        is_excessive = deviation > self.tolerance * self.target_torque * 2
        
        status = "adequate"
        if is_insufficient:
            status = "insufficient"
        elif is_excessive:
            status = "excessive"
        
        return {
            'torque_value': torque_value,
            'target_torque': self.target_torque,
            'deviation': deviation,
            'deviation_ratio': deviation_ratio,
            'status': status,
            'is_adequate': is_adequate,
            'is_insufficient': is_insufficient,
            'is_excessive': is_excessive
        }


class SoundAnalyzer:
    """声音分析器"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.target_frequency = 2000  # 2kHz咔哒声
        self.frequency_band = (1800, 2200)
    
    def analyze_click_sound(self, audio_data: np.ndarray) -> bool:
        """分析是否检测到咔哒声
        
        Args:
            audio_data: 音频数据
            
        Returns:
            bool: 是否检测到咔哒声
        """
        # FFT变换
        fft_result = np.fft.fft(audio_data)
        freqs = np.fft.fftfreq(fft_result, 1.0 / len(audio_data) * self.sample_rate)
        magnitude = np.abs(fft_result)
        
        # 检查目标频率范围内的能量
        target_band_energy = np.sum(
            magnitude[(freqs >= self.frequency_band[0]) & (freqs <= self.frequency_band[1])]
        )
        total_energy = np.sum(magnitude)
        
        # 计算能量比
        energy_ratio = target_band_energy / total_energy if total_energy > 0 else 0
        
        # 阈值判断
        return energy_ratio > 0.3


class VibrationAnalyzer:
    """振动分析器"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def analyze(self, vibration_data: np.ndarray) -> Dict[str, Any]:
        """分析振动数据
        
        Args:
            vibration_data: 振动数据
            
        Returns:
            dict: 分析结果
        """
        # 计算RMS值
        rms = np.sqrt(np.mean(vibration_data**2))
        
        # 计算峰值
        peak = np.max(np.abs(vibration_data))
        
        # 判断状态
        is_normal = rms < self.threshold
        is_excessive = rms > self.threshold * 2
        
        status = "normal"
        if is_excessive:
            status = "excessive"
        
        return {
            'rms': rms,
            'peak': peak,
            'status': status,
            'is_normal': is_normal,
            'is_excessive': is_excessive
        }


class ScrewInspectionSystem:
    """螺钉检测系统"""
    
    def __init__(
        self,
        standard_positions: List[Dict[str, float]],
        target_torque: float = 100.0
    ):
        self.detector = ScrewDetector(standard_positions)
        self.torque_analyzer = TorqueAnalyzer(target_torque)
        self.sound_analyzer = SoundAnalyzer()
        self.vibration_analyzer = VibrationAnalyzer()
        self.history: List[ScrewDetectionResult] = []
    
    def process_frame(
        self,
        frame: np.ndarray,
        screw_bbox: List[float],
        torque_data: Optional[float] = None,
        audio_data: Optional[np.ndarray] = None,
        vibration_data: Optional[np.ndarray] = None
    ) -> ScrewDetectionResult:
        """处理一帧数据
        
        Args:
            frame: 图像帧
            screw_bbox: 螺钉边界框
            torque_data: 扭矩数据
            audio_data: 音频数据
            vibration_data: 振动数据
            
        Returns:
            ScrewDetectionResult: 综合检测结果
        """
        result = self.detector.detect_screw(frame, screw_bbox)
        
        # 分析扭矩
        if torque_data is not None:
            torque_analysis = self.torque_analyzer.analyze(torque_data)
            result.torque_value = torque_data
            result.torque_target = self.torque_analyzer.target_torque
        else:
            torque_analysis = None
        
        # 分析声音
        if audio_data is not None:
            sound_detected = self.sound_analyzer.analyze_click_sound(audio_data)
            result.sound_detected = sound_detected
        else:
            result.sound_detected = False
        
        # 分析振动
        if vibration_data is not None:
            vibration_analysis = self.vibration_analyzer.analyze(vibration_data)
        else:
            vibration_analysis = None
        
        # 综合判断紧固状态
        if torque_analysis and sound_detected:
            if torque_analysis['is_adequate']:
                result.status = ScrewStatus.TIGHTENED
            elif torque_analysis['is_insufficient']:
                result.status = ScrewStatus.POSITIONED
        
        # 记录历史
        self.history.append(result)
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.history:
            return {}
        
        total_detections = len(self.history)
        positioned = sum(1 for r in self.history if r.status == ScrewStatus.POSITIONED)
        tightened = sum(1 for r in self.history if r.status == ScrewStatus.TIGHTENED)
        overtightened = sum(1 for r in self.history if r.status == ScrewStatus.OVERTIGHTENED)
        undetected = sum(1 for r in self.history if r.status == ScrewStatus.UNDETECTED)
        
        avg_confidence = np.mean([r.confidence for r in self.history])
        avg_deviation = np.mean([r.deviation for r in self.history])
        
        return {
            'total_detections': total_detections,
            'positioned': positioned,
            'tightened': tightened,
            'overtightened': overtightened,
            'undetected': undetected,
            'avg_confidence': avg_confidence,
            'avg_deviation': avg_deviation,
            'pass_rate': (positioned + tightened) / total_detections if total_detections > 0 else 0
        }
