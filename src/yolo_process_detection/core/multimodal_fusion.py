"""多模态融合系统

整合视觉、音频、力传感器等多源数据。
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from enum import Enum
import numpy as np
from datetime import datetime


class SensorType(Enum):
    """传感器类型枚举"""
    VISUAL = "visual"
    AUDIO = "audio"
    TORQUE = "torque"
    VIBRATION = "vibration"
    GAZE = "gaze"


@dataclass
class SensorData:
    """传感器数据"""
    sensor_type: SensorType
    timestamp: float
    data: Any
    confidence: float = 0.0


@dataclass
class FusionResult:
    """融合结果"""
    fused_action: Optional[str] = None
    fused_confidence: float = 0.0
    sensor_contributions: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MultimodalFusion:
    """多模态融合器"""
    
    def __init__(self):
        self.sensor_weights = {
            SensorType.VISUAL: 0.6,
            SensorType.AUDIO: 0.2,
            SensorType.TORQUE: 0.15,
            SensorType.VIBRATION: 0.05
        }
        self.confidence_threshold = 0.7
    
    def fuse_sensors(
        self,
        sensor_data: List[SensorData]
    ) -> FusionResult:
        """融合多传感器数据
        
        Args:
            sensor_data: 传感器数据列表
            
        Returns:
            FusionResult: 融合结果
        """
        if not sensor_data:
            return FusionResult()
        
        # 按类型分组
        data_by_type = {}
        for data in sensor_data:
            if data.sensor_type not in data_by_type:
                data_by_type[data.sensor_type] = []
            data_by_type[data.sensor_type].append(data)
        
        # 计算加权置信度
        total_weight = 0.0
        weighted_confidences = {}
        
        for sensor_type, data_list in data_by_type.items():
            if not data_list:
                continue
            
            weight = self.sensor_weights.get(sensor_type, 0.0)
            avg_confidence = np.mean([d.confidence for d in data_list])
            weighted_confidence = avg_confidence * weight
            weighted_confidences[sensor_type.value] = weighted_confidence
            total_weight += weight
        
        # 归一化
        if total_weight > 0:
            for sensor_type in weighted_confidences:
                weighted_confidences[sensor_type] /= total_weight
        
        # 计算最终融合置信度
        final_confidence = sum(weighted_confidences.values())
        
        # 决策逻辑
        fused_action = self._make_decision(data_by_type, weighted_confidences)
        
        return FusionResult(
            fused_action=fused_action,
            fused_confidence=final_confidence,
            sensor_contributions=weighted_confidences
        )
    
    def _make_decision(
        self,
        data_by_type: Dict[SensorType, List[SensorData]],
        weighted_confidences: Dict[str, float]
    ) -> Optional[str]:
        """基于融合结果做出决策"""
        # 简化决策：选择置信度最高的传感器结果
        best_sensor = max(weighted_confidences.items(), key=lambda x: x[1])
        best_type = best_sensor[0]
        
        if best_type == SensorType.VISUAL and data_by_type[SensorType.VISUAL]:
            return data_by_type[SensorType.VISUAL][-1].data.get('action')
        
        return None


class TemporalFusion:
    """时序融合器"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history: List[Dict[str, Any]] = []
    
    def update(self, result: FusionResult) -> None:
        """更新时序历史"""
        self.history.append({
            'timestamp': result.timestamp,
            'action': result.fused_action,
            'confidence': result.fused_confidence,
            'contributions': result.sensor_contributions
        })
        
        if len(self.history) > self.window_size:
            self.history.pop(0)
    
    def get_temporal_context(self) -> Dict[str, Any]:
        """获取时序上下文"""
        if len(self.history) < 3:
            return {}
        
        recent = self.history[-3:]
        
        # 统计最近动作
        action_counts = {}
        for item in recent:
            action = item['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # 找出最频繁的动作
        if action_counts:
            most_common_action = max(action_counts.items(), key=lambda x: x[1])[0]
            most_common_count = action_counts[most_common_action]
            
            return {
                'most_common_action': most_common_action,
                'most_common_count': most_common_count,
                'total_count': len(recent),
                'action_distribution': action_counts
            }
        
        return {}


class AttentionFusion:
    """注意力融合机制"""
    
    def __init__(self):
        self.attention_weights = {
            'visual': 0.5,
            'audio': 0.3,
            'torque': 0.2
        }
    
    def compute_attention_weights(
        self,
        visual_confidence: float,
        audio_confidence: float,
        torque_confidence: float
    ) -> Dict[str, float]:
        """计算注意力权重"""
        weights = {}
        
        # 基于置信度动态调整权重
        total_confidence = visual_confidence + audio_confidence + torque_confidence
        
        if total_confidence > 0:
            weights['visual'] = (
                self.attention_weights['visual'] * visual_confidence / total_confidence
            )
            weights['audio'] = (
                self.attention_weights['audio'] * audio_confidence / total_confidence
            )
            weights['torque'] = (
                self.attention_weights['torque'] * torque_confidence / total_confidence
            )
        
        return weights
    
    def fuse_with_attention(
        self,
        visual_result: Any,
        audio_result: Optional[Any] = None,
        torque_result: Optional[Any] = None
    ) -> FusionResult:
        """使用注意力机制融合"""
        sensor_contributions = {}
        
        # 视觉结果
        if visual_result is not None:
            sensor_contributions['visual'] = 1.0
        
        # 音频结果
        if audio_result is not None:
            sensor_contributions['audio'] = 1.0
        
        # 扭矩结果
        if torque_result is not None:
            sensor_contributions['torque'] = 1.0
        
        # 计算注意力权重
        if sensor_contributions:
            weights = self.compute_attention_weights(
                sensor_contributions.get('visual', 0.0),
                sensor_contributions.get('audio', 0.0),
                sensor_contributions.get('torque', 0.0)
            )
            
            # 加权融合
            final_confidence = sum(
                weights[key] * sensor_contributions[key]
                for key in sensor_contributions
            )
        
        return FusionResult(
            fused_action=visual_result.get('action') if visual_result else None,
            fused_confidence=final_confidence,
            sensor_contributions=sensor_contributions
        )


class MultiModalService:
    """多模态融合服务"""
    
    def __init__(
        self,
        fusion: MultimodalFusion,
        temporal: TemporalFusion,
        attention: AttentionFusion
    ):
        self.fusion = fusion
        self.temporal = temporal
        self.attention = attention
        self.fusion_history: List[FusionResult] = []
    
    def process_frame(
        self,
        frame: np.ndarray,
        visual_data: Optional[Any] = None,
        audio_data: Optional[np.ndarray] = None,
        torque_data: Optional[float] = None,
        vibration_data: Optional[np.ndarray] = None
    ) -> FusionResult:
        """处理一帧的多模态数据"""
        sensor_data = []
        
        # 视觉数据
        if visual_data is not None:
            sensor_data.append(SensorData(
                sensor_type=SensorType.VISUAL,
                timestamp=datetime.now().timestamp(),
                data=visual_data,
                confidence=visual_data.get('confidence', 0.8)
            ))
        
        # 音频数据
        if audio_data is not None:
            sensor_data.append(SensorData(
                sensor_type=SensorType.AUDIO,
                timestamp=datetime.now().timestamp(),
                data=audio_data,
                confidence=0.7
            ))
        
        # 扭矩数据
        if torque_data is not None:
            sensor_data.append(SensorData(
                sensor_type=SensorType.TORQUE,
                timestamp=datetime.now().timestamp(),
                data=torque_data,
                confidence=0.9
            ))
        
        # 振动数据
        if vibration_data is not None:
            sensor_data.append(SensorData(
                sensor_type=SensorType.VIBRATION,
                timestamp=datetime.now().timestamp(),
                data=vibration_data,
                confidence=0.6
            ))
        
        # 多模态融合
        fusion_result = self.fusion.fuse_sensors(sensor_data)
        
        # 时序融合
        self.temporal.update(fusion_result)
        
        # 注意力融合（如果有多传感器）
        if len(sensor_data) > 1:
            fusion_result = self.attention.fuse_with_attention(
                visual_data,
                audio_data,
                torque_data
            )
        
        # 记录历史
        self.fusion_history.append(fusion_result)
        
        return fusion_result
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取融合统计"""
        if not self.fusion_history:
            return {}
        
        total_fusions = len(self.fusion_history)
        
        # 按动作统计
        action_counts = {}
        for result in self.fusion_history:
            action = result.fused_action
            if action:
                action_counts[action] = action_counts.get(action, 0) + 1
        
        # 按传感器统计
        sensor_usage = {}
        for result in self.fusion_history:
            for sensor_type, contribution in result.sensor_contributions.items():
                if contribution > 0:
                    sensor_usage[sensor_type] = sensor_usage.get(sensor_type, 0) + 1
        
        # 平均置信度
        avg_confidence = np.mean([r.fused_confidence for r in self.fusion_history])
        
        return {
            'total_fusions': total_fusions,
            'action_distribution': action_counts,
            'sensor_usage': sensor_usage,
            'avg_confidence': avg_confidence
        }
