"""姿势安全识别模块

检测不安全操作姿势，预防工伤。
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np
from datetime import datetime


class PostureRisk(Enum):
    """姿势风险等级"""
    NORMAL = "normal"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class PostureType(Enum):
    """姿势类型"""
    NORMAL = "normal"
    EXCESSIVE_BENDING = "excessive_bending"
    UNSAFE_SQUATTING = "unsafe_squatting"
    HAND_IN_DANGER_ZONE = "hand_in_danger_zone"
    HEAD_TOO_CLOSE = "head_too_close"
    IMPROPER_LIFTING = "improper_lifting"
    FATIGUE_DETECTED = "fatigue_detected"


@dataclass
class Keypoint:
    """关键点"""
    name: str
    x: float
    y: float
    confidence: float


@dataclass
class PostureAnalysis:
    """姿势分析结果"""
    posture_type: PostureType
    risk_level: PostureRisk
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class PostureEstimator:
    """姿势估计器"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle'
        ]
        
        # 风险阈值
        self.bending_angle_threshold = 45.0
        self.bending_duration_threshold = 10.0
        self.danger_zone_distance = 0.3
        self.head_distance_threshold = 0.3
        self.fatigue_blink_threshold = 25
        self.fatigue_action_time_threshold = 5.0
        self.fatigue_posture_threshold = -0.2
    
    def load_model(self, model_path: str) -> bool:
        """加载姿势估计模型"""
        try:
            # 这里应该加载实际的姿势估计模型
            # 例如: OpenPose, HRNet, MediaPipe
            self.model = f"Loaded model from {model_path}"
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def detect_posture(
        self,
        frame: np.ndarray,
        keypoints: Optional[List[Keypoint]] = None
    ) -> PostureAnalysis:
        """检测姿势
        
        Args:
            frame: 输入图像帧
            keypoints: 关键点列表（17个点）
            
        Returns:
            PostureAnalysis: 姿势分析结果
        """
        if keypoints is None:
            return PostureAnalysis(
                posture_type=PostureType.NORMAL,
                risk_level=PostureRisk.NORMAL,
                confidence=0.0,
                details={'error': 'no_keypoints'}
            )
        
        # 提取关键点
        nose = self._get_keypoint(keypoints, 'nose')
        left_shoulder = self._get_keypoint(keypoints, 'left_shoulder')
        right_shoulder = self._get_keypoint(keypoints, 'right_shoulder')
        left_hip = self._get_keypoint(keypoints, 'left_hip')
        right_hip = self._get_keypoint(keypoints, 'right_hip')
        left_knee = self._get_keypoint(keypoints, 'left_knee')
        right_knee = self._get_keypoint(keypoints, 'right_knee')
        
        # 检测弯腰
        bending_result = self._detect_bending(
            nose, left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee
        )
        
        # 检测踮脚
        squatting_result = self._detect_squatting(left_knee, right_knee)
        
        # 综合判断
        if bending_result['is_bending'] and not squatting_result['is_squatting']:
            return PostureAnalysis(
                posture_type=PostureType.EXCESSIVE_BENDING,
                risk_level=PostureRisk.HIGH,
                confidence=0.8,
                details={
                    'bending_angle': bending_result['angle'],
                    'bending_duration': bending_result.get('duration', 0),
                    'knee_bent': squatting_result['is_squatting']
                }
            )
        
        if squatting_result['is_squatting']:
            return PostureAnalysis(
                posture_type=PostureType.UNSAFE_SQUATTING,
                risk_level=PostureRisk.HIGH,
                confidence=0.85,
                details={
                    'knee_angles': squatting_result['angles']
                }
            )
        
        return PostureAnalysis(
            posture_type=PostureType.NORMAL,
            risk_level=PostureRisk.NORMAL,
            confidence=0.9,
            details={}
        )
    
    def _get_keypoint(self, keypoints: List[Keypoint], name: str) -> Optional[Keypoint]:
        """获取关键点"""
        for kp in keypoints:
            if kp.name == name:
                return kp
        return None
    
    def _detect_bending(
        self,
        nose: Keypoint,
        left_shoulder: Keypoint,
        right_shoulder: Keypoint,
        left_hip: Keypoint,
        right_hip: Keypoint,
        left_knee: Keypoint,
        right_knee: Keypoint
    ) -> Dict[str, Any]:
        """检测弯腰动作"""
        # 计算髋关节角度（躯干与大腿夹角）
        left_hip_angle = self._calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = self._calculate_angle(right_shoulder, right_hip, right_knee)
        avg_hip_angle = (left_hip_angle + right_hip_angle) / 2
        
        is_bending = avg_hip_angle < self.bending_angle_threshold
        
        return {
            'is_bending': is_bending,
            'left_hip_angle': left_hip_angle,
            'right_hip_angle': right_hip_angle,
            'avg_hip_angle': avg_hip_angle,
            'angle': avg_hip_angle
        }
    
    def _detect_squatting(
        self,
        left_knee: Keypoint,
        right_knee: Keypoint
    ) -> Dict[str, Any]:
        """检测踮脚作业"""
        # 计算膝关节角度
        # 这里需要踝关节点，简化处理
        # 假设垂直站立时膝关节角度约为180度
        left_knee_angle = 180.0
        right_knee_angle = 180.0
        
        is_squatting = left_knee_angle < 120 or right_knee_angle < 120
        
        return {
            'is_squatting': is_squatting,
            'left_knee_angle': left_knee_angle,
            'right_knee_angle': right_knee_angle,
            'angles': [left_knee_angle, right_knee_angle]
        }
    
    def _calculate_angle(self, p1: Keypoint, p2: Keypoint, p3: Keypoint) -> float:
        """计算三点角度"""
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p2.x - p3.x, p2.y - p3.y])
        
        angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
        
        return float(angle)


class FatigueDetector:
    """疲劳检测器"""
    
    def __init__(
        self,
        blink_threshold: int = 25,
        action_time_threshold: float = 5.0,
        posture_threshold: float = -0.2
    ):
        self.blink_threshold = blink_threshold
        self.action_time_threshold = action_time_threshold
        self.posture_threshold = posture_threshold
        
        self.blink_history: List[float] = []
        self.action_time_history: List[float] = []
        self.posture_history: List[float] = []
        self.window_size = 60  # 60秒窗口
    
    def update(
        self,
        blink_count: int,
        action_time: float,
        posture_score: float
    ) -> Dict[str, Any]:
        """更新疲劳数据"""
        timestamp = datetime.now().timestamp()
        
        self.blink_history.append((timestamp, blink_count))
        self.action_time_history.append((timestamp, action_time))
        self.posture_history.append((timestamp, posture_score))
        
        # 保持窗口大小
        self._trim_history(timestamp)
        
        return self._detect_fatigue()
    
    def _detect_fatigue(self) -> Dict[str, Any]:
        """检测疲劳程度"""
        if len(self.blink_history) < 10:
            return {
                'fatigue_level': 'normal',
                'confidence': 0.0
            }
        
        # 计算各指标趋势
        recent_blinks = [count for _, count in self.blink_history[-10:]]
        avg_blinks = np.mean(recent_blinks)
        
        recent_action_times = [time for _, time in self.action_time_history[-10:]]
        avg_action_time = np.mean(recent_action_times)
        
        recent_postures = [score for _, score in self.posture_history[-10:]]
        avg_posture = np.mean(recent_postures)
        
        fatigue_score = 0.0
        reasons = []
        
        # 眨眼增多
        if avg_blinks > self.blink_threshold:
            fatigue_score += 0.4
            reasons.append('increased_blink_rate')
        
        # 动作变慢
        if avg_action_time > self.action_time_threshold:
            fatigue_score += 0.4
            reasons.append('slower_action_speed')
        
        # 姿势松懈
        if avg_posture < self.posture_threshold:
            fatigue_score += 0.2
            reasons.append('posture_slackness')
        
        # 判断疲劳等级
        if fatigue_score > 0.7:
            fatigue_level = 'high'
        elif fatigue_score > 0.4:
            fatigue_level = 'moderate'
        else:
            fatigue_level = 'normal'
        
        return {
            'fatigue_level': fatigue_level,
            'confidence': min(fatigue_score, 1.0),
            'reasons': reasons,
            'avg_blinks': avg_blinks,
            'avg_action_time': avg_action_time,
            'avg_posture': avg_posture
        }
    
    def _trim_history(self, current_timestamp: float) -> None:
        """保持历史窗口大小"""
        while self.blink_history and self.blink_history[0][0] < current_timestamp - self.window_size:
            self.blink_history.pop(0)
            self.action_time_history.pop(0)
            self.posture_history.pop(0)


class DangerZoneMonitor:
    """危险区域监控器"""
    
    def __init__(self, danger_zones: List[Dict[str, Any]]):
        self.danger_zones = danger_zones
    
    def check_hand_in_danger_zone(
        self,
        hand_position: np.ndarray,
        danger_zone_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """检查手部是否进入危险区域"""
        if not self.danger_zones:
            return {'in_danger': False}
        
        zone = self.danger_zones[danger_zone_id] if danger_zone_id else self.danger_zones[0]
        polygon = np.array(zone['polygon'])
        
        # 使用点在多边形内算法
        from matplotlib.path import Path
        path = Path(polygon)
        in_danger = path.contains_point(hand_position)
        
        return {
            'in_danger': in_danger,
            'zone_id': danger_zone_id,
            'zone_name': zone.get('name', 'unknown')
        }


class PostureSafetyService:
    """姿势安全服务"""
    
    def __init__(
        self,
        estimator: PostureEstimator,
        fatigue_detector: FatigueDetector,
        danger_monitor: DangerZoneMonitor
    ):
        self.estimator = estimator
        self.fatigue_detector = fatigue_detector
        self.danger_monitor = danger_monitor
        self.alert_history: List[Dict[str, Any]] = []
    
    def analyze_frame(
        self,
        frame: np.ndarray,
        keypoints: Optional[List[Keypoint]] = None,
        hand_position: Optional[np.ndarray] = None,
        head_position: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """分析一帧
        
        Args:
            frame: 输入图像帧
            keypoints: 关键点列表
            hand_position: 手部位置
            head_position: 头部位置
            
        Returns:
            dict: 分析结果
        """
        result = {
            'timestamp': datetime.now().isoformat()
        }
        
        # 姿势估计
        posture_analysis = self.estimator.detect_posture(frame, keypoints)
        result['posture'] = posture_analysis
        
        # 危险区域检测
        if hand_position is not None:
            danger_check = self.danger_monitor.check_hand_in_danger_zone(hand_position)
            result['danger_zone'] = danger_check
        else:
            result['danger_zone'] = {'in_danger': False}
        
        # 综合判断
        risk_level = posture_analysis.risk_level
        if result['danger_zone']['in_danger']:
            risk_level = PostureRisk.CRITICAL
        
        result['risk_level'] = risk_level
        
        # 检查是否需要告警
        needs_alert = (
            risk_level in [PostureRisk.HIGH, PostureRisk.CRITICAL] or
            result['danger_zone']['in_danger']
        )
        
        if needs_alert:
            alert = {
                'timestamp': result['timestamp'],
                'type': 'posture_alert',
                'risk_level': risk_level.value,
                'posture_type': posture_analysis.posture_type.value,
                'details': posture_analysis.details,
                'in_danger_zone': result['danger_zone']['in_danger']
            }
            self.alert_history.append(alert)
            result['alert'] = alert
        
        return result
    
    def update_fatigue(
        self,
        blink_count: int,
        action_time: float,
        posture_score: float
    ) -> Dict[str, Any]:
        """更新疲劳检测"""
        return self.fatigue_detector.update(blink_count, action_time, posture_score)
    
    def get_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取告警历史"""
        return self.alert_history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.alert_history:
            return {}
        
        total_alerts = len(self.alert_history)
        
        # 按风险等级统计
        risk_counts = {}
        for alert in self.alert_history:
            risk = alert.get('risk_level', 'normal')
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        # 按姿势类型统计
        posture_counts = {}
        for alert in self.alert_history:
            posture = alert.get('posture_type', 'normal')
            posture_counts[posture] = posture_counts.get(posture, 0) + 1
        
        return {
            'total_alerts': total_alerts,
            'risk_distribution': risk_counts,
            'posture_distribution': posture_counts,
            'recent_alerts': self.alert_history[-10:]
        }
