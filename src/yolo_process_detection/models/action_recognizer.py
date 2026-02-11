"""连贯动作识别模块

基于时序视觉分析的动作识别系统，理解完整操作流程。
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import numpy as np
from datetime import datetime


class ActionType(Enum):
    """动作类型枚举"""
    IDLE = "idle"
    PICK_SCREW = "pick_screw"
    POSITION_SCREW = "position_screw"
    HAND_TIGHTEN = "hand_tighten"
    POWER_TIGHTEN = "power_tighten"
    INSPECT = "inspect"
    COMPLETE = "complete"
    RETURN_TOOL = "return_tool"


@dataclass
class ActionSegment:
    """动作片段"""
    action_id: int
    action_type: ActionType
    start_time: float
    end_time: float
    confidence: float
    frame_indices: List[int]
    bbox_keyframes: Dict[str, List[List[int]]] = field(default_factory=dict)


@dataclass
class SequenceResult:
    """序列识别结果"""
    sequence_id: str
    actions: List[ActionSegment]
    is_valid: bool
    compliance_score: float
    missing_steps: List[str]
    wrong_order_errors: List[Dict[str, Any]]
    edit_distance: int = 0
    total_duration: float = 0.0


class StandardSequence:
    """标准操作序列定义"""
    
    SCREW_ASSEMBLY = [
        ActionType.PICK_SCREW,
        ActionType.POSITION_SCREW,
        ActionType.HAND_TIGHTEN,
        ActionType.POWER_TIGHTEN,
        ActionType.INSPECT,
        ActionType.COMPLETE
    ]


class SequenceValidator:
    """序列验证器"""
    
    def __init__(self, standard_sequence: List[ActionType]):
        self.standard_sequence = standard_sequence
        self.transitions = {
            ActionType.IDLE: [ActionType.PICK_SCREW],
            ActionType.PICK_SCREW: [ActionType.POSITION_SCREW],
            ActionType.POSITION_SCREW: [ActionType.HAND_TIGHTEN],
            ActionType.HAND_TIGHTEN: [ActionType.POWER_TIGHTEN],
            ActionType.POWER_TIGHTEN: [ActionType.INSPECT, ActionType.HAND_TIGHTEN],
            ActionType.INSPECT: [ActionType.COMPLETE],
            ActionType.COMPLETE: [ActionType.IDLE]
        }
    
    def validate_sequence(self, actual_sequence: List[ActionSegment]) -> SequenceResult:
        """验证操作序列
        
        Args:
            actual_sequence: 实际执行的动作序列
            
        Returns:
            SequenceResult: 验证结果
        """
        errors = []
        missing_steps = []
        wrong_order_errors = []
        
        action_types = [seg.action_type for seg in actual_sequence]
        
        # 检查遗漏步骤
        for step in self.standard_sequence:
            if step not in action_types:
                missing_steps.append(step.value)
        
        # 检查顺序错误
        for i in range(len(action_types) - 1):
            current_step = action_types[i]
            next_step = action_types[i + 1]
            
            # 在标准序列中查找索引
            try:
                std_current_idx = self.standard_sequence.index(current_step)
                std_next_idx = self.standard_sequence.index(next_step)
                
                # 如果实际顺序反了
                if std_next_idx < std_current_idx:
                    wrong_order_errors.append({
                        'type': 'WRONG_ORDER',
                        'steps': [current_step.value, next_step.value],
                        'severity': 'HIGH'
                    })
            except ValueError:
                # 出现了不应该有的步骤
                wrong_order_errors.append({
                    'type': 'INVALID_STEP',
                    'step': next_step.value,
                    'severity': 'CRITICAL'
                })
        
        # 计算编辑距离（简化版）
        edit_distance = len(wrong_order_errors)
        
        # 计算合规分数
        total_steps = len(self.standard_sequence)
        completed_steps = len(action_types)
        compliance_score = completed_steps / total_steps if total_steps > 0 else 0.0
        
        is_valid = len(missing_steps) == 0 and len(wrong_order_errors) == 0
        
        return SequenceResult(
            sequence_id=f"seq_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            actions=actual_sequence,
            is_valid=is_valid,
            compliance_score=compliance_score,
            missing_steps=missing_steps,
            wrong_order_errors=wrong_order_errors,
            edit_distance=edit_distance,
            total_duration=sum(seg.end_time - seg.start_time for seg in actual_sequence)
        )


class TimeConstraintChecker:
    """时间约束检查器"""
    
    def __init__(self):
        self.constraints = {
            ('APPLY_GLUE', 'ASSEMBLE'): {'min': 10, 'max': 30},
            ('HEAT', 'ASSEMBLE'): {'min': 0, 'max': 5},
            ('HAND_TIGHTEN', 'POWER_TIGHTEN'): {'min': 0, 'max': 2}
        }
        self.step_timestamps = {}
    
    def record_step(self, step_name: str, timestamp: float) -> None:
        """记录步骤完成时间"""
        self.step_timestamps[step_name] = timestamp
    
    def check_constraint(self, step_a: str, step_b: str) -> Dict[str, Any]:
        """检查两步骤间时间约束"""
        if (step_a, step_b) not in self.constraints:
            return {'valid': True}
        
        if step_a not in self.step_timestamps or step_b not in self.step_timestamps:
            return {'valid': True, 'reason': 'incomplete'}
        
        time_diff = self.step_timestamps[step_b] - self.step_timestamps[step_a]
        constraint = self.constraints[(step_a, step_b)]
        
        if time_diff < constraint['min']:
            return {
                'valid': False,
                'reason': 'too_fast',
                'time_diff': time_diff,
                'required_min': constraint['min']
            }
        elif time_diff > constraint['max']:
            return {
                'valid': False,
                'reason': 'too_slow',
                'time_diff': time_diff,
                'required_max': constraint['max']
            }
        else:
            return {'valid': True}


class WorkflowGraph:
    """工序依赖图"""
    
    def __init__(self):
        self.graph = {
            'WS1_BASE_SCREW': [],
            'WS2_CIRCUIT_INSTALL': ['WS1_BASE_SCREW'],
            'WS3_CONNECTOR_PLUG': ['WS2_CIRCUIT_INSTALL'],
            'WS4_HOUSING_FIX': ['WS2_CIRCUIT_INSTALL', 'WS3_CONNECTOR_PLUG'],
            'WS5_FINAL_CHECK': ['WS4_HOUSING_FIX']
        }
    
    def validate_sequence(self, completed_workstations: List[str]) -> Dict[str, Any]:
        """验证工序执行顺序是否合法"""
        for ws in completed_workstations:
            dependencies = self.graph[ws]
            for dep in dependencies:
                if dep not in completed_workstations[:completed_workstations.index(ws)]:
                    return {
                        'valid': False,
                        'error': f'{ws}需要先完成{dep}'
                    }
        return {'valid': True}
    
    def get_next_available(self, completed: List[str]) -> List[str]:
        """获取下一步可以执行的工序"""
        available = []
        for ws, deps in self.graph.items():
            if ws not in completed:
                if all(dep in completed for dep in deps):
                    available.append(ws)
        return available


class ActionPredictor:
    """动作预测器"""
    
    def __init__(self, lookahead: int = 5):
        self.lookahead = lookahead
        self.action_history: List[ActionType] = []
    
    def update_history(self, action: ActionType) -> None:
        """更新动作历史"""
        self.action_history.append(action)
        if len(self.action_history) > 10:
            self.action_history.pop(0)
    
    def predict_next_actions(self) -> List[ActionType]:
        """预测接下来的动作"""
        if len(self.action_history) < 3:
            return []
        
        # 简化预测：基于历史模式
        recent_actions = self.action_history[-3:]
        
        # 统计每个动作出现的频率
        action_counts = {}
        for action in recent_actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # 基于频率预测
        predictions = []
        for _ in range(self.lookahead):
            # 选择最可能出现的动作
            if action_counts:
                next_action = max(action_counts.items(), key=lambda x: x[1])[0]
                predictions.append(next_action)
            else:
                predictions.append(ActionType.IDLE)
        
        return predictions
    
    def check_potential_error(self, standard_sequence: List[ActionType]) -> Optional[Dict[str, Any]]:
        """检查是否可能出错"""
        predicted_actions = self.predict_next_actions()
        
        if not predicted_actions:
            return None
        
        current_step = len(self.action_history)
        
        if current_step < len(standard_sequence):
            expected_next = standard_sequence[current_step]
            
            # 如果预测的第一个动作不是预期的
            if predicted_actions and predicted_actions[0] != expected_next:
                return {
                    'potential_error': True,
                    'expected': expected_next.value,
                    'predicted': predicted_actions[0].value,
                    'confidence': 0.7,
                    'action': 'WARN_BEFORE_ERROR'
                }
        
        return {'potential_error': False}


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """计算三点角度"""
    v1 = p1 - p2
    v2 = p2 - p3
    v3 = p3 - p1
    
    # 计算角度
    angle1 = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    angle2 = np.arccos(np.dot(v2, v3) / (np.linalg.norm(v2) * np.linalg.norm(v3)))
    
    return np.degrees(angle1 + angle2)


def analyze_grip_posture(hand_landmarks: np.ndarray) -> str:
    """分析手部抓取姿势"""
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    middle_tip = hand_landmarks[12]
    
    # 计算拇指与其他手指距离
    thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
    thumb_middle_dist = np.linalg.norm(thumb_tip - middle_tip)
    
    # 标准抓取：拇指与食指/中指距离较小
    if thumb_index_dist < 0.03 and thumb_middle_dist < 0.05:
        return "standard_grip"
    elif thumb_index_dist < 0.02:
        return "pinch_grip"
    else:
        return "palm_grip"


def detect_inspection_action(pose_sequence: List[Dict[str, Any]], gaze_data: Optional[Dict[str, Any]] = None) -> float:
    """检测检查动作"""
    if not pose_sequence:
        return 0.0
    
    # 检测头部位置
    head_positions = [pose.get('nose', np.array([0, 0, 0])) for pose in pose_sequence]
    
    # 计算头部与工件距离（简化）
    distances = [np.linalg.norm(pos) for pos in head_positions]
    
    # 判断是否有靠近动作
    close_inspection = any(d < 0.3 for d in distances)
    
    # 检测手部是否触摸
    hand_touch = any(pose.get('wrist', np.array([0, 0, 0]))[1] < 0.3 
                     for pose in pose_sequence)
    
    # 综合判断
    inspection_score = 0.0
    if close_inspection:
        inspection_score += 0.6
    if hand_touch:
        inspection_score += 0.4
    
    return inspection_score


def analyze_lifting_posture(pose_sequence: List[Dict[str, Any]]) -> str:
    """分析抬举动作安全性"""
    if not pose_sequence:
        return "NORMAL"
    
    # 提取关键关节角度序列
    hip_angles = []
    knee_angles = []
    
    for pose in pose_sequence:
        # 计算髋关节角度
        hip_angle = calculate_angle(
            pose.get('shoulder', np.array([0, 0, 0])),
            pose.get('hip', np.array([0, 0, 0])),
            pose.get('knee', np.array([0, 0, 0]))
        )
        hip_angles.append(hip_angle)
        
        # 计算膝关节角度
        knee_angle = calculate_angle(
            pose.get('hip', np.array([0, 0, 0])),
            pose.get('knee', np.array([0, 0, 0])),
            pose.get('ankle', np.array([0, 0, 0]))
        )
        knee_angles.append(knee_angle)
    
    # 判断是否屈膝下蹲
    knee_bent = any(angle < 120 for angle in knee_angles)
    
    # 判断是否直接弯腰
    excessive_hip_bend = any(angle < 90 for angle in hip_angles)
    
    if excessive_hip_bend and not knee_bent:
        return "UNSAFE_BENDING"
    elif knee_bent:
        return "SAFE_SQUATTING"
    else:
        return "NORMAL"
