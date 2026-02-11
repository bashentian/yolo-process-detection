"""物料识别检测模块

验证工人取用正确物料，检测取料行为。
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np
from datetime import datetime


class MaterialType(Enum):
    """物料类型枚举"""
    SCREW_M4 = "screw_m4"
    SCREW_M6 = "screw_m6"
    CIRCUIT_BOARD = "circuit_board"
    CONNECTOR = "connector"
    HOUSING = "housing"
    UNKNOWN = "unknown"


@dataclass
class MaterialInfo:
    """物料信息"""
    material_id: str
    name: str
    type: MaterialType
    image_features: Optional[np.ndarray] = None
    qr_code: Optional[str] = None
    description: Optional[str] = None


@dataclass
class MaterialDetectionResult:
    """物料检测结果"""
    material_id: Optional[str] = None
    material_name: Optional[str] = None
    confidence: float = 0.0
    bbox: Optional[List[float]] = None
    is_correct: bool = False
    expected_material: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PickAction:
    """取料动作"""
    action_id: str
    start_time: float
    end_time: float
    hand_position: Optional[np.ndarray] = None
    material_in_hand: Optional[str] = None
    material_confidence: float = 0.0
    is_valid: bool = True


class MaterialClassifier:
    """物料分类器"""
    
    def __init__(self, material_database: Dict[str, MaterialInfo]):
        self.material_database = material_database
        self.confidence_threshold = 0.85
    
    def classify(
        self,
        frame: np.ndarray,
        bbox: List[float]
    ) -> MaterialDetectionResult:
        """分类物料
        
        Args:
            frame: 输入图像帧
            bbox: 物料边界框 [x_min, y_min, x_max, y_max]
            
        Returns:
            MaterialDetectionResult: 分类结果
        """
        # 裁剪物料区域
        x_min, y_min, x_max, y_max = bbox
        material_region = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
        
        # 提取特征
        features = self._extract_features(material_region)
        
        # 匹配物料库
        best_match = None
        best_confidence = 0.0
        
        for material_id, material_info in self.material_database.items():
            if material_info.image_features is None:
                continue
            
            similarity = self._calculate_similarity(features, material_info.image_features)
            
            if similarity > best_confidence:
                best_match = material_id
                best_confidence = similarity
        
        is_correct = best_confidence > self.confidence_threshold
        
        return MaterialDetectionResult(
            material_id=best_match,
            material_name=self.material_database[best_match].name if best_match else None,
            confidence=best_confidence,
            bbox=bbox,
            is_correct=is_correct
        )
    
    def _extract_features(self, region: np.ndarray) -> np.ndarray:
        """提取图像特征"""
        # 颜色直方图
        hist_r = cv2.calcHist([region], [0, 256], [0, 256])
        hist_g = cv2.calcHist([region], [1, 256], [0, 256])
        hist_b = cv2.calcHist([region], [2, 256], [0, 256])
        
        # 形状特征
        contours, _ = cv2.findContours(
            cv2.cvtColor(region, cv2.COLOR_BGR2GRAY),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 组合特征
        features = np.concatenate([
            hist_r.flatten(),
            hist_g.flatten(),
            hist_b.flatten()
        ])
        
        return features
    
    def _calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """计算特征相似度"""
        # 使用余弦相似度
        norm1 = features1 / (np.linalg.norm(features1) + 1e-8)
        norm2 = features2 / (np.linalg.norm(features2) + 1e-8)
        similarity = np.dot(norm1, norm2)
        
        return float(similarity)


class PickBehaviorDetector:
    """取料行为检测器"""
    
    def __init__(self, material_classifier: MaterialClassifier):
        self.classifier = material_classifier
        self.hand_tracker = None
        self.pick_history: List[PickAction] = []
        self.current_pick: Optional[PickAction] = None
    
    def process_frame(
        self,
        frame: np.ndarray,
        hand_keypoints: Optional[np.ndarray] = None,
        material_info: Optional[MaterialInfo] = None
    ) -> Optional[PickAction]:
        """处理一帧
        
        Args:
            frame: 输入图像帧
            hand_keypoints: 手部关键点
            material_info: 当前工序应取的物料
            
        Returns:
            PickAction: 取料动作或None
        """
        if hand_keypoints is None:
            return None
        
        # 检测手部位置
        wrist_pos = hand_keypoints[0]
        
        # 判断是否在料盒区域
        if material_info and material_info.position:
            box_center = np.array([
                (material_info.position['x'] + material_info.position['width'] / 2,
                (material_info.position['y'] + material_info.position['height']) / 2
            ])
            hand_pos = np.array([wrist_pos[0], wrist_pos[1]])
            
            distance = np.linalg.norm(hand_pos - box_center)
            
            if distance < 100:
                # 手在料盒附近，检测取料
                return self._detect_pick_action(frame, wrist_pos, material_info)
        
        return None
    
    def _detect_pick_action(
        self,
        frame: np.ndarray,
        hand_pos: np.ndarray,
        material_info: MaterialInfo
    ) -> PickAction:
        """检测取料动作"""
        # 检测手部是否持有物料
        is_holding = self._check_holding_material(frame, hand_pos, material_info)
        
        if is_holding:
            # 检测到取料动作
            action = PickAction(
                action_id=f"pick_{datetime.now().strftime('%H%M%S')}",
                start_time=datetime.now().timestamp(),
                end_time=datetime.now().timestamp() + 0.5,
                hand_position=hand_pos,
                material_in_hand=material_info.material_id,
                material_confidence=0.9,
                is_valid=True
            )
            self.current_pick = action
            self.pick_history.append(action)
            return action
        
        return None
    
    def _check_holding_material(
        self,
        frame: np.ndarray,
        hand_pos: np.ndarray,
        material_info: MaterialInfo
    ) -> bool:
        """检查手部是否持有物料"""
        # 简化：检查手部是否在物料位置附近
        material_center = np.array([
            material_info.position['x'] + material_info.position['width'] / 2,
            material_info.position['y'] + material_info.position['height'] / 2
        ])
        
        distance = np.linalg.norm(hand_pos - material_center)
        
        return distance < 80
    
    def validate_pick_sequence(
        self,
        expected_materials: List[str],
        actual_picks: List[PickAction]
    ) -> Dict[str, Any]:
        """验证取料序列
        
        Args:
            expected_materials: 预期的物料序列
            actual_picks: 实际取料动作列表
            
        Returns:
            dict: 验证结果
        """
        if not actual_picks:
            return {
                'valid': False,
                'error': 'no_picks_detected'
            }
        
        actual_materials = [p.material_in_hand for p in actual_picks if p.material_in_hand]
        expected_materials = expected_materials[:len(actual_materials)]
        
        # 检查物料是否正确
        is_correct = actual_materials == expected_materials
        
        # 检查顺序
        sequence_correct = all(
            actual_picks[i].material_in_hand == expected_materials[i]
            for i in range(len(expected_materials))
        )
        
        # 检查重复取料
        has_duplicate = len(actual_materials) != len(set(actual_materials))
        
        return {
            'valid': is_correct and sequence_correct and not has_duplicate,
            'expected_materials': expected_materials,
            'actual_materials': actual_materials,
            'is_correct': is_correct,
            'sequence_correct': sequence_correct,
            'has_duplicate': has_duplicate
        }


class MaterialService:
    """物料识别服务"""
    
    def __init__(
        self,
        material_database: Dict[str, MaterialInfo],
        classifier: MaterialClassifier,
        behavior_detector: PickBehaviorDetector
    ):
        self.material_database = material_database
        self.classifier = classifier
        self.behavior_detector = behavior_detector
        self.pick_records: List[Dict[str, Any]] = []
    
    def process_frame(
        self,
        frame: np.ndarray,
        hand_keypoints: Optional[np.ndarray] = None,
        expected_material: Optional[str] = None
    ) -> Dict[str, Any]:
        """处理一帧
        
        Args:
            frame: 输入图像帧
            hand_keypoints: 手部关键点
            expected_material: 当前工序应取的物料
            
        Returns:
            dict: 处理结果
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'expected_material': expected_material
        }
        
        # 物料分类
        if expected_material and expected_material in self.material_database:
            material_info = self.material_database[expected_material]
            
            # 检测物料位置
            if material_info.position:
                detection_result = self.classifier.classify(
                    frame,
                    [
                        material_info.position['x'],
                        material_info.position['y'],
                        material_info.position['x'] + material_info.position['width'],
                        material_info.position['y'] + material_info.position['height']
                    ]
                )
                
                result['material_detection'] = {
                    'material_id': detection_result.material_id,
                    'material_name': detection_result.material_name,
                    'confidence': detection_result.confidence,
                    'is_correct': detection_result.is_correct
                }
        
        # 取料行为检测
        pick_action = self.behavior_detector.process_frame(
            frame, hand_keypoints, material_info
        )
        
        if pick_action:
            result['pick_action'] = {
                'action_id': pick_action.action_id,
                'material_in_hand': pick_action.material_in_hand,
                'is_valid': pick_action.is_valid
            }
        
        return result
    
    def validate_pick_sequence(
        self,
        expected_materials: List[str],
        picks: List[PickAction]
    ) -> Dict[str, Any]:
        """验证取料序列"""
        return self.behavior_detector.validate_pick_sequence(expected_materials, picks)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.pick_records:
            return {}
        
        total_picks = len(self.pick_records)
        correct_picks = sum(1 for r in self.pick_records if r.is_valid)
        
        # 按物料统计
        material_stats = {}
        for record in self.pick_records:
            material = record.material_in_hand
            if material not in material_stats:
                material_stats[material] = {'total': 0, 'correct': 0}
            material_stats[material]['total'] += 1
            if record.is_valid:
                material_stats[material]['correct'] += 1
        
        return {
            'total_picks': total_picks,
            'correct_picks': correct_picks,
            'accuracy': correct_picks / total_picks if total_picks > 0 else 0,
            'material_stats': material_stats
        }
