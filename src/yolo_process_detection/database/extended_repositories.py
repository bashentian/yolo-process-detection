"""扩展仓储类

支持新增的数据库模型。
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from ..database.models import (
    Base,
    Camera,
    Detection,
    ActionSegment,
    ActionSequence,
    ScrewDetection,
    MaterialPick,
    PostureEvent,
    FatigueRecord,
    SensorReading,
    WorkflowStep,
    WorkflowExecution,
    TrainingData,
    User
)


class ActionSegmentRepository:
    """动作片段仓储"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_segment(
        self,
        camera_id: int,
        action_type: str,
        start_time: float,
        end_time: float,
        confidence: float,
        bbox_keyframes: Dict[str, List[int]]
    ) -> ActionSegment:
        """创建动作片段"""
        segment = ActionSegment(
            camera_id=camera_id,
            action_type=action_type,
            start_time=start_time,
            end_time=end_time,
            confidence=confidence,
            bbox_keyframes=bbox_keyframes
        )
        
        self.db.add(segment)
        self.db.commit()
        self.db.refresh(segment)
        
        return segment
    
    def get_recent_segments(
        self,
        camera_id: int,
        limit: int = 100
    ) -> List[ActionSegment]:
        """获取最近的动作片段"""
        return self.db.query(ActionSegment).filter(
            ActionSegment.camera_id == camera_id
        ).order_by(
            ActionSegment.timestamp.desc()
        ).limit(limit).all()
    
    def get_by_sequence_id(
        self,
        sequence_id: str
    ) -> List[ActionSegment]:
        """根据序列ID获取片段"""
        return self.db.query(ActionSegment).filter(
            ActionSegment.sequence_id == sequence_id
        ).order_by(
            ActionSegment.timestamp.asc()
        ).all()


class ScrewDetectionRepository:
    """螺钉检测仓储"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_detection(
        self,
        camera_id: int,
        position_x: float,
        position_y: float,
        width: float,
        height: float,
        confidence: float,
        status: str,
        deviation: float = 0.0,
        torque_value: Optional[float] = None,
        sound_detected: bool = False
    ) -> ScrewDetection:
        """创建螺钉检测记录"""
        detection = ScrewDetection(
            camera_id=camera_id,
            position_x=position_x,
            position_y=position_y,
            width=width,
            height=height,
            confidence=confidence,
            status=status,
            deviation=deviation,
            torque_value=torque_value,
            sound_detected=sound_detected
        )
        
        self.db.add(detection)
        self.db.commit()
        self.db.refresh(detection)
        
        return detection
    
    def get_recent_detections(
        self,
        camera_id: int,
        limit: int = 100
    ) -> List[ScrewDetection]:
        """获取最近的螺钉检测记录"""
        return self.db.query(ScrewDetection).filter(
            ScrewDetection.camera_id == camera_id
        ).order_by(
            ScrewDetection.timestamp.desc()
        ).limit(limit).all()
    
    def get_statistics(self, camera_id: int) -> Dict[str, Any]:
        """获取螺钉检测统计"""
        detections = self.get_recent_detections(camera_id, limit=1000)
        
        if not detections:
            return {}
        
        total = len(detections)
        positioned = sum(1 for d in detections if d.status == "positioned")
        tightened = sum(1 for d in detections if d.status == "tightened")
        overtightened = sum(1 for d in detections if d.status == "overtightened")
        undetected = sum(1 for d in detections if d.status == "undetected")
        
        avg_confidence = sum(d.confidence for d in detections) / total if total > 0 else 0.0
        avg_deviation = sum(d.deviation for d in detections) / total if total > 0 else 0.0
        
        return {
            'total_detections': total,
            'positioned': positioned,
            'tightened': tightened,
            'overtightened': overtightened,
            'undetected': undetected,
            'avg_confidence': avg_confidence,
            'avg_deviation': avg_deviation,
            'pass_rate': (positioned + tightened) / total if total > 0 else 0
        }


class MaterialPickRepository:
    """物料取用仓储"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_pick(
        self,
        camera_id: int,
        worker_id: int,
        material_id: str,
        material_name: str,
        confidence: float,
        is_correct: bool,
        expected_material: str,
        hand_position: Optional[str] = None,
        bbox: Optional[str] = None
    ) -> MaterialPick:
        """创建物料取用记录"""
        pick = MaterialPick(
            camera_id=camera_id,
            worker_id=worker_id,
            material_id=material_id,
            material_name=material_name,
            confidence=confidence,
            is_correct=is_correct,
            expected_material=expected_material,
            hand_position=hand_position,
            bbox=bbox
        )
        
        self.db.add(pick)
        self.db.commit()
        self.db.refresh(pick)
        
        return pick
    
    def get_recent_picks(
        self,
        camera_id: int,
        limit: int = 100
    ) -> List[MaterialPick]:
        """获取最近的物料取用记录"""
        return self.db.query(MaterialPick).filter(
            MaterialPick.camera_id == camera_id
        ).order_by(
            MaterialPick.timestamp.desc()
        ).limit(limit).all()
    
    def validate_pick_sequence(
        self,
        expected_materials: List[str],
        actual_picks: List[MaterialPick]
    ) -> Dict[str, Any]:
        """验证取料序列"""
        if not actual_picks:
            return {
                'valid': False,
                'error': 'no_picks_detected'
            }
        
        actual_materials = [p.material_id for p in actual_picks if p.material_id]
        expected_materials = expected_materials[:len(actual_materials)]
        
        is_correct = actual_materials == expected_materials
        sequence_correct = all(
            actual_picks[i].material_id == expected_materials[i]
            for i in range(len(expected_materials))
        )
        
        has_duplicate = len(actual_materials) != len(set(actual_materials))
        
        return {
            'valid': is_correct and sequence_correct and not has_duplicate,
            'expected_materials': expected_materials,
            'actual_materials': actual_materials,
            'is_correct': is_correct,
            'sequence_correct': sequence_correct,
            'has_duplicate': has_duplicate
        }


class PostureEventRepository:
    """姿势安全事件仓储"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_event(
        self,
        camera_id: int,
        worker_id: int,
        posture_type: str,
        risk_level: str,
        confidence: float,
        details: Optional[Dict[str, Any]] = None
    ) -> PostureEvent:
        """创建姿势安全事件"""
        event = PostureEvent(
            camera_id=camera_id,
            worker_id=worker_id,
            posture_type=posture_type,
            risk_level=risk_level,
            confidence=confidence,
            details=details
        )
        
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        
        return event
    
    def get_recent_events(
        self,
        camera_id: int,
        limit: int = 100
    ) -> List[PostureEvent]:
        """获取最近的姿势安全事件"""
        return self.db.query(PostureEvent).filter(
            PostureEvent.camera_id == camera_id
        ).order_by(
            PostureEvent.timestamp.desc()
        ).limit(limit).all()
    
    def get_statistics(self, camera_id: int) -> Dict[str, Any]:
        """获取姿势安全统计"""
        events = self.get_recent_events(camera_id, limit=1000)
        
        if not events:
            return {}
        
        total = len(events)
        
        risk_counts = {}
        for event in events:
            risk = event.risk_level
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        return {
            'total_events': total,
            'risk_distribution': risk_counts,
            'high_risk_count': risk_counts.get('high', 0),
            'moderate_risk_count': risk_counts.get('moderate', 0),
            'low_risk_count': risk_counts.get('normal', 0)
        }


class FatigueRecordRepository:
    """疲劳检测记录仓储"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_record(
        self,
        camera_id: int,
        worker_id: int,
        fatigue_level: str,
        blink_count: int,
        action_time: float,
        posture_score: float,
        details: Optional[Dict[str, Any]] = None
    ) -> FatigueRecord:
        """创建疲劳检测记录"""
        record = FatigueRecord(
            camera_id=camera_id,
            worker_id=worker_id,
            fatigue_level=fatigue_level,
            blink_count=blink_count,
            action_time=action_time,
            posture_score=posture_score,
            details=details
        )
        
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        
        return record
    
    def get_recent_records(
        self,
        camera_id: int,
        limit: int = 100
    ) -> List[FatigueRecord]:
        """获取最近的疲劳检测记录"""
        return self.db.query(FatigueRecord).filter(
            FatigueRecord.camera_id == camera_id
        ).order_by(
            FatigueRecord.timestamp.desc()
        ).limit(limit).all()


class SensorReadingRepository:
    """传感器读数仓储"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_reading(
        self,
        camera_id: int,
        sensor_type: str,
        data: Any,
        confidence: float
    ) -> SensorReading:
        """创建传感器读数记录"""
        reading = SensorReading(
            camera_id=camera_id,
            sensor_type=sensor_type,
            data=data,
            confidence=confidence
        )
        
        self.db.add(reading)
        self.db.commit()
        self.db.refresh(reading)
        
        return reading
    
    def get_recent_readings(
        self,
        camera_id: int,
        limit: int = 100
    ) -> List[SensorReading]:
        """获取最近的传感器读数"""
        return self.db.query(SensorReading).filter(
            SensorReading.camera_id == camera_id
        ).order_by(
            SensorReading.timestamp.desc()
        ).limit(limit).all()
