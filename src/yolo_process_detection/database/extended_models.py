"""扩展数据库模型

添加新的表支持连贯动作识别、螺钉检测、物料识别和姿势安全功能。
"""
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    Text,
    ForeignKey,
    Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class ActionSegment(Base):
    """动作片段模型"""
    __tablename__ = "action_segments"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    action_type = Column(String(50), nullable=False, index=True)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    confidence = Column(Float, default=0.0)
    
    bbox_keyframes = Column(Text, nullable=True)
    attributes = Column(Text, nullable=True)
    
    camera = relationship("Camera", back_populates="action_segments")
    
    __table_args__ = (
        Index("idx_camera_timestamp", "camera_id", "timestamp"),
    )
    
    def __repr__(self):
        return f"<ActionSegment(id={self.id}, action_type='{self.action_type}')>"


class ActionSequence(Base):
    """操作序列模型"""
    __tablename__ = "action_sequences"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    worker_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    sequence_id = Column(String(100), nullable=False, index=True)
    actions = Column(Text, nullable=True)
    is_valid = Column(Boolean, default=True)
    compliance_score = Column(Float, default=0.0)
    missing_steps = Column(Text, nullable=True)
    wrong_order_errors = Column(Text, nullable=True)
    edit_distance = Column(Integer, default=0)
    
    total_duration = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    camera = relationship("Camera", back_populates="action_sequences")
    worker = relationship("User", back_populates="action_sequences")
    segments = relationship("ActionSegment", back_populates="action_sequences")
    
    __table_args__ = (
        Index("idx_sequence_id", "sequence_id"),
    )


class ScrewDetection(Base):
    """螺钉检测记录模型"""
    __tablename__ = "screw_detections"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    position_x = Column(Float, nullable=False)
    position_y = Column(Float, nullable=False)
    width = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    
    status = Column(String(20), nullable=False, index=True)
    deviation = Column(Float, default=0.0)
    deviation_x = Column(Float, default=0.0)
    deviation_y = Column(Float, default=0.0)
    confidence = Column(Float, default=0.0)
    
    torque_value = Column(Float, nullable=True)
    torque_target = Column(Float, nullable=True)
    tightening_duration = Column(Float, nullable=True)
    rotation_count = Column(Integer, default=0)
    sound_detected = Column(Boolean, default=False)
    
    image_path = Column(String(500), nullable=True)
    
    camera = relationship("Camera", back_populates="screw_detections")
    
    __table_args__ = (
        Index("idx_camera_timestamp", "camera_id", "timestamp"),
    )


class MaterialPick(Base):
    """物料取用记录模型"""
    __tablename__ = "material_picks"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    worker_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    material_id = Column(String(50), nullable=False, index=True)
    material_name = Column(String(100), nullable=True)
    confidence = Column(Float, default=0.0)
    
    is_correct = Column(Boolean, default=True)
    expected_material = Column(String(50), nullable=True)
    
    hand_position = Column(Text, nullable=True)
    bbox = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    camera = relationship("Camera", back_populates="material_picks")
    worker = relationship("User", back_populates="material_picks")


class PostureEvent(Base):
    """姿势安全事件模型"""
    __tablename__ = "posture_events"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    worker_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    posture_type = Column(String(50), nullable=False, index=True)
    risk_level = Column(String(20), nullable=False, index=True)
    confidence = Column(Float, default=0.0)
    
    details = Column(Text, nullable=True)
    
    image_path = Column(String(500), nullable=True)
    
    camera = relationship("Camera", back_populates="posture_events")
    worker = relationship("User", back_populates="posture_events")


class FatigueRecord(Base):
    """疲劳检测记录模型"""
    __tablename__ = "fatigue_records"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    worker_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    fatigue_level = Column(String(20), nullable=False, index=True)
    blink_count = Column(Integer, default=0)
    action_time = Column(Float, default=0.0)
    posture_score = Column(Float, default=0.0)
    
    details = Column(Text, nullable=True)
    
    camera = relationship("Camera", back_populates="fatigue_records")
    worker = relationship("User", back_populates="fatigue_records")


class SensorReading(Base):
    """传感器读数模型"""
    __tablename__ = "sensor_readings"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    sensor_type = Column(String(20), nullable=False, index=True)
    data = Column(Text, nullable=True)
    confidence = Column(Float, default=0.0)
    
    camera = relationship("Camera", back_populates="sensor_readings")


class WorkflowStep(Base):
    """工作流步骤模型"""
    __tablename__ = "workflow_steps"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    expected_duration_min = Column(Float, nullable=True)
    expected_duration_max = Column(Float, nullable=True)
    
    dependencies = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class WorkflowExecution(Base):
    """工作流执行记录模型"""
    __tablename__ = "workflow_executions"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    worker_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    sequence_id = Column(String(100), ForeignKey("action_sequences.sequence_id"), nullable=False, index=True)
    
    is_valid = Column(Boolean, default=True)
    actual_duration = Column(Float, nullable=True)
    expected_duration = Column(Float, nullable=True)
    
    errors = Column(Text, nullable=True)
    
    started_at = Column(DateTime, default=datetime.utcnow())
    completed_at = Column(DateTime, nullable=True)
    
    camera = relationship("Camera", back_populates="workflow_executions")
    worker = relationship("User", back_populates="workflow_executions")
    sequence = relationship("ActionSequence", back_populates="workflow_executions")


class TrainingData(Base):
    """训练数据模型"""
    __tablename__ = "training_data"
    
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(String(100), nullable=False, index=True)
    worker_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    action_type = Column(String(50), nullable=False, index=True)
    action_label = Column(String(50), nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    
    bbox_keyframes = Column(Text, nullable=True)
    attributes = Column(Text, nullable=True)
    
    quality_score = Column(Float, default=0.0)
    is_valid = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    worker = relationship("User", back_populates="training_data")
