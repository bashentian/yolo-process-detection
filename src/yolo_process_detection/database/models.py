"""数据库模型模块

提供数据库模型定义和ORM映射。
"""
from datetime import datetime
from typing import Optional
from enum import Enum

from sqlalchemy import (
    create_engine,
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
from sqlalchemy.orm import (
    sessionmaker,
    relationship,
    Session
)


Base = declarative_base()


class CameraStatus(str, Enum):
    """摄像头状态"""
    STOPPED = "stopped"
    RUNNING = "running"
    ERROR = "error"


class Camera(Base):
    """摄像头模型"""
    __tablename__ = "cameras"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    location = Column(String(200), nullable=False)
    source = Column(String(500), nullable=False)
    resolution = Column(String(20), default="1920x1080")
    fps = Column(Integer, default=30)
    status = Column(String(20), default=CameraStatus.STOPPED)
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    detections = relationship("Detection", back_populates="camera", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Camera(id={self.id}, name='{self.name}', status='{self.status}')>"


class Detection(Base):
    """检测记录模型"""
    __tablename__ = "detections"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    frame_number = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    x_min = Column(Float, nullable=False)
    y_min = Column(Float, nullable=False)
    x_max = Column(Float, nullable=False)
    y_max = Column(Float, nullable=False)
    
    confidence = Column(Float, nullable=False)
    class_id = Column(Integer, nullable=False)
    class_name = Column(String(50), nullable=False)
    track_id = Column(Integer, nullable=True)
    
    image_path = Column(String(500), nullable=True)
    thumbnail_path = Column(String(500), nullable=True)
    
    camera = relationship("Camera", back_populates="detections")
    
    __table_args__ = (
        Index("idx_camera_timestamp", "camera_id", "timestamp"),
    )
    
    def __repr__(self):
        return f"<Detection(id={self.id}, class_name='{self.class_name}', confidence={self.confidence})>"


class DetectionStatistics(Base):
    """检测统计模型"""
    __tablename__ = "detection_statistics"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=True, index=True)
    date = Column(DateTime, nullable=False, index=True)
    
    total_detections = Column(Integer, default=0)
    by_class = Column(Text, nullable=True)
    avg_confidence = Column(Float, default=0.0)
    max_confidence = Column(Float, default=0.0)
    min_confidence = Column(Float, default=0.0)
    
    frame_count = Column(Integer, default=0)
    processing_time = Column(Float, default=0.0)
    fps = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_camera_date", "camera_id", "date"),
    )
    
    def __repr__(self):
        return f"<DetectionStatistics(id={self.id}, date='{self.date}', total={self.total_detections})>"


class SystemEvent(Base):
    """系统事件模型"""
    __tablename__ = "system_events"
    
    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    event_level = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=True)
    
    extra_data = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index("idx_event_type_time", "event_type", "created_at"),
    )
    
    def __repr__(self):
        return f"<SystemEvent(id={self.id}, type='{self.event_type}', level='{self.event_level}')>"


class Alert(Base):
    """警报模型"""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)
    
    detection_id = Column(Integer, ForeignKey("detections.id"), nullable=True)
    image_path = Column(String(500), nullable=True)
    
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime, nullable=True)
    acknowledged_by = Column(String(100), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index("idx_alert_camera_time", "camera_id", "created_at"),
    )
    
    def __repr__(self):
        return f"<Alert(id={self.id}, type='{self.alert_type}', severity='{self.severity}')>"


class User(Base):
    """用户模型"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    
    full_name = Column(String(100), nullable=True)
    role = Column(String(20), default="user")
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"


class Settings(Base):
    """系统设置模型"""
    __tablename__ = "settings"
    
    id = Column(Integer, primary_key=True, index=True)
    category = Column(String(50), nullable=False, index=True)
    key = Column(String(100), nullable=False)
    value = Column(Text, nullable=False)
    value_type = Column(String(20), default="string")
    
    description = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(100), nullable=True)
    
    __table_args__ = (
        Index("idx_category_key", "category", "key"),
    )
    
    def __repr__(self):
        return f"<Settings(id={self.id}, category='{self.category}', key='{self.key}')>"
