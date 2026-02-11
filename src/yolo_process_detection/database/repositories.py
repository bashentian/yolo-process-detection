"""数据库存储层模块

提供数据库操作的仓储模式实现。
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from .models import Camera, Detection, DetectionStatistics, SystemEvent, Alert, User, Settings
from .manager import get_db_manager


class CameraRepository:
    """摄像头仓储"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_all(self) -> List[Camera]:
        """获取所有摄像头"""
        return self.db.query(Camera).all()
    
    def get_by_id(self, camera_id: int) -> Optional[Camera]:
        """根据ID获取摄像头"""
        return self.db.query(Camera).filter(Camera.id == camera_id).first()
    
    def get_active(self) -> List[Camera]:
        """获取启用的摄像头"""
        return self.db.query(Camera).filter(Camera.enabled == True).all()
    
    def get_running(self) -> List[Camera]:
        """获取运行中的摄像头"""
        return self.db.query(Camera).filter(Camera.status == "running").all()
    
    def create(self, **kwargs) -> Camera:
        """创建摄像头"""
        camera = Camera(**kwargs)
        self.db.add(camera)
        self.db.commit()
        self.db.refresh(camera)
        return camera
    
    def update(self, camera_id: int, **kwargs) -> Optional[Camera]:
        """更新摄像头"""
        camera = self.get_by_id(camera_id)
        if camera:
            for key, value in kwargs.items():
                setattr(camera, key, value)
            camera.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(camera)
        return camera
    
    def delete(self, camera_id: int) -> bool:
        """删除摄像头"""
        camera = self.get_by_id(camera_id)
        if camera:
            self.db.delete(camera)
            self.db.commit()
            return True
        return False
    
    def update_status(self, camera_id: int, status: str) -> bool:
        """更新摄像头状态"""
        camera = self.get_by_id(camera_id)
        if camera:
            camera.status = status
            camera.updated_at = datetime.utcnow()
            self.db.commit()
            return True
        return False


class DetectionRepository:
    """检测记录仓储"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, **kwargs) -> Detection:
        """创建检测记录"""
        detection = Detection(**kwargs)
        self.db.add(detection)
        self.db.commit()
        self.db.refresh(detection)
        return detection
    
    def create_batch(self, detections: List[Dict[str, Any]]) -> List[Detection]:
        """批量创建检测记录"""
        detection_objects = [Detection(**d) for d in detections]
        self.db.add_all(detection_objects)
        self.db.commit()
        for det in detection_objects:
            self.db.refresh(det)
        return detection_objects
    
    def get_by_id(self, detection_id: int) -> Optional[Detection]:
        """根据ID获取检测记录"""
        return self.db.query(Detection).filter(Detection.id == detection_id).first()
    
    def get_by_camera(
        self,
        camera_id: int,
        limit: int = 100,
        offset: int = 0
    ) -> List[Detection]:
        """获取摄像头的检测记录"""
        return self.db.query(Detection)\
            .filter(Detection.camera_id == camera_id)\
            .order_by(Detection.timestamp.desc())\
            .limit(limit)\
            .offset(offset)\
            .all()
    
    def get_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        camera_id: Optional[int] = None
    ) -> List[Detection]:
        """获取时间范围内的检测记录"""
        query = self.db.query(Detection)\
            .filter(Detection.timestamp >= start_time)\
            .filter(Detection.timestamp <= end_time)
        
        if camera_id is not None:
            query = query.filter(Detection.camera_id == camera_id)
        
        return query.order_by(Detection.timestamp.desc()).all()
    
    def get_by_class(
        self,
        class_name: str,
        limit: int = 100
    ) -> List[Detection]:
        """根据类别获取检测记录"""
        return self.db.query(Detection)\
            .filter(Detection.class_name == class_name)\
            .order_by(Detection.timestamp.desc())\
            .limit(limit)\
            .all()
    
    def get_statistics(
        self,
        camera_id: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """获取统计信息"""
        query = self.db.query(Detection)
        
        if camera_id is not None:
            query = query.filter(Detection.camera_id == camera_id)
        
        if start_time is not None:
            query = query.filter(Detection.timestamp >= start_time)
        
        if end_time is not None:
            query = query.filter(Detection.timestamp <= end_time)
        
        total = query.count()
        
        by_class = query.with_entities(
            Detection.class_name,
            func.count(Detection.id).label('count')
        ).group_by(Detection.class_name).all()
        
        avg_confidence = query.with_entities(
            func.avg(Detection.confidence)
        ).scalar() or 0.0
        
        return {
            "total": total,
            "by_class": {cls: count for cls, count in by_class},
            "avg_confidence": float(avg_confidence)
        }
    
    def delete_old(self, days: int = 30) -> int:
        """删除旧记录"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        deleted = self.db.query(Detection)\
            .filter(Detection.timestamp < cutoff_date)\
            .delete()
        self.db.commit()
        return deleted


class AlertRepository:
    """警报仓储"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, **kwargs) -> Alert:
        """创建警报"""
        alert = Alert(**kwargs)
        self.db.add(alert)
        self.db.commit()
        self.db.refresh(alert)
        return alert
    
    def get_all(
        self,
        acknowledged: Optional[bool] = None,
        limit: int = 100
    ) -> List[Alert]:
        """获取所有警报"""
        query = self.db.query(Alert)
        
        if acknowledged is not None:
            query = query.filter(Alert.acknowledged == acknowledged)
        
        return query.order_by(Alert.created_at.desc()).limit(limit).all()
    
    def get_by_camera(
        self,
        camera_id: int,
        limit: int = 50
    ) -> List[Alert]:
        """获取摄像头的警报"""
        return self.db.query(Alert)\
            .filter(Alert.camera_id == camera_id)\
            .order_by(Alert.created_at.desc())\
            .limit(limit)\
            .all()
    
    def acknowledge(self, alert_id: int, acknowledged_by: str) -> bool:
        """确认警报"""
        alert = self.db.query(Alert).filter(Alert.id == alert_id).first()
        if alert:
            alert.acknowledged = True
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = acknowledged_by
            self.db.commit()
            return True
        return False
    
    def delete_old(self, days: int = 7) -> int:
        """删除旧警报"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        deleted = self.db.query(Alert)\
            .filter(Alert.created_at < cutoff_date)\
            .delete()
        self.db.commit()
        return deleted


class UserRepository:
    """用户仓储"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取用户"""
        return self.db.query(User).filter(User.username == username).first()
    
    def get_by_email(self, email: str) -> Optional[User]:
        """根据邮箱获取用户"""
        return self.db.query(User).filter(User.email == email).first()
    
    def get_by_id(self, user_id: int) -> Optional[User]:
        """根据ID获取用户"""
        return self.db.query(User).filter(User.id == user_id).first()
    
    def get_all(self, skip: int = 0, limit: int = 100) -> List[User]:
        """获取所有用户"""
        return self.db.query(User)\
            .offset(skip)\
            .limit(limit)\
            .all()
    
    def create(self, **kwargs) -> User:
        """创建用户"""
        user = User(**kwargs)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user
    
    def update_last_login(self, user_id: int):
        """更新最后登录时间"""
        user = self.get_by_id(user_id)
        if user:
            user.last_login = datetime.utcnow()
            self.db.commit()


class SettingsRepository:
    """设置仓储"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get(self, category: str, key: str) -> Optional[Settings]:
        """获取设置"""
        return self.db.query(Settings)\
            .filter(Settings.category == category)\
            .filter(Settings.key == key)\
            .first()
    
    def get_all(self) -> List[Settings]:
        """获取所有设置"""
        return self.db.query(Settings).all()
    
    def get_by_category(self, category: str) -> List[Settings]:
        """获取类别的所有设置"""
        return self.db.query(Settings)\
            .filter(Settings.category == category)\
            .all()
    
    def set(self, category: str, key: str, value: str, value_type: str = "string") -> Settings:
        """设置值"""
        setting = self.get(category, key)
        if setting:
            setting.value = value
            setting.value_type = value_type
            setting.updated_at = datetime.utcnow()
        else:
            setting = Settings(
                category=category,
                key=key,
                value=value,
                value_type=value_type
            )
            self.db.add(setting)
        
        self.db.commit()
        self.db.refresh(setting)
        return setting
    
    def delete(self, category: str, key: str) -> bool:
        """删除设置"""
        setting = self.get(category, key)
        if setting:
            self.db.delete(setting)
            self.db.commit()
            return True
        return False
