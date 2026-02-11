"""数据库管理模块

提供数据库连接、会话管理和初始化功能。
"""
from pathlib import Path
from typing import Generator, Optional
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .models import Base, Camera, Detection, DetectionStatistics, SystemEvent, Alert, User, Settings
from ..core.config import get_settings


class DatabaseManager:
    """数据库管理器
    
    管理数据库连接、会话和初始化。
    """
    
    def __init__(self, database_url: Optional[str] = None):
        self._settings = get_settings()
        
        if database_url is None:
            db_path = self._settings.data_root / "detection.db"
            database_url = f"sqlite:///{db_path}"
        
        self._engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False
        )
        
        self._SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self._engine
        )
    
    def create_tables(self):
        """创建所有表"""
        Base.metadata.create_all(bind=self._engine, checkfirst=True)
    
    def drop_tables(self):
        """删除所有表"""
        Base.metadata.drop_all(bind=self._engine)
    
    def init_db(self):
        """初始化数据库"""
        self.create_tables()
        self._seed_default_data()
    
    def _seed_default_data(self):
        """插入默认数据"""
        session = self.get_session()
        
        try:
            if not session.query(Camera).first():
                default_cameras = [
                    Camera(
                        id=0,
                        name="摄像头 01",
                        location="生产线A",
                        source="0",
                        resolution="1920x1080",
                        fps=30
                    ),
                    Camera(
                        id=1,
                        name="摄像头 02",
                        location="生产线B",
                        source="1",
                        resolution="1920x1080",
                        fps=30
                    ),
                    Camera(
                        id=2,
                        name="摄像头 03",
                        location="仓库入口",
                        source="2",
                        resolution="1920x1080",
                        fps=30
                    ),
                    Camera(
                        id=3,
                        name="摄像头 04",
                        location="包装区",
                        source="3",
                        resolution="1920x1080",
                        fps=30
                    )
                ]
                session.add_all(default_cameras)
            
            if not session.query(User).first():
                admin_user = User(
                    username="admin",
                    email="admin@example.com",
                    hashed_password="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5NU7vBj3q0iG",
                    full_name="系统管理员",
                    role="admin",
                    is_active=True
                )
                session.add(admin_user)
            
            session.commit()
        
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_session(self) -> Session:
        """获取数据库会话"""
        return self._SessionLocal()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """会话上下文管理器"""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def close(self):
        """关闭数据库连接"""
        self._engine.dispose()


_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """获取数据库管理器单例"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def get_db() -> Generator[Session, None, None]:
    """获取数据库会话（用于FastAPI依赖注入）"""
    db_manager = get_db_manager()
    db = db_manager.get_session()
    try:
        yield db
    finally:
        db.close()


def init_database():
    """初始化数据库"""
    db_manager = get_db_manager()
    db_manager.init_db()


def reset_database():
    """重置数据库"""
    db_manager = get_db_manager()
    db_manager.drop_tables()
    db_manager.init_db()
