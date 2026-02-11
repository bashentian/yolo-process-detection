"""数据库模块初始化"""
from .models import (
    Base,
    Camera,
    Detection,
    DetectionStatistics,
    SystemEvent,
    Alert,
    User,
    Settings
)
from .manager import (
    DatabaseManager,
    get_db_manager,
    get_db,
    init_database,
    reset_database
)
from .repositories import (
    CameraRepository,
    DetectionRepository,
    AlertRepository,
    UserRepository,
    SettingsRepository
)

__all__ = [
    "Base",
    "Camera",
    "Detection",
    "DetectionStatistics",
    "SystemEvent",
    "Alert",
    "User",
    "Settings",
    "DatabaseManager",
    "get_db_manager",
    "get_db",
    "init_database",
    "reset_database",
    "CameraRepository",
    "DetectionRepository",
    "AlertRepository",
    "UserRepository",
    "SettingsRepository"
]
