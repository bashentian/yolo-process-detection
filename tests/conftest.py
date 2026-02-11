"""测试配置文件

提供pytest fixtures和测试配置。
"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, AsyncMock
from typing import Generator


@pytest.fixture
def sample_frame() -> np.ndarray:
    """生成测试用的示例帧"""
    return np.zeros((640, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_detections() -> list:
    """生成测试用的示例检测结果"""
    return [
        (100.0, 100.0, 200.0, 200.0, 0.95, 0, "worker"),
        (300.0, 300.0, 400.0, 400.0, 0.85, 1, "machine"),
    ]


@pytest.fixture
def mock_settings():
    """模拟设置对象"""
    from pydantic_settings import BaseSettings
    
    class MockSettings(BaseSettings):
        model_name: str = "yolo11n.pt"
        confidence_threshold: float = 0.5
        iou_threshold: float = 0.45
        device: str = "cpu"
        tracking_enabled: bool = True
        tracking_max_age: int = 30
        scene_understanding_enabled: bool = True
    
    return MockSettings()


@pytest.fixture
def mock_detector():
    """模拟检测器"""
    detector = Mock()
    detector.detect.return_value = []
    detector.__call__ = detector.detect
    return detector


@pytest.fixture
def mock_tracker():
    """模拟跟踪器"""
    tracker = Mock()
    tracker.update.return_value = []
    tracker.reset.return_value = None
    return tracker


@pytest.fixture
def temp_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """提供临时目录"""
    (tmp_path / "models").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "outputs").mkdir()
    (tmp_path / "uploads").mkdir()
    (tmp_path / "cache").mkdir()
    (tmp_path / "logs").mkdir()
    yield tmp_path


@pytest.fixture
def mock_upload_file():
    """模拟上传文件"""
    from fastapi import UploadFile
    from io import BytesIO
    
    content = b"fake image content"
    file = BytesIO(content)
    
    upload_file = Mock(spec=UploadFile)
    upload_file.filename = "test.jpg"
    upload_file.content_type = "image/jpeg"
    upload_file.read = AsyncMock(return_value=content)
    upload_file.seek = Mock()
    upload_file.write = Mock()
    upload_file.close = Mock()
    
    return upload_file


@pytest.fixture
def api_client():
    """创建测试API客户端"""
    from fastapi.testclient import TestClient
    from ...main import app
    
    with TestClient(app) as client:
        yield client


def pytest_collection_modifyitems(config, items):
    """修改测试项目收集"""
    for item in items:
        if item.get_closest_marker("slow") is None:
            item.add_marker(pytest.mark.fast)
