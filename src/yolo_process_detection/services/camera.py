"""摄像头管理模块

提供摄像头设备的注册、管理和控制功能。
"""
import asyncio
import threading
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import cv2
import numpy as np

from ..core.config import get_settings


class CameraStatus(str, Enum):
    """摄像头状态"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class CameraInfo:
    """摄像头信息"""
    id: int
    name: str
    location: str
    source: str
    resolution: str = "1920x1080"
    fps: int = 30
    status: CameraStatus = CameraStatus.STOPPED
    last_frame_time: Optional[datetime] = None
    frame_count: int = 0
    detection_count: int = 0
    error_message: Optional[str] = None


@dataclass
class CameraFrame:
    """摄像头帧数据"""
    camera_id: int
    frame: np.ndarray
    timestamp: datetime
    frame_number: int


class CameraManager:
    """摄像头管理器
    
    管理所有摄像头的生命周期，包括启动、停止、状态监控等。
    """
    
    def __init__(self):
        self._cameras: Dict[int, CameraInfo] = {}
        self._captures: Dict[int, cv2.VideoCapture] = {}
        self._frame_queues: Dict[int, asyncio.Queue] = {}
        self._capture_threads: Dict[int, threading.Thread] = {}
        self._running = False
        self._lock = threading.Lock()
        self._settings = get_settings()
    
    def register_camera(
        self,
        camera_id: int,
        name: str,
        location: str,
        source: str,
        resolution: str = "1920x1080",
        fps: int = 30
    ) -> CameraInfo:
        """注册摄像头"""
        with self._lock:
            if camera_id in self._cameras:
                raise ValueError(f"摄像头 {camera_id} 已存在")
            
            camera = CameraInfo(
                id=camera_id,
                name=name,
                location=location,
                source=source,
                resolution=resolution,
                fps=fps,
                status=CameraStatus.STOPPED
            )
            
            self._cameras[camera_id] = camera
            self._frame_queues[camera_id] = asyncio.Queue(maxsize=32)
            
            return camera
    
    def unregister_camera(self, camera_id: int) -> bool:
        """注销摄像头"""
        with self._lock:
            if camera_id not in self._cameras:
                return False
            
            if self._cameras[camera_id].status == CameraStatus.RUNNING:
                self.stop_camera(camera_id)
            
            del self._cameras[camera_id]
            del self._frame_queues[camera_id]
            
            return True
    
    def get_camera(self, camera_id: int) -> Optional[CameraInfo]:
        """获取摄像头信息"""
        with self._lock:
            return self._cameras.get(camera_id)
    
    def get_all_cameras(self) -> List[CameraInfo]:
        """获取所有摄像头"""
        with self._lock:
            return list(self._cameras.values())
    
    def start_camera(self, camera_id: int) -> bool:
        """启动摄像头"""
        with self._lock:
            camera = self._cameras.get(camera_id)
            if not camera:
                raise ValueError(f"摄像头 {camera_id} 不存在")
            
            if camera.status == CameraStatus.RUNNING:
                return True
            
            camera.status = CameraStatus.STARTING
            camera.error_message = None
        
        try:
            capture = cv2.VideoCapture(camera.source)
            if not capture.isOpened():
                raise RuntimeError(f"无法打开摄像头 {camera_id}: {camera.source}")
            
            width, height = map(int, camera.resolution.split('x'))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv2.CAP_PROP_FPS, camera.fps)
            
            with self._lock:
                self._captures[camera_id] = capture
                camera.status = CameraStatus.RUNNING
            
            thread = threading.Thread(
                target=self._capture_loop,
                args=(camera_id,),
                daemon=True
            )
            thread.start()
            self._capture_threads[camera_id] = thread
            
            return True
        
        except Exception as e:
            with self._lock:
                camera.status = CameraStatus.ERROR
                camera.error_message = str(e)
            return False
    
    def stop_camera(self, camera_id: int) -> bool:
        """停止摄像头"""
        with self._lock:
            camera = self._cameras.get(camera_id)
            if not camera:
                return False
            
            if camera.status != CameraStatus.RUNNING:
                return True
            
            camera.status = CameraStatus.STOPPING
        
        capture = self._captures.pop(camera_id, None)
        if capture:
            capture.release()
        
        thread = self._capture_threads.pop(camera_id, None)
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        
        with self._lock:
            camera.status = CameraStatus.STOPPED
        
        return True
    
    def start_all(self) -> Dict[int, bool]:
        """启动所有摄像头"""
        results = {}
        for camera_id in self._cameras:
            results[camera_id] = self.start_camera(camera_id)
        return results
    
    def stop_all(self) -> Dict[int, bool]:
        """停止所有摄像头"""
        results = {}
        for camera_id in list(self._cameras.keys()):
            results[camera_id] = self.stop_camera(camera_id)
        return results
    
    async def get_frame(self, camera_id: int, timeout: float = 1.0) -> Optional[CameraFrame]:
        """获取摄像头帧"""
        queue = self._frame_queues.get(camera_id)
        if not queue:
            return None
        
        try:
            return await asyncio.wait_for(queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
    
    def update_camera_stats(self, camera_id: int, detection_count: int = 0):
        """更新摄像头统计信息"""
        with self._lock:
            camera = self._cameras.get(camera_id)
            if camera:
                camera.detection_count += detection_count
    
    def _capture_loop(self, camera_id: int):
        """摄像头捕获循环"""
        camera = self._cameras.get(camera_id)
        capture = self._captures.get(camera_id)
        queue = self._frame_queues.get(camera_id)
        
        if not camera or not capture or not queue:
            return
        
        frame_number = 0
        fps = camera.fps
        frame_interval = 1.0 / fps if fps > 0 else 0.033
        
        while camera.status == CameraStatus.RUNNING:
            try:
                ret, frame = capture.read()
                if not ret:
                    raise RuntimeError("无法读取帧")
                
                frame_number += 1
                camera_frame = CameraFrame(
                    camera_id=camera_id,
                    frame=frame,
                    timestamp=datetime.now(),
                    frame_number=frame_number
                )
                
                camera.last_frame_time = camera_frame.timestamp
                camera.frame_count = frame_number
                
                if not queue.full():
                    try:
                        queue.put_nowait(camera_frame)
                    except asyncio.QueueFull:
                        pass
                
                import time
                time.sleep(frame_interval)
            
            except Exception as e:
                with self._lock:
                    camera.status = CameraStatus.ERROR
                    camera.error_message = str(e)
                break
    
    def cleanup(self):
        """清理资源"""
        self.stop_all()
        
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()
        
        for thread in self._capture_threads.values():
            if thread.is_alive():
                thread.join(timeout=2.0)
        self._capture_threads.clear()


_camera_manager: Optional[CameraManager] = None


def get_camera_manager() -> CameraManager:
    """获取摄像头管理器单例"""
    global _camera_manager
    if _camera_manager is None:
        _camera_manager = CameraManager()
    return _camera_manager


def initialize_default_cameras():
    """初始化默认摄像头"""
    manager = get_camera_manager()
    
    default_cameras = [
        {
            "id": 0,
            "name": "摄像头 01",
            "location": "生产线A",
            "source": 0,
            "resolution": "1920x1080",
            "fps": 30
        },
        {
            "id": 1,
            "name": "摄像头 02",
            "location": "生产线B",
            "source": 1,
            "resolution": "1920x1080",
            "fps": 30
        },
        {
            "id": 2,
            "name": "摄像头 03",
            "location": "仓库入口",
            "source": 2,
            "resolution": "1920x1080",
            "fps": 30
        },
        {
            "id": 3,
            "name": "摄像头 04",
            "location": "包装区",
            "source": 3,
            "resolution": "1920x1080",
            "fps": 30
        }
    ]
    
    for cam_config in default_cameras:
        try:
            manager.register_camera(**cam_config)
        except ValueError:
            pass
