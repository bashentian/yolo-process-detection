"""视频流处理模块

提供视频流的实时检测和流式输出功能。
"""
import asyncio
import cv2
import numpy as np
from typing import Optional, List, AsyncGenerator
from datetime import datetime
import io
import base64

from ..models.detector import YOLODetector
from ..models.tracker import ObjectTracker
from ..services.camera import CameraManager, CameraFrame, get_camera_manager
from ..core.config import get_settings


class StreamProcessor:
    """视频流处理器
    
    处理来自摄像头的视频流，执行检测和跟踪。
    """
    
    def __init__(
        self,
        camera_id: int,
        detector: Optional[YOLODetector] = None,
        tracker: Optional[ObjectTracker] = None
    ):
        self.camera_id = camera_id
        self._camera_manager = get_camera_manager()
        self._settings = get_settings()
        
        self.detector = detector or YOLODetector(
            model_name=self._settings.model_name,
            confidence_threshold=self._settings.confidence_threshold,
            iou_threshold=self._settings.iou_threshold,
            device=self._settings.device
        )
        
        self.tracker = tracker or ObjectTracker(
            max_age=self._settings.max_age,
            min_hits=self._settings.min_hits
        )
        
        self._running = False
        self._processing_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动处理"""
        if self._running:
            return
        
        self._running = True
        self._processing_task = asyncio.create_task(self._process_loop())
    
    async def stop(self):
        """停止处理"""
        self._running = False
        
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
    
    async def _process_loop(self):
        """处理循环"""
        while self._running:
            try:
                camera_frame = await self._camera_manager.get_frame(
                    self.camera_id,
                    timeout=1.0
                )
                
                if camera_frame is None:
                    await asyncio.sleep(0.1)
                    continue
                
                await self._process_frame(camera_frame)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"处理帧时出错: {e}")
                await asyncio.sleep(0.5)
    
    async def _process_frame(self, camera_frame: CameraFrame):
        """处理单帧"""
        detections = self.detector.detect(camera_frame.frame)
        
        if self._settings.tracking_enabled:
            tracked_objects = self.tracker.update(detections)
        else:
            tracked_objects = []
        
        self._camera_manager.update_camera_stats(
            self.camera_id,
            len(detections)
        )
        
        return {
            "camera_id": self.camera_id,
            "timestamp": camera_frame.timestamp.isoformat(),
            "frame_number": camera_frame.frame_number,
            "detections": [
                {
                    "bbox": det.bbox,
                    "confidence": det.confidence,
                    "class_id": det.class_id,
                    "class_name": det.class_name
                }
                for det in detections
            ],
            "tracked_objects": [
                {
                    "track_id": obj.track_id,
                    "bbox": obj.bbox,
                    "class_name": obj.class_name,
                    "age": obj.age
                }
                for obj in tracked_objects
            ]
        }
    
    def draw_detections(self, frame: np.ndarray, detections: List) -> np.ndarray:
        """在帧上绘制检测结果"""
        frame_copy = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            conf = det.confidence
            class_name = det.class_name
            
            color = self._get_class_color(det.class_id)
            
            cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            label = f"{class_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                frame_copy,
                (int(x1), int(y1) - label_size[1] - 10),
                (int(x1) + label_size[0], int(y1)),
                color,
                -1
            )
            cv2.putText(
                frame_copy,
                label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        return frame_copy
    
    def _get_class_color(self, class_id: int) -> tuple:
        """获取类别颜色"""
        colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255)
        ]
        return colors[class_id % len(colors)]


class MJPEGStreamer:
    """MJPEG流式输出器
    
    生成MJPEG格式的视频流，用于Web浏览器显示。
    """
    
    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self._camera_manager = get_camera_manager()
        self._processor: Optional[StreamProcessor] = None
        self._clients: List[asyncio.Queue] = []
        self._running = False
        self._stream_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动流"""
        if self._running:
            return
        
        self._processor = StreamProcessor(self.camera_id)
        await self._processor.start()
        
        self._running = True
        self._stream_task = asyncio.create_task(self._stream_loop())
    
    async def stop(self):
        """停止流"""
        self._running = False
        
        if self._processor:
            await self._processor.stop()
        
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        
        for client in self._clients:
            await client.put(None)
        self._clients.clear()
    
    def add_client(self) -> asyncio.Queue:
        """添加客户端"""
        client_queue = asyncio.Queue(maxsize=10)
        self._clients.append(client_queue)
        return client_queue
    
    def remove_client(self, client_queue: asyncio.Queue):
        """移除客户端"""
        if client_queue in self._clients:
            self._clients.remove(client_queue)
    
    async def _stream_loop(self):
        """流循环"""
        while self._running:
            try:
                camera_frame = await self._camera_manager.get_frame(
                    self.camera_id,
                    timeout=1.0
                )
                
                if camera_frame is None:
                    await asyncio.sleep(0.1)
                    continue
                
                frame_bytes = self._encode_frame(camera_frame.frame)
                
                for client in self._clients:
                    try:
                        client.put_nowait(frame_bytes)
                    except asyncio.QueueFull:
                        pass
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"流处理出错: {e}")
                await asyncio.sleep(0.5)
    
    def _encode_frame(self, frame: np.ndarray) -> bytes:
        """编码帧为JPEG"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, encoded = cv2.imencode('.jpg', frame, encode_param)
        return encoded.tobytes()
    
    async def generate_stream(self) -> AsyncGenerator[bytes, None]:
        """生成MJPEG流"""
        boundary = b"--frame"
        
        client_queue = self.add_client()
        
        try:
            while True:
                frame_bytes = await client_queue.get()
                
                if frame_bytes is None:
                    break
                
                yield (
                    boundary + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
                    + frame_bytes + b"\r\n"
                )
        
        finally:
            self.remove_client(client_queue)


class FrameStreamer:
    """帧流式输出器
    
    生成JSON格式的帧数据流，用于WebSocket推送。
    """
    
    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self._camera_manager = get_camera_manager()
        self._processor: Optional[StreamProcessor] = None
        self._running = False
        self._stream_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动流"""
        if self._running:
            return
        
        self._processor = StreamProcessor(self.camera_id)
        await self._processor.start()
        
        self._running = True
        self._stream_task = asyncio.create_task(self._stream_loop())
    
    async def stop(self):
        """停止流"""
        self._running = False
        
        if self._processor:
            await self._processor.stop()
        
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
    
    async def _stream_loop(self):
        """流循环"""
        while self._running:
            try:
                camera_frame = await self._camera_manager.get_frame(
                    self.camera_id,
                    timeout=1.0
                )
                
                if camera_frame is None:
                    await asyncio.sleep(0.1)
                    continue
                
                frame_data = await self._processor._process_frame(camera_frame)
                
                yield frame_data
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"帧流处理出错: {e}")
                await asyncio.sleep(0.5)
    
    async def generate_frames(self) -> AsyncGenerator[dict, None]:
        """生成帧数据流"""
        async for frame_data in self._stream_loop():
            yield frame_data


_streamers: dict[int, MJPEGStreamer] = {}


def get_streamer(camera_id: int) -> MJPEGStreamer:
    """获取流式输出器"""
    if camera_id not in _streamers:
        _streamers[camera_id] = MJPEGStreamer(camera_id)
    return _streamers[camera_id]


def cleanup_streamers():
    """清理所有流式输出器"""
    for streamer in _streamers.values():
        asyncio.create_task(streamer.stop())
    _streamers.clear()
