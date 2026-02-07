from fastapi import FastAPI, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pathlib import Path
import uvicorn
from datetime import datetime
import json
import asyncio
from typing import Optional
import cv2
import numpy as np
import threading
import queue
import time

from config import ProcessDetectionConfig
from detector import ProcessDetector
from video_processor import VideoProcessor
from analyzer import ProcessAnalyzer

app = FastAPI(
    title="YOLO Process Detection Web",
    description="Industrial process monitoring using YOLO",
    version="2.0.0"
)

config = ProcessDetectionConfig()
detector = ProcessDetector(config)
processor = VideoProcessor(config)
analyzer = ProcessAnalyzer(config)

# 全局视频流管理器
class CameraStreamManager:
    """摄像头视频流管理器"""
    
    def __init__(self):
        self.camera = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.thread = None
        self.camera_index = 0
        self.is_simulation = False
        self.simulation_frame_count = 0
        
    def start(self, camera_index: int = 0, use_simulation: bool = False):
        """启动摄像头或模拟视频流"""
        self.camera_index = camera_index
        self.is_simulation = use_simulation
        
        if use_simulation:
            # 使用模拟视频流
            self.is_running = True
            self.thread = threading.Thread(target=self._simulate_frames)
            self.thread.daemon = True
            self.thread.start()
            return True
        
        # 尝试打开真实摄像头
        self.camera = cv2.VideoCapture(camera_index)
        
        if not self.camera.isOpened():
            # 如果摄像头不可用，自动切换到模拟模式
            print(f"摄像头 {camera_index} 不可用，切换到模拟模式")
            self.is_simulation = True
            self.is_running = True
            self.thread = threading.Thread(target=self._simulate_frames)
            self.thread.daemon = True
            self.thread.start()
            return True
        
        # 设置摄像头参数
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_frames)
        self.thread.daemon = True
        self.thread.start()
        
        return True
    
    def _capture_frames(self):
        """捕获帧的线程"""
        while self.is_running and not self.is_simulation:
            ret, frame = self.camera.read()
            if ret:
                # 如果队列满了，移除旧帧
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                # 处理帧（检测）
                try:
                    detections, processed_frame = detector.detect_frame(frame)
                    stage = detector.analyze_process_stage(detections)
                    annotated_frame = detector.draw_detections(
                        processed_frame, detections, stage
                    )
                    
                    # 添加FPS信息
                    cv2.putText(annotated_frame, "Live Camera", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    self.frame_queue.put(annotated_frame)
                except Exception as e:
                    print(f"处理帧时出错: {e}")
                    self.frame_queue.put(frame)
            
            time.sleep(0.001)  # 小延迟避免CPU占用过高
    
    def _simulate_frames(self):
        """模拟视频流（用于测试）"""
        print("启动模拟视频流...")
        
        while self.is_running:
            # 创建模拟帧
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # 添加背景
            frame[:] = (50, 50, 50)
            
            # 添加模拟对象（移动的矩形）
            self.simulation_frame_count += 1
            x = 100 + (self.simulation_frame_count * 5) % 400
            y = 200
            
            # 绘制模拟对象
            cv2.rectangle(frame, (x, y), (x + 100, y + 80), (0, 255, 0), 2)
            cv2.putText(frame, "Object", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 添加状态信息
            cv2.putText(frame, "SIMULATION MODE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Frame: {self.simulation_frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, "No Camera Detected", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
            
            # 添加时间戳
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(frame, timestamp, (400, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 如果队列满了，移除旧帧
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.frame_queue.put(frame)
            
            time.sleep(0.033)  # 约30fps
    
    def get_frame(self):
        """获取最新帧"""
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None
    
    def stop(self):
        """停止摄像头"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.camera:
            self.camera.release()
            self.camera = None
        self.is_simulation = False
        self.simulation_frame_count = 0
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

# 全局摄像头管理器实例
camera_manager = CameraStreamManager()

def generate_frames():
    """生成视频流帧"""
    while camera_manager.is_running:
        frame = camera_manager.get_frame()
        if frame is not None:
            # 编码为JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # 如果没有帧，发送一个空白帧或等待
            time.sleep(0.01)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

Path("static").mkdir(exist_ok=True)
Path("static/css").mkdir(parents=True, exist_ok=True)
Path("static/js").mkdir(parents=True, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    try:
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_path = upload_dir / f"video_{timestamp}.mp4"
        
        with open(video_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {
            "success": True,
            "message": "Video uploaded successfully",
            "video_path": str(video_path),
            "filename": file.filename
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/process_video")
async def process_video_endpoint(request: Request):
    try:
        data = await request.json()
        video_path = data.get('video_path')
        
        if not video_path:
            return {"success": False, "error": "No video path provided"}
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"outputs/result_{timestamp}.mp4"
        
        async def process_video_async():
            def process_callback(frame, detections, stage):
                analyzer.record_detections(detections, processor.frame_count, datetime.now())
                current_stage = analyzer.get_current_stage()
                if current_stage != stage:
                    analyzer.record_stage_change(stage, datetime.now())
            
            processor.process_video(video_path, output_path, process_callback)
        
        await asyncio.to_thread(process_video_async)
        
        stats = analyzer.calculate_statistics()
        efficiency = analyzer.analyze_process_efficiency()
        
        return {
            "success": True,
            "message": "Video processed successfully",
            "output_path": output_path,
            "statistics": stats,
            "efficiency": efficiency
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/statistics")
async def get_statistics():
    try:
        stats = analyzer.calculate_statistics()
        return {"success": True, "data": stats}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/efficiency")
async def get_efficiency():
    try:
        efficiency = analyzer.analyze_process_efficiency()
        return {"success": True, "data": efficiency}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/timeline")
async def get_timeline():
    try:
        timeline = analyzer.generate_timeline()
        return {"success": True, "data": timeline}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/anomalies")
async def get_anomalies():
    try:
        anomalies = analyzer.detect_anomalies()
        return {"success": True, "data": anomalies}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/reset")
async def reset_analysis():
    try:
        analyzer.reset()
        return {"success": True, "message": "Analysis reset successfully"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/export_results")
async def export_results():
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"outputs/results_{timestamp}.json"
        analyzer.export_results(output_path)
        return {"success": True, "message": f"Results exported to {output_path}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": detector.model is not None
    }


@app.post("/api/camera/start")
async def start_camera(camera_index: int = 0, use_simulation: bool = False):
    """启动摄像头"""
    try:
        if camera_manager.is_running:
            camera_manager.stop()
        
        camera_manager.start(camera_index, use_simulation)
        
        if camera_manager.is_simulation:
            return {
                "success": True,
                "message": f"摄像头 {camera_index} 不可用，已切换到模拟模式",
                "camera_index": camera_index,
                "is_simulation": True
            }
        else:
            return {
                "success": True,
                "message": f"摄像头 {camera_index} 启动成功",
                "camera_index": camera_index,
                "is_simulation": False
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/camera/stop")
async def stop_camera():
    """停止摄像头"""
    try:
        camera_manager.stop()
        return {
            "success": True,
            "message": "摄像头已停止"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/camera/status")
async def camera_status():
    """获取摄像头状态"""
    return {
        "success": True,
        "is_running": camera_manager.is_running,
        "camera_index": camera_manager.camera_index if camera_manager.is_running else None,
        "is_simulation": camera_manager.is_simulation if camera_manager.is_running else False
    }


@app.get("/api/camera/stream")
async def camera_stream():
    """视频流端点"""
    if not camera_manager.is_running:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "摄像头未启动，请先调用 /api/camera/start"}
        )
    
    return StreamingResponse(
        generate_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == "__main__":
    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )