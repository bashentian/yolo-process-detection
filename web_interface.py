from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import base64
import os
import logging
from contextlib import contextmanager

from video_processor import VideoProcessor
from analyzer import ProcessAnalyzer
from detector import ProcessDetector
from config import ProcessDetectionConfig


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

config = ProcessDetectionConfig()
processor = VideoProcessor(config)
analyzer = ProcessAnalyzer(config)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path('uploads').resolve()
OUTPUT_DIR = Path('outputs').resolve()
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}


def validate_safe_path(path_str: str, base_dir: Path) -> tuple[bool, Path]:
    """
    验证路径是否在允许的目录内，防止路径遍历攻击
    Returns: (is_valid, resolved_path)
    """
    try:
        target = Path(path_str).resolve()
        base = base_dir.resolve()
        
        if not str(target).startswith(str(base)):
            logger.warning(f"Path traversal attempt detected: {path_str}")
            return False, target
        
        return True, target
    except Exception as e:
        logger.error(f"Path validation error: {e}")
        return False, Path(path_str)


def allowed_file(filename: str) -> bool:
    """检查文件扩展名是否允许"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@contextmanager
def safe_video_capture(source):
    """安全地管理视频捕获资源"""
    cap = None
    try:
        cap = cv2.VideoCapture(source)
        yield cap
    finally:
        if cap is not None:
            cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/camera_test')
def camera_test():
    """摄像头测试页面"""
    return render_template('camera_test.html')


@app.route('/api/health')
def health_check():
    """健康检查接口"""
    import torch
    
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {}
    }
    
    # 检查模型加载状态
    try:
        if processor.detector.model is not None:
            health_status['services']['model'] = {
                'status': 'ok',
                'model_name': config.MODEL_NAME,
                'device': config.DEVICE
            }
        else:
            health_status['services']['model'] = {
                'status': 'error',
                'message': 'Model not loaded'
            }
            health_status['status'] = 'degraded'
    except Exception as e:
        health_status['services']['model'] = {
            'status': 'error',
            'message': str(e)
        }
        health_status['status'] = 'degraded'
    
    # 检查 CUDA 可用性
    try:
        cuda_available = torch.cuda.is_available()
        health_status['services']['cuda'] = {
            'status': 'ok' if cuda_available else 'not_available',
            'available': cuda_available
        }
        if cuda_available:
            health_status['services']['cuda']['device_count'] = torch.cuda.device_count()
            health_status['services']['cuda']['current_device'] = torch.cuda.current_device()
    except Exception as e:
        health_status['services']['cuda'] = {
            'status': 'error',
            'message': str(e)
        }
    
    # 检查摄像头状态
    try:
        if _camera_capture is not None and _camera_capture.isOpened():
            health_status['services']['camera'] = {
                'status': 'running',
                'camera_index': _camera_index
            }
        else:
            health_status['services']['camera'] = {
                'status': 'stopped'
            }
    except Exception as e:
        health_status['services']['camera'] = {
            'status': 'error',
            'message': str(e)
        }
    
    # 检查目录权限
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 测试写入权限
        test_file = UPLOAD_DIR / '.health_check'
        test_file.touch()
        test_file.unlink()
        
        health_status['services']['filesystem'] = {
            'status': 'ok',
            'upload_dir': str(UPLOAD_DIR),
            'output_dir': str(OUTPUT_DIR)
        }
    except Exception as e:
        health_status['services']['filesystem'] = {
            'status': 'error',
            'message': str(e)
        }
        health_status['status'] = 'degraded'
    
    # 根据状态返回相应的 HTTP 状态码
    status_code = 200 if health_status['status'] == 'healthy' else 503
    
    return jsonify(health_status), status_code


@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
    
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_filename = f"video_{timestamp}.mp4"
    video_path = UPLOAD_DIR / safe_filename
    
    try:
        file.save(video_path)
        logger.info(f"Video uploaded successfully: {video_path}")
        
        return jsonify({
            'message': 'Video uploaded successfully',
            'video_path': str(video_path)
        })
    except Exception as e:
        logger.error(f"Failed to save uploaded video: {e}")
        return jsonify({'error': 'Failed to save video file'}), 500


@app.route('/api/process_video', methods=['POST'])
def process_video():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON data'}), 400
    
    video_path = data.get('video_path')
    
    if not video_path:
        return jsonify({'error': 'No video path provided'}), 400
    
    is_valid, validated_path = validate_safe_path(video_path, UPLOAD_DIR)
    if not is_valid:
        return jsonify({'error': 'Invalid video path'}), 403
    
    if not validated_path.exists():
        return jsonify({'error': 'Video file not found'}), 404
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = OUTPUT_DIR / f"result_{timestamp}.mp4"
    
    try:
        def process_callback(frame, detections, stage):
            analyzer.record_detections(detections, processor.frame_count, datetime.now())
            current_stage = analyzer.get_current_stage()
            if current_stage != stage:
                analyzer.record_stage_change(stage, datetime.now())
        
        processor.process_video(str(validated_path), str(output_path), process_callback)
        
        stats = analyzer.calculate_statistics()
        efficiency = analyzer.analyze_process_efficiency()
        
        return jsonify({
            'message': 'Video processed successfully',
            'output_path': str(output_path),
            'statistics': stats,
            'efficiency': efficiency
        })
    
    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        return jsonify({'error': 'Failed to process video. Please check server logs.'}), 500


@app.route('/api/stream_video')
def stream_video():
    def generate():
        frame_count = 0
        max_frames_without_client = 30  # 如果30帧没有客户端读取，断开连接
        
        with safe_video_capture(config.VIDEO_SOURCE) as cap:
            if not cap.isOpened():
                logger.error("Failed to open video source for streaming")
                return
            
            while True:
                try:
                    ret, frame = cap.read()
                    
                    if not ret:
                        logger.warning("Failed to read frame from video source")
                        break
                    
                    detections, processed_frame = processor.detector.detect_frame(frame)
                    stage = processor.detector.analyze_process_stage(detections)
                    
                    annotated_frame = processor.detector.draw_detections(
                        processed_frame, detections, stage
                    )
                    
                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    frame_bytes = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                    frame_count += 1
                    
                except GeneratorExit:
                    logger.info("Client disconnected from video stream")
                    break
                except Exception as e:
                    logger.error(f"Error in video stream: {e}")
                    break
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/statistics')
def get_statistics():
    stats = analyzer.calculate_statistics()
    return jsonify(stats)


@app.route('/api/efficiency')
def get_efficiency():
    efficiency = analyzer.analyze_process_efficiency()
    return jsonify(efficiency)


@app.route('/api/timeline')
def get_timeline():
    timeline = analyzer.generate_timeline()
    return jsonify(timeline)


@app.route('/api/anomalies')
def get_anomalies():
    anomalies = analyzer.detect_anomalies()
    return jsonify(anomalies)


@app.route('/api/export_results', methods=['POST'])
def export_results():
    data = request.get_json() or {}
    output_path = data.get('output_path', 'outputs/results.json')
    
    is_valid, validated_path = validate_safe_path(output_path, OUTPUT_DIR)
    if not is_valid:
        return jsonify({'error': 'Invalid output path'}), 403
    
    try:
        analyzer.export_results(str(validated_path))
        return jsonify({'message': f'Results exported to {output_path}'})
    except Exception as e:
        logger.error(f"Error exporting results: {e}", exc_info=True)
        return jsonify({'error': 'Failed to export results. Please check server logs.'}), 500


@app.route('/api/reset', methods=['POST'])
def reset_analysis():
    analyzer.reset()
    return jsonify({'message': 'Analysis reset successfully'})


# ==================== 摄像头控制 API ====================

# 全局变量存储摄像头状态
_camera_capture = None
_camera_index = 0
_is_camera_running = False


@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """启动摄像头"""
    global _camera_capture, _camera_index, _is_camera_running
    
    camera_index = request.args.get('camera_index', 0, type=int)
    
    try:
        # 如果摄像头已经在运行，先停止
        if _camera_capture is not None:
            _camera_capture.release()
        
        # 打开新摄像头
        _camera_capture = cv2.VideoCapture(camera_index)
        
        if not _camera_capture.isOpened():
            return jsonify({'error': f'Failed to open camera {camera_index}'}), 400
        
        _camera_index = camera_index
        _is_camera_running = True
        
        # 获取摄像头信息
        width = int(_camera_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(_camera_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = _camera_capture.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Camera {camera_index} started: {width}x{height} @ {fps}fps")
        
        return jsonify({
            'message': f'Camera {camera_index} started successfully',
            'camera_index': camera_index,
            'width': width,
            'height': height,
            'fps': fps
        })
        
    except Exception as e:
        logger.error(f"Error starting camera: {e}", exc_info=True)
        return jsonify({'error': 'Failed to start camera'}), 500


@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """停止摄像头"""
    global _camera_capture, _is_camera_running
    
    try:
        if _camera_capture is not None:
            _camera_capture.release()
            _camera_capture = None
        
        _is_camera_running = False
        
        logger.info("Camera stopped")
        return jsonify({'message': 'Camera stopped successfully'})
        
    except Exception as e:
        logger.error(f"Error stopping camera: {e}", exc_info=True)
        return jsonify({'error': 'Failed to stop camera'}), 500


@app.route('/api/camera/status')
def get_camera_status():
    """获取摄像头状态"""
    global _camera_capture, _camera_index, _is_camera_running
    
    status = {
        'is_running': _is_camera_running,
        'camera_index': _camera_index
    }
    
    if _camera_capture is not None and _camera_capture.isOpened():
        status['width'] = int(_camera_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        status['height'] = int(_camera_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        status['fps'] = _camera_capture.get(cv2.CAP_PROP_FPS)
    
    return jsonify(status)


@app.route('/api/camera/frame')
def get_camera_frame():
    """获取单帧图像"""
    global _camera_capture, _is_camera_running
    
    if _camera_capture is None or not _camera_capture.isOpened():
        return jsonify({'error': 'Camera is not running'}), 400
    
    try:
        # 读取多帧以确保获取最新帧
        for _ in range(3):
            _camera_capture.grab()
        
        ret, frame = _camera_capture.read()
        
        if not ret or frame is None:
            return jsonify({'error': 'Failed to capture frame'}), 500
        
        # 处理帧（添加异常处理）
        try:
            detections, processed_frame = processor.detector.detect_frame(frame)
            stage = processor.detector.analyze_process_stage(detections)
            annotated_frame = processor.detector.draw_detections(
                processed_frame, detections, stage
            )
        except Exception as det_err:
            logger.error(f"Detection error: {det_err}")
            annotated_frame = frame
        
        # 编码为 JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
        result, buffer = cv2.imencode('.jpg', annotated_frame, encode_params)
        
        if not result:
            return jsonify({'error': 'Failed to encode frame'}), 500
        
        frame_bytes = buffer.tobytes()
        
        return Response(
            frame_bytes,
            mimetype='image/jpeg',
            headers={
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        )
        
    except Exception as e:
        logger.error(f"Error capturing frame: {e}", exc_info=True)
        return jsonify({'error': 'Failed to capture frame'}), 500


@app.route('/api/camera/stream')
def camera_stream():
    """摄像头实时视频流 - 优化版本"""
    global _camera_capture, _is_camera_running
    
    if _camera_capture is None or not _camera_capture.isOpened():
        return jsonify({'error': 'Camera is not running. Please call /api/camera/start first'}), 400
    
    def generate():
        global _camera_capture
        import time
        
        frame_count = 0
        start_time = time.time()
        last_yield_time = time.time()
        
        # 预分配缓冲区，避免重复创建
        last_annotated_frame = None
        
        while True:
            try:
                if _camera_capture is None or not _camera_capture.isOpened():
                    logger.warning("Camera not available for streaming")
                    break
                
                # 快速读取帧
                ret = _camera_capture.grab()
                if not ret:
                    continue
                
                # 控制发送帧率 - 目标 20 FPS
                current_time = time.time()
                if current_time - last_yield_time < 0.05:  # 50ms = 20 FPS
                    continue
                
                # 解码帧
                ret, frame = _camera_capture.retrieve()
                if not ret or frame is None:
                    continue
                
                last_yield_time = current_time
                
                # 处理帧 - 使用线程池或简化处理
                try:
                    # 每2帧进行一次检测，平衡性能和流畅度
                    if frame_count % 2 == 0:
                        detections, processed_frame = processor.detector.detect_frame(frame)
                        stage = processor.detector.analyze_process_stage(detections)
                        annotated_frame = processor.detector.draw_detections(
                            processed_frame, detections, stage
                        )
                        last_annotated_frame = annotated_frame.copy()
                    else:
                        # 使用上一帧的检测结果或简单处理
                        if last_annotated_frame is not None:
                            annotated_frame = last_annotated_frame.copy()
                            # 更新时间戳
                            time_str = time.strftime("%H:%M:%S")
                            cv2.putText(annotated_frame, time_str, (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        else:
                            annotated_frame = frame.copy()
                except Exception as det_err:
                    logger.error(f"Detection error: {det_err}")
                    annotated_frame = frame
                
                # 快速调整大小
                height, width = annotated_frame.shape[:2]
                if width > 480:  # 降低分辨率到 480p 提高流畅度
                    annotated_frame = cv2.resize(annotated_frame, (480, int(480 * height / width)))
                
                # 快速编码 - 使用较低质量
                encode_params = [
                    cv2.IMWRITE_JPEG_QUALITY, 60,  # 降低质量提高速度
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1
                ]
                result, buffer = cv2.imencode('.jpg', annotated_frame, encode_params)
                
                if not result:
                    continue
                
                frame_bytes = buffer.tobytes()
                
                # 计算 FPS（每3秒记录一次）
                frame_count += 1
                elapsed = current_time - start_time
                if elapsed >= 3.0:
                    fps = frame_count / elapsed
                    logger.info(f"Camera stream FPS: {fps:.1f}, Frame size: {len(frame_bytes)/1024:.1f}KB")
                    frame_count = 0
                    start_time = current_time
                
                # 立即发送帧
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n'
                       b'\r\n' + frame_bytes + b'\r\n')
                
            except GeneratorExit:
                logger.info("Client disconnected from camera stream")
                break
            except Exception as e:
                logger.error(f"Error in camera stream: {e}", exc_info=True)
                continue
    
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
    )


@app.route('/api/camera/list')
def list_cameras():
    """列出可用的摄像头"""
    available_cameras = []
    
    # 检查前5个摄像头索引
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            available_cameras.append({
                'index': i,
                'width': width,
                'height': height,
                'name': f'Camera {i}'
            })
        cap.release()
    
    return jsonify({
        'cameras': available_cameras,
        'count': len(available_cameras)
    })


if __name__ == '__main__':
    templates_dir = Path(__file__).parent / 'templates'
    templates_dir.mkdir(exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
