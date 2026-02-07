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


if __name__ == '__main__':
    templates_dir = Path(__file__).parent / 'templates'
    templates_dir.mkdir(exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
