from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import base64

from video_processor import VideoProcessor
from analyzer import ProcessAnalyzer
from detector import ProcessDetector
from config import ProcessDetectionConfig


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

config = ProcessDetectionConfig()
processor = VideoProcessor(config)
analyzer = ProcessAnalyzer(config)


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
    
    upload_dir = Path('uploads')
    upload_dir.mkdir(exist_ok=True)
    
    video_path = upload_dir / f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    file.save(video_path)
    
    return jsonify({
        'message': 'Video uploaded successfully',
        'video_path': str(video_path)
    })


@app.route('/api/process_video', methods=['POST'])
def process_video():
    data = request.json
    video_path = data.get('video_path')
    
    if not video_path:
        return jsonify({'error': 'No video path provided'}), 400
    
    output_path = f"outputs/result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    
    try:
        def process_callback(frame, detections, stage):
            analyzer.record_detections(detections, processor.frame_count, datetime.now())
            current_stage = analyzer.get_current_stage()
            if current_stage != stage:
                analyzer.record_stage_change(stage, datetime.now())
        
        processor.process_video(video_path, output_path, process_callback)
        
        stats = analyzer.calculate_statistics()
        efficiency = analyzer.analyze_process_efficiency()
        
        return jsonify({
            'message': 'Video processed successfully',
            'output_path': output_path,
            'statistics': stats,
            'efficiency': efficiency
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stream_video')
def stream_video():
    def generate():
        cap = cv2.VideoCapture(config.VIDEO_SOURCE)
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
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
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
    
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
    output_path = request.json.get('output_path', 'outputs/results.json')
    
    try:
        analyzer.export_results(output_path)
        return jsonify({'message': f'Results exported to {output_path}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reset', methods=['POST'])
def reset_analysis():
    analyzer.reset()
    return jsonify({'message': 'Analysis reset successfully'})


if __name__ == '__main__':
    templates_dir = Path(__file__).parent / 'templates'
    templates_dir.mkdir(exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
