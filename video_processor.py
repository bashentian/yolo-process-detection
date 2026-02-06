import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime
import threading

from detector import ProcessDetector
from tracker import ObjectTracker
from config import ProcessDetectionConfig


class VideoProcessor:
    def __init__(self, config: ProcessDetectionConfig):
        self.config = config
        self.detector = ProcessDetector(config)
        self.tracker = ObjectTracker(
            max_age=config.TRACKING_MAX_AGE,
            min_hits=config.TRACKING_MIN_HITS
        )
        
        self.video_source = None
        self.video_writer = None
        self.is_running = False
        self.current_frame = None
        self.current_detections = []
        self.current_stage = "idle"
        self.frame_count = 0
        
    def initialize_video_source(self, source):
        if isinstance(source, (str, Path)):
            self.video_source = cv2.VideoCapture(str(source))
        elif isinstance(source, int):
            self.video_source = cv2.VideoCapture(source)
        else:
            raise ValueError("Source must be file path or camera index")
        
        if not self.video_source.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        
        fps = self.video_source.get(cv2.CAP_PROP_FPS)
        width = int(self.video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video source initialized: {width}x{height} @ {fps} FPS")
        
        return width, height, fps
    
    def initialize_video_writer(self, output_path: str, width: int, height: int, fps: float):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.video_writer = cv2.VideoWriter(
            output_path, fourcc, fps, (width, height)
        )
        
        if not self.video_writer.isOpened():
            raise RuntimeError(f"Cannot create video writer: {output_path}")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self.frame_count += 1
        
        detections, processed_frame = self.detector.detect_frame(frame)
        tracked_objects = self.tracker.update(detections)
        
        self.current_stage = self.detector.analyze_process_stage(detections)
        
        annotated_frame = self.detector.draw_detections(
            processed_frame, detections, self.current_stage
        )
        
        for track in tracked_objects:
            x, y = track.detection.center
            label = f"ID:{track.track_id}"
            cv2.putText(annotated_frame, label,
                       (int(x), int(y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        self.current_frame = annotated_frame
        self.current_detections = detections
        
        return annotated_frame
    
    def process_video(self, source, output_path: Optional[str] = None,
                     callback: Optional[Callable] = None):
        width, height, fps = self.initialize_video_source(source)
        
        if output_path and self.config.SAVE_RESULTS:
            self.initialize_video_writer(output_path, width, height, fps)
        
        self.is_running = True
        
        try:
            while self.is_running:
                ret, frame = self.video_source.read()
                
                if not ret:
                    print("Video processing completed")
                    break
                
                annotated_frame = self.process_frame(frame)
                
                if self.video_writer:
                    self.video_writer.write(annotated_frame)
                
                if callback:
                    callback(annotated_frame, self.current_detections, self.current_stage)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False
                    break
                
        finally:
            self.release_resources()
    
    def start_processing(self, source, output_path: Optional[str] = None):
        def default_callback(frame, detections, stage):
            display_frame = cv2.resize(frame, self.config.DISPLAY_SIZE)
            cv2.imshow('Process Detection', display_frame)
        
        self.process_video(source, output_path, default_callback)
    
    def process_frame_async(self, frame: np.ndarray) -> np.ndarray:
        return self.process_frame(frame)
    
    def get_current_detections(self):
        return self.current_detections
    
    def get_current_stage(self):
        return self.current_stage
    
    def get_statistics(self) -> dict:
        return {
            "frame_count": self.frame_count,
            "current_stage": self.current_stage,
            "detection_count": len(self.current_detections),
            "tracked_objects": len(self.tracker.tracks)
        }
    
    def release_resources(self):
        self.is_running = False
        
        if self.video_source:
            self.video_source.release()
        
        if self.video_writer:
            self.video_writer.release()
        
        cv2.destroyAllWindows()
    
    def save_frame(self, frame: np.ndarray, output_path: str):
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, frame)
    
    def extract_key_frames(self, output_dir: str, interval: int = 30):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        frame_idx = 0
        key_frame_idx = 0
        
        while True:
            ret, frame = self.video_source.read()
            if not ret:
                break
            
            if frame_idx % interval == 0:
                key_frame_path = output_path / f"keyframe_{key_frame_idx:06d}.jpg"
                annotated_frame = self.process_frame(frame)
                cv2.imwrite(str(key_frame_path), annotated_frame)
                key_frame_idx += 1
            
            frame_idx += 1
