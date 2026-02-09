import logging
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional


class Logger:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if Logger._initialized:
            return
        
        Logger._initialized = True
        self.logger = logging.getLogger('YoloProcessDetection')
        self.logger.setLevel(logging.DEBUG)
        
        self._setup_handlers()
    
    def _setup_handlers(self):
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        log_dir = Path(__file__).parent / "outputs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"yolo_detection_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str, exc_info: bool = False):
        self.logger.error(message, exc_info=exc_info)
    
    def critical(self, message: str, exc_info: bool = False):
        self.logger.critical(message, exc_info=exc_info)


logger = Logger()


class ErrorHandler:
    @staticmethod
    def handle_error(error: Exception, context: str = "", 
                   raise_exception: bool = False):
        error_msg = f"Error in {context}: {str(error)}"
        logger.error(error_msg, exc_info=True)
        
        if raise_exception:
            raise error
        return False
    
    @staticmethod
    def validate_file_path(file_path: str, must_exist: bool = True) -> bool:
        path = Path(file_path)
        
        if must_exist and not path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        return True
    
    @staticmethod
    def validate_video_file(video_path: str) -> bool:
        import cv2
        
        if not ErrorHandler.validate_file_path(video_path):
            return False
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {video_path}")
            cap.release()
            return False
        
        cap.release()
        return True
    
    @staticmethod
    def validate_config(config, required_fields: list) -> bool:
        for field in required_fields:
            if not hasattr(config, field):
                logger.error(f"Missing required config field: {field}")
                return False
        return True


class PerformanceMonitor:
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        logger.info(f"Starting: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"Completed: {self.name} (Duration: {duration:.2f}s)")
    
    def get_duration(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


class MemoryMonitor:
    @staticmethod
    def get_memory_usage() -> dict:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'percent': process.memory_percent()
        }
    
    @staticmethod
    def log_memory_usage():
        memory_info = MemoryMonitor.get_memory_usage()
        logger.debug(f"Memory usage: RSS={memory_info['rss_mb']:.2f}MB, "
                    f"VMS={memory_info['vms_mb']:.2f}MB, "
                    f"Percent={memory_info['percent']:.2f}%")


class ModelValidator:
    @staticmethod
    def validate_model_file(model_path: str) -> bool:
        from pathlib import Path
        
        model_file = Path(model_path)
        
        if not model_file.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
        
        if model_file.suffix not in ['.pt', '.onnx', '.engine']:
            logger.error(f"Unsupported model format: {model_file.suffix}")
            return False
        
        return True
    
    @staticmethod
    def validate_detection_results(results, min_confidence: float = 0.0) -> bool:
        if not results or len(results) == 0:
            logger.warning("No detection results returned")
            return True
        
        for result in results:
            if not hasattr(result, 'confidence') or result.confidence < min_confidence:
                logger.warning(f"Detection with low confidence: {result.confidence}")
        
        return True


class DataValidator:
    @staticmethod
    def validate_dataset_structure(data_dir: str) -> dict:
        from pathlib import Path
        
        data_path = Path(data_dir)
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        required_dirs = ['train', 'val']
        for dir_name in required_dirs:
            dir_path = data_path / dir_name
            if not dir_path.exists():
                validation_result['valid'] = False
                validation_result['errors'].append(f"Missing directory: {dir_name}")
            else:
                images_dir = dir_path / 'images'
                labels_dir = dir_path / 'labels'
                
                if not images_dir.exists():
                    validation_result['errors'].append(f"Missing images directory in {dir_name}")
                
                if not labels_dir.exists():
                    validation_result['warnings'].append(f"Missing labels directory in {dir_name}")
                
                if images_dir.exists():
                    image_count = len(list(images_dir.glob('*.*')))
                    if image_count == 0:
                        validation_result['warnings'].append(f"No images found in {dir_name}/images")
        
        return validation_result
    
    @staticmethod
    def validate_annotation_format(label_path: str, num_classes: int) -> bool:
        from pathlib import Path
        
        label_file = Path(label_path)
        if not label_file.exists():
            return True
        
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        logger.error(f"Invalid annotation format in {label_path}")
                        return False
                    
                    class_id = int(parts[0])
                    if class_id >= num_classes:
                        logger.error(f"Invalid class ID {class_id} in {label_path}")
                        return False
        except Exception as e:
            logger.error(f"Error reading {label_path}: {e}")
            return False
        
        return True


def setup_exception_handler():
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = handle_exception


setup_exception_handler()
