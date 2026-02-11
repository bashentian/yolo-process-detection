import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import functools
import traceback
import asyncio


def setup_logging(log_dir: Optional[Path] = None, log_level: int = logging.INFO) -> logging.Logger:
    """设置日志记录
    
    Args:
        log_dir: 日志文件目录，如果为None则只输出到控制台
        log_level: 日志级别
        
    Returns:
        配置好的logger实例
    """
    logger = logging.getLogger("yolo_process_detection")
    logger.setLevel(log_level)
    
    if logger.handlers:
        logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"app_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"日志文件已创建: {log_file}")
    
    return logger


def log_execution_time(logger: Optional[logging.Logger] = None):
    """装饰器：记录函数执行时间（支持同步和异步函数）

    Args:
        logger: 日志记录器，如果为None则使用默认logger
    """
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                nonlocal logger
                if logger is None:
                    logger = logging.getLogger("yolo_process_detection")

                start_time = datetime.now()
                try:
                    result = await func(*args, **kwargs)
                    end_time = datetime.now()
                    execution_time = (end_time - start_time).total_seconds()
                    logger.debug(f"函数 {func.__name__} 执行时间: {execution_time:.4f}秒")
                    return result
                except Exception as e:
                    end_time = datetime.now()
                    execution_time = (end_time - start_time).total_seconds()
                    logger.error(f"函数 {func.__name__} 执行失败 (耗时{execution_time:.4f}秒): {str(e)}")
                    logger.error(traceback.format_exc())
                    raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                nonlocal logger
                if logger is None:
                    logger = logging.getLogger("yolo_process_detection")

                start_time = datetime.now()
                try:
                    result = func(*args, **kwargs)
                    end_time = datetime.now()
                    execution_time = (end_time - start_time).total_seconds()
                    logger.debug(f"函数 {func.__name__} 执行时间: {execution_time:.4f}秒")
                    return result
                except Exception as e:
                    end_time = datetime.now()
                    execution_time = (end_time - start_time).total_seconds()
                    logger.error(f"函数 {func.__name__} 执行失败 (耗时{execution_time:.4f}秒): {str(e)}")
                    logger.error(traceback.format_exc())
                    raise
            return sync_wrapper
    return decorator


def handle_exceptions(logger: Optional[logging.Logger] = None, default_return=None):
    """装饰器：捕获并处理异常
    
    Args:
        logger: 日志记录器
        default_return: 异常时的默认返回值
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger("yolo_process_detection")
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"函数 {func.__name__} 发生异常: {str(e)}")
                logger.error(traceback.format_exc())
                return default_return
        return wrapper
    return decorator


class PerformanceMonitor:
    """性能监控类"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("yolo_process_detection")
        self.metrics = {}
        
    def start_timer(self, name: str):
        """开始计时"""
        self.metrics[name] = {'start': datetime.now(), 'end': None, 'duration': None}
        
    def end_timer(self, name: str) -> float:
        """结束计时并返回耗时"""
        if name in self.metrics:
            self.metrics[name]['end'] = datetime.now()
            duration = (self.metrics[name]['end'] - self.metrics[name]['start']).total_seconds()
            self.metrics[name]['duration'] = duration
            self.logger.debug(f"{name} 耗时: {duration:.4f}秒")
            return duration
        return 0.0
        
    def get_metrics(self) -> dict:
        """获取所有性能指标"""
        return {name: data['duration'] for name, data in self.metrics.items() if data['duration'] is not None}
        
    def print_summary(self):
        """打印性能摘要"""
        self.logger.info("=" * 50)
        self.logger.info("性能监控摘要")
        self.logger.info("=" * 50)
        for name, duration in self.get_metrics().items():
            self.logger.info(f"{name}: {duration:.4f}秒")
        self.logger.info("=" * 50)


def validate_image_path(image_path: Path) -> bool:
    """验证图像路径是否有效
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        是否有效
    """
    if not image_path.exists():
        return False
    if not image_path.is_file():
        return False
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    return image_path.suffix.lower() in valid_extensions


def validate_video_path(video_path: Path) -> bool:
    """验证视频路径是否有效
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        是否有效
    """
    if not video_path.exists():
        return False
    if not video_path.is_file():
        return False
    
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    return video_path.suffix.lower() in valid_extensions


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全除法，避免除以零错误
    
    Args:
        numerator: 分子
        denominator: 分母
        default: 除零时的默认值
        
    Returns:
        除法结果或默认值
    """
    if denominator == 0:
        return default
    return numerator / denominator


class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, total: int, description: str = "Processing", logger: Optional[logging.Logger] = None):
        self.total = total
        self.current = 0
        self.description = description
        self.logger = logger or logging.getLogger("yolo_process_detection")
        self.start_time = datetime.now()
        
    def update(self, increment: int = 1):
        """更新进度"""
        self.current += increment
        percentage = (self.current / self.total) * 100 if self.total > 0 else 0
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if self.current > 0:
            estimated_total = elapsed * self.total / self.current
            remaining = estimated_total - elapsed
            self.logger.info(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%) - 预计剩余: {remaining:.1f}秒")
        else:
            self.logger.info(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%)")
            
    def finish(self):
        """完成进度"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(f"{self.description} 完成! 总耗时: {elapsed:.2f}秒")
