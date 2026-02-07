"""性能优化模块

提供内存管理、批处理优化和缓存机制，用于提升YOLO检测系统的整体性能。
"""

import gc
import sys
import time
import psutil
import numpy as np
from typing import List, Callable, Optional, Iterator, TypeVar, Generic, Any
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
import threading
import queue
import logging

T = TypeVar('T')


@dataclass
class MemoryStats:
    """内存统计信息"""
    rss_mb: float  # 实际使用内存 (MB)
    vms_mb: float  # 虚拟内存 (MB)
    percent: float  # 内存使用百分比
    available_mb: float  # 可用内存 (MB)


@dataclass
class PerformanceMetrics:
    """性能指标"""
    processing_time: float
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    fps: float


class MemoryManager:
    """内存管理器
    
    监控和管理内存使用，提供内存优化策略。
    """
    
    def __init__(self, 
                 memory_threshold: float = 80.0,
                 gc_threshold: int = 3,
                 logger: Optional[logging.Logger] = None):
        """初始化内存管理器
        
        Args:
            memory_threshold: 内存使用阈值百分比
            gc_threshold: 触发GC的阈值（连续检测次数）
            logger: 日志记录器
        """
        self.memory_threshold = memory_threshold
        self.gc_threshold = gc_threshold
        self.logger = logger or logging.getLogger(__name__)
        self._gc_counter = 0
        self._process = psutil.Process()
        
    def get_memory_stats(self) -> MemoryStats:
        """获取当前内存统计信息"""
        memory_info = self._process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return MemoryStats(
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            percent=system_memory.percent,
            available_mb=system_memory.available / 1024 / 1024
        )
    
    def check_memory(self) -> bool:
        """检查内存状态
        
        Returns:
            True if memory is within acceptable limits
        """
        stats = self.get_memory_stats()
        
        if stats.percent > self.memory_threshold:
            self.logger.warning(
                f"内存使用过高: {stats.percent:.1f}% "
                f"(阈值: {self.memory_threshold}%)"
            )
            return False
        
        return True
    
    def optimize_memory(self, force: bool = False):
        """优化内存使用
        
        Args:
            force: 是否强制进行垃圾回收
        """
        self._gc_counter += 1
        
        if force or self._gc_counter >= self.gc_threshold:
            gc.collect()
            self._gc_counter = 0
            
            if self.logger:
                stats = self.get_memory_stats()
                self.logger.debug(
                    f"内存优化完成 - RSS: {stats.rss_mb:.1f}MB, "
                    f"使用率: {stats.percent:.1f}%"
                )
    
    def __enter__(self):
        """上下文管理器入口"""
        self._initial_stats = self.get_memory_stats()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.optimize_memory(force=True)
        final_stats = self.get_memory_stats()
        
        delta = final_stats.rss_mb - self._initial_stats.rss_mb
        if abs(delta) > 10:  # 如果内存变化超过10MB
            self.logger.info(f"内存变化: {delta:+.1f}MB")


class BatchProcessor:
    """批处理器
    
    提供高效的批处理功能，支持动态批大小和内存优化。
    """
    
    def __init__(self,
                 batch_size: int = 8,
                 max_batch_size: int = 32,
                 memory_manager: Optional[MemoryManager] = None,
                 logger: Optional[logging.Logger] = None):
        """初始化批处理器
        
        Args:
            batch_size: 默认批大小
            max_batch_size: 最大批大小
            memory_manager: 内存管理器实例
            logger: 日志记录器
        """
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.memory_manager = memory_manager or MemoryManager()
        self.logger = logger or logging.getLogger(__name__)
        self._dynamic_batch = True
        
    def process_batches(self,
                       items: List[T],
                       process_fn: Callable[[List[T]], Any]) -> Iterator[Any]:
        """分批处理项目
        
        Args:
            items: 待处理的项目列表
            process_fn: 处理函数，接收一个批次并返回结果
            
        Yields:
            每个批次的处理结果
        """
        current_batch_size = self.batch_size
        
        for i in range(0, len(items), current_batch_size):
            # 动态调整批大小
            if self._dynamic_batch:
                if not self.memory_manager.check_memory():
                    current_batch_size = max(1, current_batch_size // 2)
                    self.logger.info(f"降低批大小至: {current_batch_size}")
                elif current_batch_size < self.max_batch_size:
                    current_batch_size = min(
                        current_batch_size + 1,
                        self.max_batch_size
                    )
            
            batch = items[i:i + current_batch_size]
            
            try:
                start_time = time.time()
                result = process_fn(batch)
                processing_time = time.time() - start_time
                
                self.logger.debug(
                    f"批次处理完成 - 大小: {len(batch)}, "
                    f"耗时: {processing_time:.3f}s"
                )
                
                yield result
                
                # 定期优化内存
                self.memory_manager.optimize_memory()
                
            except Exception as e:
                self.logger.error(f"批次处理失败: {e}")
                raise
    
    def process_stream(self,
                      stream: Iterator[T],
                      process_fn: Callable[[List[T]], Any],
                      timeout: float = 1.0) -> Iterator[Any]:
        """流式批处理
        
        从流中收集项目并分批处理。
        
        Args:
            stream: 项目流
            process_fn: 处理函数
            timeout: 等待新项目的最长时间（秒）
            
        Yields:
            每个批次的处理结果
        """
        batch = []
        last_item_time = time.time()
        
        for item in stream:
            batch.append(item)
            last_item_time = time.time()
            
            # 检查是否达到批大小或超时
            if len(batch) >= self.batch_size:
                yield process_fn(batch)
                batch = []
                self.memory_manager.optimize_memory()
            
            # 检查超时
            if time.time() - last_item_time > timeout and batch:
                yield process_fn(batch)
                batch = []
        
        # 处理剩余项目
        if batch:
            yield process_fn(batch)


class ImageCache:
    """图像缓存管理器
    
    提供图像缓存功能，支持LRU缓存策略。
    """
    
    def __init__(self,
                 max_size: int = 100,
                 max_memory_mb: float = 500.0,
                 logger: Optional[logging.Logger] = None):
        """初始化图像缓存
        
        Args:
            max_size: 最大缓存条目数
            max_memory_mb: 最大内存使用（MB）
            logger: 日志记录器
        """
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.logger = logger or logging.getLogger(__name__)
        self._cache: dict = {}
        self._access_times: dict = {}
        self._memory_usage_mb = 0.0
        self._lock = threading.Lock()
        
    def _get_image_memory(self, image: np.ndarray) -> float:
        """计算图像内存占用（MB）"""
        return image.nbytes / 1024 / 1024
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """获取缓存的图像
        
        Args:
            key: 缓存键
            
        Returns:
            缓存的图像，如果不存在则返回None
        """
        with self._lock:
            if key in self._cache:
                self._access_times[key] = time.time()
                self.logger.debug(f"缓存命中: {key}")
                return self._cache[key].copy()
            return None
    
    def put(self, key: str, image: np.ndarray):
        """添加图像到缓存
        
        Args:
            key: 缓存键
            image: 要缓存的图像
        """
        image_memory = self._get_image_memory(image)
        
        with self._lock:
            # 检查是否会超出内存限制
            if self._memory_usage_mb + image_memory > self.max_memory_mb:
                self._evict_oldest()
            
            # 检查缓存大小限制
            if len(self._cache) >= self.max_size:
                self._evict_oldest()
            
            self._cache[key] = image.copy()
            self._access_times[key] = time.time()
            self._memory_usage_mb += image_memory
            
            self.logger.debug(
                f"缓存添加: {key}, "
                f"内存使用: {self._memory_usage_mb:.1f}MB"
            )
    
    def _evict_oldest(self):
        """移除最旧的缓存条目"""
        if not self._cache:
            return
        
        oldest_key = min(self._access_times, key=self._access_times.get)
        removed_image = self._cache.pop(oldest_key)
        self._access_times.pop(oldest_key)
        self._memory_usage_mb -= self._get_image_memory(removed_image)
        
        self.logger.debug(f"缓存移除: {oldest_key}")
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._memory_usage_mb = 0.0
            self.logger.info("缓存已清空")
    
    def get_stats(self) -> dict:
        """获取缓存统计信息"""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage_mb': self._memory_usage_mb,
                'max_memory_mb': self.max_memory_mb
            }


class ParallelProcessor:
    """并行处理器
    
    提供多线程并行处理功能。
    """
    
    def __init__(self,
                 num_workers: int = 4,
                 queue_size: int = 10,
                 logger: Optional[logging.Logger] = None):
        """初始化并行处理器
        
        Args:
            num_workers: 工作线程数
            queue_size: 任务队列大小
            logger: 日志记录器
        """
        self.num_workers = num_workers
        self.queue_size = queue_size
        self.logger = logger or logging.getLogger(__name__)
        self._task_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._result_queue: queue.Queue = queue.Queue()
        self._workers: List[threading.Thread] = []
        self._running = False
        
    def _worker_loop(self, worker_id: int):
        """工作线程循环"""
        self.logger.debug(f"工作线程 {worker_id} 启动")
        
        while self._running:
            try:
                task = self._task_queue.get(timeout=1.0)
                if task is None:  # 停止信号
                    break
                
                func, args, kwargs = task
                try:
                    result = func(*args, **kwargs)
                    self._result_queue.put(('success', result))
                except Exception as e:
                    self._result_queue.put(('error', e))
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"工作线程 {worker_id} 错误: {e}")
        
        self.logger.debug(f"工作线程 {worker_id} 停止")
    
    def start(self):
        """启动工作线程"""
        self._running = True
        self._workers = []
        
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self._workers.append(worker)
        
        self.logger.info(f"启动了 {self.num_workers} 个工作线程")
    
    def stop(self):
        """停止工作线程"""
        self._running = False
        
        # 发送停止信号
        for _ in self._workers:
            self._task_queue.put(None)
        
        # 等待线程结束
        for worker in self._workers:
            worker.join(timeout=5.0)
        
        self.logger.info("工作线程已停止")
    
    def submit(self, func: Callable, *args, **kwargs):
        """提交任务
        
        Args:
            func: 要执行的函数
            *args, **kwargs: 函数参数
        """
        self._task_queue.put((func, args, kwargs))
    
    def get_results(self, timeout: Optional[float] = None) -> Iterator[Any]:
        """获取结果
        
        Args:
            timeout: 等待超时时间
            
        Yields:
            任务结果
        """
        while True:
            try:
                status, result = self._result_queue.get(timeout=timeout)
                if status == 'error':
                    raise result
                yield result
            except queue.Empty:
                break
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()


class PerformanceMonitor:
    """性能监控器
    
    监控系统性能指标。
    """
    
    def __init__(self, 
                 log_interval: int = 100,
                 logger: Optional[logging.Logger] = None):
        """初始化性能监控器
        
        Args:
            log_interval: 记录间隔（处理次数）
            logger: 日志记录器
        """
        self.log_interval = log_interval
        self.logger = logger or logging.getLogger(__name__)
        self._counter = 0
        self._total_time = 0.0
        self._memory_manager = MemoryManager(logger=logger)
        
    def record(self, processing_time: float):
        """记录处理时间
        
        Args:
            processing_time: 处理耗时（秒）
        """
        self._counter += 1
        self._total_time += processing_time
        
        if self._counter % self.log_interval == 0:
            avg_time = self._total_time / self._counter
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0.0
            
            memory_stats = self._memory_manager.get_memory_stats()
            
            self.logger.info(
                f"性能统计 - 处理数: {self._counter}, "
                f"平均耗时: {avg_time*1000:.2f}ms, "
                f"平均FPS: {avg_fps:.2f}, "
                f"内存: {memory_stats.rss_mb:.1f}MB"
            )
            
            # 重置计数器
            self._counter = 0
            self._total_time = 0.0


def optimize_numpy_operations():
    """优化NumPy操作设置"""
    # 设置NumPy的线程数
    import os
    
    # 避免NumPy过度使用线程
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    # 设置NumPy打印选项
    np.set_printoptions(precision=4, suppress=True)


def get_optimal_batch_size(image_shape: tuple,
                          target_memory_mb: float = 1000.0,
                          dtype_size: int = 4) -> int:
    """计算最优批大小
    
    Args:
        image_shape: 图像形状 (H, W, C)
        target_memory_mb: 目标内存使用（MB）
        dtype_size: 数据类型大小（字节）
        
    Returns:
        最优批大小
    """
    # 计算单张图像内存占用
    image_memory = np.prod(image_shape) * dtype_size / 1024 / 1024
    
    # 考虑预处理和后处理的额外内存
    overhead_factor = 3.0
    
    # 计算批大小
    batch_size = int(target_memory_mb / (image_memory * overhead_factor))
    
    # 限制范围
    return max(1, min(batch_size, 64))


# 全局性能优化设置
optimize_numpy_operations()