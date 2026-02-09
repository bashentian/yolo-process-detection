"""统一的性能监控模块

提供性能监控、内存管理和性能指标收集功能，支持同步和异步操作。
"""

import asyncio
import time
import psutil
import functools
import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import threading
from collections import deque


@dataclass
class PerformanceStats:
    """性能统计信息"""
    timestamp: datetime
    duration: float
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    cpu_percent: float
    fps: Optional[float] = None


class PerformanceMonitor:
    """性能监控器

    支持同步和异步函数的性能监控，提供内存管理和性能指标收集。
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        max_history: int = 1000,
        memory_threshold_mb: float = 1000.0
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.max_history = max_history
        self.memory_threshold_mb = memory_threshold_mb
        self.history: deque = deque(maxlen=max_history)
        self._lock = threading.Lock()
        self._enabled = True

    @contextmanager
    def sync_measure(
        self,
        name: str,
        enable_memory: bool = True,
        log_threshold: float = 0.1
    ) -> None:
        """同步性能测量上下文管理器

        Args:
            name: 测量名称
            enable_memory: 是否启用内存监控
            log_threshold: 超过此时间才记录日志
        """
        start_time = time.time()
        memory_before = self._get_memory_mb() if enable_memory else None
        cpu_percent = self._get_cpu_percent()

        try:
            yield
        finally:
            duration = time.time() - start_time
            memory_after = self._get_memory_mb() if enable_memory else None

            stats = PerformanceStats(
                timestamp=datetime.now(),
                duration=duration,
                memory_before_mb=memory_before or 0.0,
                memory_after_mb=memory_after or 0.0,
                memory_delta_mb=(memory_after - memory_before) if memory_before and memory_after else 0.0,
                cpu_percent=cpu_percent,
                fps=1.0 / duration if duration > 0 else None
            )

            # 记录到历史
            with self._lock:
                self.history.append(stats)

            # 检查内存阈值
            if memory_after and memory_after > self.memory_threshold_mb:
                self.logger.warning(
                    f"内存使用过高: {memory_after:.1f}MB (阈值: {self.memory_threshold_mb}MB)"
                )

            # 记录长时间运行的函数
            if duration > log_threshold:
                self.logger.info(
                    f"{name} 耗时: {duration*1000:.2f}ms, "
                    f"内存: {memory_after:.1f}MB" if memory_after else f"内存: 未知"
                )

    @asynccontextmanager
    async def async_measure(
        self,
        name: str,
        enable_memory: bool = True,
        log_threshold: float = 0.1
    ) -> None:
        """异步性能测量上下文管理器

        Args:
            name: 测量名称
            enable_memory: 是否启用内存监控
            log_threshold: 超过此时间才记录日志
        """
        start_time = time.time()
        memory_before = self._get_memory_mb() if enable_memory else None
        cpu_percent = self._get_cpu_percent()

        try:
            yield
        finally:
            duration = time.time() - start_time
            memory_after = self._get_memory_mb() if enable_memory else None

            stats = PerformanceStats(
                timestamp=datetime.now(),
                duration=duration,
                memory_before_mb=memory_before or 0.0,
                memory_after_mb=memory_after or 0.0,
                memory_delta_mb=(memory_after - memory_before) if memory_before and memory_after else 0.0,
                cpu_percent=cpu_percent,
                fps=1.0 / duration if duration > 0 else None
            )

            with self._lock:
                self.history.append(stats)

            if memory_after and memory_after > self.memory_threshold_mb:
                self.logger.warning(
                    f"内存使用过高: {memory_after:.1f}MB (阈值: {self.memory_threshold_mb}MB)"
                )

            if duration > log_threshold:
                self.logger.info(
                    f"{name} 耗时: {duration*1000:.2f}ms, "
                    f"内存: {memory_after:.1f}MB" if memory_after else f"内存: 未知"
                )

    def _get_memory_mb(self) -> float:
        """获取当前内存使用量（MB）"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def _get_cpu_percent(self) -> float:
        """获取当前CPU使用率"""
        return psutil.cpu_percent()

    def get_recent_stats(self, count: int = 10) -> List[PerformanceStats]:
        """获取最近的性能统计

        Args:
            count: 要获取的数量

        Returns:
            性能统计列表
        """
        with self._lock:
            return list(self.history)[-count:]

    def get_average_duration(self, seconds: float = 60.0) -> float:
        """获取最近N秒的平均处理时间

        Args:
            seconds: 时间范围（秒）

        Returns:
            平均处理时间（秒）
        """
        cutoff_time = datetime.now().timestamp() - seconds
        recent_stats = [
            stat for stat in self.history
            if stat.timestamp.timestamp() > cutoff_time
        ]

        if not recent_stats:
            return 0.0

        return sum(stat.duration for stat in recent_stats) / len(recent_stats)

    def get_stats_summary(self) -> Dict[str, Any]:
        """获取性能统计摘要

        Returns:
            包含摘要信息的字典
        """
        with self._lock:
            if not self.history:
                return {"count": 0, "avg_duration": 0.0}

            durations = [stat.duration for stat in self.history]
            memory_deltas = [stat.memory_delta_mb for stat in self.history]

            return {
                "count": len(self.history),
                "avg_duration": sum(durations) / len(durations),
                "max_duration": max(durations),
                "min_duration": min(durations),
                "avg_memory_delta_mb": sum(memory_deltas) / len(memory_deltas),
                "current_memory_mb": self._get_memory_mb(),
                "current_cpu_percent": self._get_cpu_percent()
            }

    def print_stats_summary(self):
        """打印性能统计摘要"""
        summary = self.get_stats_summary()
        self.logger.info("性能统计摘要:")
        self.logger.info(f"  记录数: {summary['count']}")
        self.logger.info(f"  平均处理时间: {summary['avg_duration']*1000:.2f}ms")
        self.logger.info(f"  最大处理时间: {summary['max_duration']*1000:.2f}ms")
        self.logger.info(f"  最小处理时间: {summary['min_duration']*1000:.2f}ms")
        self.logger.info(f"  平均内存变化: {summary['avg_memory_delta_mb']:+.1f}MB")
        self.logger.info(f"  当前内存使用: {summary['current_memory_mb']:.1f}MB")
        self.logger.info(f"  当前CPU使用率: {summary['current_cpu_percent']:.1f}%")

    def clear_history(self):
        """清空历史记录"""
        with self._lock:
            self.history.clear()
            self.logger.info("性能监控历史已清空")

    def enable(self):
        """启用性能监控"""
        self._enabled = True
        self.logger.info("性能监控已启用")

    def disable(self):
        """禁用性能监控"""
        self._enabled = False
        self.logger.info("性能监控已禁用")


# 全局性能监控实例
default_performance_monitor = PerformanceMonitor()


# 便捷装饰器
def measure_performance(
    name: Optional[str] = None,
    enable_memory: bool = True,
    log_threshold: float = 0.1,
    monitor: Optional[PerformanceMonitor] = None
):
    """性能监控装饰器（自动检测同步/异步）

    Args:
        name: 测量名称，默认使用函数名
        enable_memory: 是否启用内存监控
        log_threshold: 超过此时间才记录日志
        monitor: 性能监控实例，默认使用全局实例
    """
    def decorator(func):
        func_name = name or f"{func.__module__}.{func.__name__}"
        monitor_instance = monitor or default_performance_monitor

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with monitor_instance.async_measure(
                    func_name, enable_memory, log_threshold
                ):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with monitor_instance.sync_measure(
                    func_name, enable_memory, log_threshold
                ):
                    return func(*args, **kwargs)
            return sync_wrapper

    return decorator


# 便捷上下文管理器
def measure_sync(
    name: str,
    enable_memory: bool = True,
    log_threshold: float = 0.1,
    monitor: Optional[PerformanceMonitor] = None
):
    """同步性能测量上下文管理器"""
    return (monitor or default_performance_monitor).sync_measure(
        name, enable_memory, log_threshold
    )


async def measure_async(
    name: str,
    enable_memory: bool = True,
    log_threshold: float = 0.1,
    monitor: Optional[PerformanceMonitor] = None
):
    """异步性能测量上下文管理器"""
    await (monitor or default_performance_monitor).async_measure(
        name, enable_memory, log_threshold
    )
