"""性能优化模块

提供性能监控、缓存和优化工具。
"""
import time
import functools
import asyncio
from typing import Callable, Any, Dict, Optional
from collections import defaultdict
from contextlib import contextmanager
import threading


class PerformanceMonitor:
    """性能监控器
    
    跟踪函数执行时间、调用次数等性能指标。
    """
    
    def __init__(self):
        self._metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0,
            "errors": 0
        })
        self._lock = threading.Lock()
    
    def record(self, name: str, duration: float, success: bool = True):
        """记录性能指标"""
        with self._lock:
            metrics = self._metrics[name]
            metrics["count"] += 1
            metrics["total_time"] += duration
            metrics["min_time"] = min(metrics["min_time"], duration)
            metrics["max_time"] = max(metrics["max_time"], duration)
            if not success:
                metrics["errors"] += 1
    
    def get_metrics(self, name: Optional[str] = None) -> Dict[str, Any]:
        """获取性能指标"""
        with self._lock:
            if name:
                metrics = self._metrics.get(name, {})
                if metrics:
                    return {
                        "name": name,
                        **metrics,
                        "avg_time": metrics["total_time"] / metrics["count"] if metrics["count"] > 0 else 0
                    }
                return {}
            
            return {
                name: {
                    **metrics,
                    "avg_time": metrics["total_time"] / metrics["count"] if metrics["count"] > 0 else 0
                }
                for name, metrics in self._metrics.items()
            }
    
    def reset(self, name: Optional[str] = None):
        """重置性能指标"""
        with self._lock:
            if name:
                if name in self._metrics:
                    del self._metrics[name]
            else:
                self._metrics.clear()


_monitor = PerformanceMonitor()


def get_monitor() -> PerformanceMonitor:
    """获取性能监控器单例"""
    return _monitor


def monitor_performance(name: Optional[str] = None):
    """性能监控装饰器
    
    用法:
        @monitor_performance("function_name")
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        metric_name = name or f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    duration = time.perf_counter() - start_time
                    _monitor.record(metric_name, duration, True)
                    return result
                except Exception as e:
                    duration = time.perf_counter() - start_time
                    _monitor.record(metric_name, duration, False)
                    raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    duration = time.perf_counter() - start_time
                    _monitor.record(metric_name, duration, True)
                    return result
                except Exception as e:
                    duration = time.perf_counter() - start_time
                    _monitor.record(metric_name, duration, False)
                    raise
            return sync_wrapper
    
    return decorator


class SimpleCache:
    """简单缓存实现
    
    基于内存的缓存，支持TTL（过期时间）。
    """
    
    def __init__(self, default_ttl: int = 300):
        self._cache: Dict[str, tuple[Any, float]] = {}
        self._default_ttl = default_ttl
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            if key not in self._cache:
                return None
            
            value, expiry = self._cache[key]
            
            if time.time() > expiry:
                del self._cache[key]
                return None
            
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存值"""
        expiry = time.time() + (ttl if ttl is not None else self._default_ttl)
        with self._lock:
            self._cache[key] = (value, expiry)
    
    def delete(self, key: str):
        """删除缓存值"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
    
    def cleanup(self):
        """清理过期缓存"""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, (_, expiry) in self._cache.items()
                if current_time > expiry
            ]
            for key in expired_keys:
                del self._cache[key]


_cache = SimpleCache()


def get_cache() -> SimpleCache:
    """获取缓存单例"""
    return _cache


def cached(ttl: int = 300, key_prefix: str = ""):
    """缓存装饰器
    
    用法:
        @cached(ttl=60, key_prefix="user:")
        def get_user(user_id: int):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{key_prefix}{func.__name__}:{args}:{kwargs}"
            cached_value = _cache.get(cache_key)
            
            if cached_value is not None:
                return cached_value
            
            result = func(*args, **kwargs)
            _cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    
    return decorator


@contextmanager
def performance_context(name: str):
    """性能上下文管理器
    
    用法:
        with performance_context("operation_name"):
            # 执行操作
            pass
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        _monitor.record(name, duration, True)


class RateLimiter:
    """速率限制器
    
    限制函数调用的频率。
    """
    
    def __init__(self, max_calls: int, period: float):
        self._max_calls = max_calls
        self._period = period
        self._calls: Dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def allow(self, key: str) -> bool:
        """检查是否允许调用"""
        with self._lock:
            current_time = time.time()
            calls = self._calls[key]
            
            calls = [t for t in calls if current_time - t < self._period]
            self._calls[key] = calls
            
            if len(calls) >= self._max_calls:
                return False
            
            calls.append(current_time)
            return True
    
    def reset(self, key: str):
        """重置限制"""
        with self._lock:
            if key in self._calls:
                del self._calls[key]


def rate_limit(max_calls: int, period: float, key_func: Optional[Callable] = None):
    """速率限制装饰器
    
    用法:
        @rate_limit(max_calls=10, period=60)
        def my_function():
            pass
    """
    limiter = RateLimiter(max_calls, period)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = key_func(*args, **kwargs) if key_func else func.__name__
            
            if not limiter.allow(key):
                raise Exception(f"Rate limit exceeded for {key}")
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator
