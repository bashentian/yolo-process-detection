"""统一的异常处理模块

提供异常捕获、错误日志记录和错误恢复功能，支持同步和异步操作。
"""

import asyncio
import functools
import logging
import traceback
from typing import Optional, Callable, Any, Union
from enum import Enum


class ErrorSeverity(Enum):
    """错误严重程度"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AppException(Exception):
    """应用基础异常类"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[dict] = None
    ):
        self.message = message
        self.severity = severity
        self.context = context or {}
        super().__init__(message)


class ModelLoadError(AppException):
    """模型加载错误"""
    pass


class VideoProcessingError(AppException):
    """视频处理错误"""
    pass


class DetectionError(AppException):
    """检测错误"""
    pass


class ConfigError(AppException):
    """配置错误"""
    pass


class ErrorHandler:
    """错误处理器

    提供统一的异常处理和日志记录功能。
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts: dict = {}

    def handle(
        self,
        error: Exception,
        context: str = "",
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        raise_exception: bool = False,
        default_return: Any = None
    ) -> Any:
        """处理异常

        Args:
            error: 异常对象
            context: 上下文信息
            severity: 错误严重程度
            raise_exception: 是否重新抛出异常
            default_return: 默认返回值

        Returns:
            默认返回值或抛出异常
        """
        # 统计错误
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # 构建错误消息
        error_msg = f"{context}: {error}" if context else str(error)
        if hasattr(error, "context"):
            error_msg += f" | 上下文: {error.context}"

        # 记录日志
        log_func = {
            ErrorSeverity.DEBUG: self.logger.debug,
            ErrorSeverity.INFO: self.logger.info,
            ErrorSeverity.WARNING: self.logger.warning,
            ErrorSeverity.ERROR: self.logger.error,
            ErrorSeverity.CRITICAL: self.logger.critical
        }.get(severity, self.logger.error)

        if severity in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]:
            log_func(error_msg, exc_info=True)
        else:
            log_func(error_msg)

        if raise_exception:
            raise error

        return default_return

    def get_error_summary(self) -> dict:
        """获取错误统计摘要

        Returns:
            错误统计字典
        """
        return {
            "total_errors": sum(self.error_counts.values()),
            "by_type": self.error_counts.copy()
        }

    def clear_error_counts(self):
        """清空错误计数"""
        self.error_counts.clear()
        self.logger.info("错误计数已清空")


def handle_exceptions(
    context: str = "",
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    default_return: Any = None,
    logger: Optional[logging.Logger] = None,
    handler: Optional[ErrorHandler] = None
):
    """异常处理装饰器（支持同步和异步函数）

    Args:
        context: 上下文信息
        severity: 错误严重程度
        default_return: 默认返回值
        logger: 日志记录器
        handler: 错误处理器实例
    """
    error_handler = handler or ErrorHandler(logger)

    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    return error_handler.handle(
                        e,
                        context or f"{func.__name__}",
                        severity,
                        raise_exception=False,
                        default_return=default_return
                    )
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    return error_handler.handle(
                        e,
                        context or f"{func.__name__}",
                        severity,
                        raise_exception=False,
                        default_return=default_return
                    )
            return sync_wrapper

    return decorator


def validate_file_path(
    file_path: str,
    must_exist: bool = True,
    extensions: Optional[list] = None,
    logger: Optional[logging.Logger] = None
) -> bool:
    """验证文件路径

    Args:
        file_path: 文件路径
        must_exist: 是否必须存在
        extensions: 允许的扩展名列表
        logger: 日志记录器

    Returns:
        是否有效
    """
    from pathlib import Path

    log = logger or logging.getLogger(__name__)
    path = Path(file_path)

    if must_exist and not path.exists():
        log.error(f"文件不存在: {file_path}")
        return False

    if extensions:
        if path.suffix.lower() not in extensions:
            log.error(f"不支持的文件扩展名: {path.suffix}")
            return False

    return True


def validate_image_path(
    image_path: str,
    logger: Optional[logging.Logger] = None
) -> bool:
    """验证图像文件路径

    Args:
        image_path: 图像路径
        logger: 日志记录器

    Returns:
        是否有效
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
    return validate_file_path(image_path, must_exist=True, extensions=valid_extensions, logger=logger)


def validate_video_path(
    video_path: str,
    logger: Optional[logging.Logger] = None
) -> bool:
    """验证视频文件路径

    Args:
        video_path: 视频路径
        logger: 日志记录器

    Returns:
        是否有效
    """
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    return validate_file_path(video_path, must_exist=True, extensions=valid_extensions, logger=logger)


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
    logger: Optional[logging.Logger] = None
) -> float:
    """安全除法，避免除以零错误

    Args:
        numerator: 分子
        denominator: 分母
        default: 除零时的默认值
        logger: 日志记录器

    Returns:
        除法结果或默认值
    """
    log = logger or logging.getLogger(__name__)

    if denominator == 0:
        log.warning(f"除零错误: {numerator}/{denominator}, 返回默认值 {default}")
        return default

    return numerator / denominator


# 全局错误处理器实例
default_error_handler = ErrorHandler()


def setup_global_exception_handler(
    logger: Optional[logging.Logger] = None
):
    """设置全局异常处理器

    Args:
        logger: 日志记录器
    """
    log = logger or logging.getLogger(__name__)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # 允许 Ctrl+C 退出
            log.info("程序被用户中断")
            return

        log.critical("未捕获的异常", exc_info=(exc_type, exc_value, exc_traceback))

    import sys
    sys.excepthook = handle_exception


# 便捷装饰器
def catch_errors(
    context: str = "",
    default_return: Any = None,
    logger: Optional[logging.Logger] = None
):
    """捕获错误的便捷装饰器

    Args:
        context: 上下文信息
        default_return: 默认返回值
        logger: 日志记录器
    """
    return handle_exceptions(
        context=context,
        severity=ErrorSeverity.ERROR,
        default_return=default_return,
        logger=logger
    )


def catch_and_log(
    context: str = "",
    default_return: Any = None,
    logger: Optional[logging.Logger] = None
):
    """捕获并记录错误的便捷装饰器（仅记录，不抛出）

    Args:
        context: 上下文信息
        default_return: 默认返回值
        logger: 日志记录器
    """
    return handle_exceptions(
        context=context,
        severity=ErrorSeverity.WARNING,
        default_return=default_return,
        logger=logger
    )
