"""异常处理模块

提供项目中使用的自定义异常类和错误处理逻辑。
"""

from typing import Any
from enum import Enum


class ErrorCode(str, Enum):
    """错误代码枚举"""
    DETECTION_ERROR = "DETECTION_ERROR"
    MODEL_LOAD_ERROR = "MODEL_LOAD_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    VIDEO_SOURCE_ERROR = "VIDEO_SOURCE_ERROR"
    TRACKING_ERROR = "TRACKING_ERROR"
    ANALYSIS_ERROR = "ANALYSIS_ERROR"
    API_ERROR = "API_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class BaseError(Exception):
    """基础异常类
    
    所有自定义异常的基类，提供统一的错误代码和消息格式。
    """
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.cause = cause
    
    def __str__(self) -> str:
        """返回格式化的错误消息"""
        parts = [f"[{self.code.value}] {self.message}"]
        if self.details:
            parts.append(f"Details: {self.details}")
        return " ".join(parts)
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式"""
        return {
            "error": self.code.value,
            "message": self.message,
            "details": self.details
        }


class DetectionError(BaseError):
    """检测相关错误"""
    
    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None
    ):
        super().__init__(
            message=message,
            code=ErrorCode.DETECTION_ERROR,
            details=details,
            cause=cause
        )


class ModelLoadError(BaseError):
    """模型加载错误"""
    
    def __init__(
        self,
        message: str,
        model_path: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None
    ):
        error_details = {"model_path": model_path}
        if details:
            error_details.update(details)
        
        super().__init__(
            message=message,
            code=ErrorCode.MODEL_LOAD_ERROR,
            details=error_details,
            cause=cause
        )


class ConfigurationError(BaseError):
    """配置相关错误"""
    
    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        config_value: Any | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None
    ):
        error_details = {
            "config_key": config_key,
            "config_value": config_value
        }
        if details:
            error_details.update(details)
        
        super().__init__(
            message=message,
            code=ErrorCode.CONFIGURATION_ERROR,
            details=error_details,
            cause=cause
        )


class ValidationError(BaseError):
    """数据验证错误"""
    
    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any | None = None,
        constraint: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None
    ):
        error_details = {
            "field": field,
            "value": value,
            "constraint": constraint
        }
        if details:
            error_details.update(details)
        
        super().__init__(
            message=message,
            code=ErrorCode.VALIDATION_ERROR,
            details=error_details,
            cause=cause
        )


class VideoSourceError(BaseError):
    """视频源相关错误"""
    
    def __init__(
        self,
        message: str,
        source: str | int | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None
    ):
        error_details = {"source": str(source) if source is not None else None}
        if details:
            error_details.update(details)
        
        super().__init__(
            message=message,
            code=ErrorCode.VIDEO_SOURCE_ERROR,
            details=error_details,
            cause=cause
        )


class TrackingError(BaseError):
    """跟踪相关错误"""
    
    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None
    ):
        super().__init__(
            message=message,
            code=ErrorCode.TRACKING_ERROR,
            details=details,
            cause=cause
        )


class AnalysisError(BaseError):
    """分析相关错误"""
    
    def __init__(
        self,
        message: str,
        analysis_type: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None
    ):
        error_details = {"analysis_type": analysis_type}
        if details:
            error_details.update(details)
        
        super().__init__(
            message=message,
            code=ErrorCode.ANALYSIS_ERROR,
            details=error_details,
            cause=cause
        )


class APIError(BaseError):
    """API相关错误"""
    
    def __init__(
        self,
        message: str,
        endpoint: str | None = None,
        method: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None
    ):
        error_details = {
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code
        }
        if details:
            error_details.update(details)
        
        super().__init__(
            message=message,
            code=ErrorCode.API_ERROR,
            details=error_details,
            cause=cause
        )


def handle_exception(error: Exception) -> tuple[dict[str, Any], int]:
    """异常处理函数
    
    将异常转换为标准化的错误响应格式。
    
    Args:
        error: 异常对象
        
    Returns:
        tuple: (错误字典, HTTP状态码)
    """
    if isinstance(error, BaseError):
        response = error.to_dict()
        status_code = _get_status_code(error.code)
    else:
        response = {
            "error": ErrorCode.INTERNAL_ERROR.value,
            "message": str(error),
            "details": {"type": type(error).__name__}
        }
        status_code = 500
    
    return response, status_code


def _get_status_code(code: ErrorCode) -> int:
    """根据错误代码获取HTTP状态码"""
    status_map: dict[ErrorCode, int] = {
        ErrorCode.VALIDATION_ERROR: 422,
        ErrorCode.CONFIGURATION_ERROR: 500,
        ErrorCode.MODEL_LOAD_ERROR: 500,
        ErrorCode.VIDEO_SOURCE_ERROR: 400,
        ErrorCode.TRACKING_ERROR: 500,
        ErrorCode.ANALYSIS_ERROR: 500,
        ErrorCode.DETECTION_ERROR: 500,
        ErrorCode.API_ERROR: 400,
        ErrorCode.INTERNAL_ERROR: 500
    }
    return status_map.get(code, 500)
