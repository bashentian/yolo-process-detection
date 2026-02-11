"""错误处理模块

提供统一的错误处理和异常管理。
"""
import logging
from typing import Any, Dict, Optional, Tuple
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)


class BaseError(Exception):
    """基础错误类"""
    
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }


class ValidationError(BaseError):
    """验证错误"""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if details is None:
            details = {}
        if field:
            details["field"] = field
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="VALIDATION_ERROR",
            details=details
        )


class NotFoundError(BaseError):
    """资源未找到错误"""
    
    def __init__(
        self,
        resource: str,
        identifier: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        message = f"{resource} not found"
        if identifier:
            message = f"{resource} '{identifier}' not found"
        
        if details is None:
            details = {}
        details["resource"] = resource
        if identifier:
            details["identifier"] = identifier
        
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="NOT_FOUND",
            details=details
        )


class ConflictError(BaseError):
    """冲突错误"""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            error_code="CONFLICT",
            details=details
        )


class AuthenticationError(BaseError):
    """认证错误"""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="AUTHENTICATION_ERROR",
            details=details
        )


class AuthorizationError(BaseError):
    """授权错误"""
    
    def __init__(
        self,
        message: str = "Permission denied",
        required_permission: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if details is None:
            details = {}
        if required_permission:
            details["required_permission"] = required_permission
        
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="AUTHORIZATION_ERROR",
            details=details
        )


class RateLimitError(BaseError):
    """速率限制错误"""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if details is None:
            details = {}
        if retry_after is not None:
            details["retry_after"] = retry_after
        
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_EXCEEDED",
            details=details
        )


class ServiceUnavailableError(BaseError):
    """服务不可用错误"""
    
    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="SERVICE_UNAVAILABLE",
            details=details
        )


class CameraError(BaseError):
    """摄像头错误"""
    
    def __init__(
        self,
        message: str,
        camera_id: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if details is None:
            details = {}
        if camera_id is not None:
            details["camera_id"] = camera_id
        
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="CAMERA_ERROR",
            details=details
        )


class DetectionError(BaseError):
    """检测错误"""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="DETECTION_ERROR",
            details=details
        )


class DatabaseError(BaseError):
    """数据库错误"""
    
    def __init__(
        self,
        message: str = "Database operation failed",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="DATABASE_ERROR",
            details=details
        )


def handle_exception(exc: Exception) -> Tuple[JSONResponse, int]:
    """处理异常并返回响应"""
    
    if isinstance(exc, BaseError):
        logger.error(
            f"{exc.error_code}: {exc.message}",
            extra={"details": exc.details}
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict()
        ), exc.status_code
    
    elif isinstance(exc, HTTPException):
        logger.error(f"HTTP Exception: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP_ERROR",
                "message": exc.detail
            }
        ), exc.status_code
    
    else:
        logger.exception("Unhandled exception occurred")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred"
            }
        ), status.HTTP_500_INTERNAL_SERVER_ERROR


def create_error_response(
    error_code: str,
    message: str,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    details: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """创建错误响应"""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": error_code,
            "message": message,
            "details": details or {}
        }
    )


def log_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None
):
    """记录错误日志"""
    logger.error(
        f"Error: {type(error).__name__}: {str(error)}",
        extra={"context": context or {}},
        exc_info=not isinstance(error, BaseError)
    )
