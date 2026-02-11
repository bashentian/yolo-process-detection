"""核心模块"""
from .config import Settings, get_settings
from .exceptions import (
    DetectionError,
    ModelLoadError,
    ConfigurationError,
    ValidationError,
    handle_exception
)
from .auth import (
    verify_password,
    get_password_hash,
    create_access_token,
    decode_access_token,
    get_current_user,
    get_current_active_user,
    require_admin,
    require_user
)
from .performance import (
    PerformanceMonitor,
    get_monitor,
    monitor_performance,
    SimpleCache,
    get_cache,
    cached,
    performance_context,
    RateLimiter,
    rate_limit
)
from .errors import (
    BaseError,
    ValidationError as APIValidationError,
    NotFoundError,
    ConflictError,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    ServiceUnavailableError,
    CameraError,
    DetectionError as APIDetectionError,
    DatabaseError,
    handle_exception as handle_api_exception,
    create_error_response,
    log_error
)

__all__ = [
    "Settings",
    "get_settings",
    "DetectionError",
    "ModelLoadError",
    "ConfigurationError",
    "ValidationError",
    "handle_exception",
    "verify_password",
    "get_password_hash",
    "create_access_token",
    "decode_access_token",
    "get_current_user",
    "get_current_active_user",
    "require_admin",
    "require_user",
    "PerformanceMonitor",
    "get_monitor",
    "monitor_performance",
    "SimpleCache",
    "get_cache",
    "cached",
    "performance_context",
    "RateLimiter",
    "rate_limit",
    "BaseError",
    "NotFoundError",
    "ConflictError",
    "AuthenticationError",
    "AuthorizationError",
    "RateLimitError",
    "ServiceUnavailableError",
    "CameraError",
    "DatabaseError",
    "handle_api_exception",
    "create_error_response",
    "log_error"
]
