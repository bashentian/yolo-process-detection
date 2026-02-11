"""认证模块

提供JWT认证和用户管理功能。
"""
from jose import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, TYPE_CHECKING
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..core.config import get_settings

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from ..database.models import User


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


class AuthError(Exception):
    """认证错误"""
    def __init__(self, message: str, status_code: int = status.HTTP_401_UNAUTHORIZED):
        self.message = message
        self.status_code = status_code


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """获取密码哈希"""
    return pwd_context.hash(password)


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """创建访问令牌"""
    settings = get_settings()
    
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm="HS256"
    )
    
    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """解码访问令牌"""
    settings = get_settings()
    
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=["HS256"]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthError("令牌已过期")
    except jwt.InvalidTokenError:
        raise AuthError("无效的令牌")


def _get_db():
    """延迟导入数据库依赖"""
    from ..database import get_db
    return get_db


def _get_user_repository(db):
    """延迟导入用户仓储"""
    from ..database import UserRepository
    return UserRepository(db)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db = Depends(_get_db)
):
    """获取当前用户"""
    token = credentials.credentials
    
    try:
        payload = decode_access_token(token)
        username: str = payload.get("sub")
        
        if username is None:
            raise AuthError("无效的认证凭据")
    
    except AuthError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail=e.message,
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    repo = _get_user_repository(db)
    user = repo.get_by_username(username)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="用户已被禁用"
        )
    
    return user


async def get_current_active_user(
    current_user = Depends(get_current_user)
):
    """获取当前活跃用户"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="用户已被禁用"
        )
    return current_user


class RoleChecker:
    """角色检查器"""
    
    def __init__(self, allowed_roles: list[str]):
        self.allowed_roles = allowed_roles
    
    def __call__(self, current_user = Depends(get_current_user)):
        if current_user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="权限不足"
            )
        return current_user


def require_admin() -> RoleChecker:
    """要求管理员权限"""
    return RoleChecker(allowed_roles=["admin"])


def require_user() -> RoleChecker:
    """要求用户权限"""
    return RoleChecker(allowed_roles=["admin", "user"])
