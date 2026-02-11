"""摄像头相关模式定义"""
from typing import Optional
from pydantic import BaseModel, Field


class CameraCreate(BaseModel):
    """创建摄像头请求"""
    id: int = Field(..., description="摄像头ID")
    name: str = Field(..., min_length=1, max_length=100, description="摄像头名称")
    location: str = Field(..., min_length=1, max_length=200, description="摄像头位置")
    source: str = Field(..., description="摄像头源（设备ID或URL）")
    resolution: str = Field(default="1920x1080", description="分辨率")
    fps: int = Field(default=30, ge=1, le=120, description="帧率")
    enabled: bool = Field(default=True, description="是否启用")


class CameraUpdate(BaseModel):
    """更新摄像头请求"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    location: Optional[str] = Field(None, min_length=1, max_length=200)
    source: Optional[str] = None
    resolution: Optional[str] = None
    fps: Optional[int] = Field(None, ge=1, le=120)
    enabled: Optional[bool] = None


class CameraResponse(BaseModel):
    """摄像头响应"""
    id: int
    name: str
    location: str
    source: str
    resolution: str
    fps: int
    status: str
    enabled: bool
    created_at: Optional[str]
    updated_at: Optional[str]


class CameraListResponse(BaseModel):
    """摄像头列表响应"""
    success: bool
    total: int
    cameras: list[CameraResponse]


class CameraActionResponse(BaseModel):
    """摄像头操作响应"""
    success: bool
    camera_id: int
    action: str
    message: str
