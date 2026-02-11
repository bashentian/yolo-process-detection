"""系统API路由模块

提供系统监控和管理的API端点。
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ..core.performance import get_monitor, get_cache
from ..core.auth import require_admin
from ..database import get_db, CameraRepository, DetectionRepository


router = APIRouter(prefix="/system", tags=["系统"])


class SystemStatusResponse(BaseModel):
    """系统状态响应"""
    success: bool
    status: str
    cameras: dict
    detections: dict
    cache: dict


class PerformanceMetricsResponse(BaseModel):
    """性能指标响应"""
    success: bool
    metrics: dict


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(db = Depends(get_db)):
    """获取系统状态"""
    camera_repo = CameraRepository(db)
    detection_repo = DetectionRepository(db)
    
    cameras = camera_repo.get_all()
    running_cameras = camera_repo.get_running()
    
    stats = detection_repo.get_statistics()
    
    return SystemStatusResponse({
        "success": True,
        "status": "running",
        "cameras": {
            "total": len(cameras),
            "running": len(running_cameras),
            "stopped": len(cameras) - len(running_cameras)
        },
        "detections": {
            "total": stats.get("total", 0),
            "by_class": stats.get("by_class", {})
        },
        "cache": {
            "enabled": True,
            "size": len(get_cache()._cache)
        }
    })


@router.get("/performance", response_model=PerformanceMetricsResponse)
async def get_performance_metrics(
    name: Optional[str] = None,
    current_user = Depends(require_admin())
):
    """获取性能指标（仅管理员）"""
    monitor = get_monitor()
    metrics = monitor.get_metrics(name)
    
    return PerformanceMetricsResponse({
        "success": True,
        "metrics": metrics
    })


@router.post("/performance/reset")
async def reset_performance_metrics(
    name: Optional[str] = None,
    current_user = Depends(require_admin())
):
    """重置性能指标（仅管理员）"""
    monitor = get_monitor()
    monitor.reset(name)
    
    return {
        "success": True,
        "message": f"性能指标已重置: {name if name else '全部'}"
    }


@router.post("/cache/clear")
async def clear_cache(
    current_user = Depends(require_admin())
):
    """清空缓存（仅管理员）"""
    cache = get_cache()
    cache.clear()
    
    return {
        "success": True,
        "message": "缓存已清空"
    }


@router.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": "2026-02-10"
    }
