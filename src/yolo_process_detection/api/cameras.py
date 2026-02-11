"""摄像头API路由模块

提供摄像头管理的API端点。
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..database import get_db, CameraRepository
from ..database.models import Camera
from ..services.camera import get_camera_manager, CameraStatus
from ..schemas.camera import (
    CameraCreate,
    CameraUpdate,
    CameraResponse,
    CameraListResponse,
    CameraActionResponse
)


router = APIRouter(prefix="/cameras", tags=["摄像头"])


@router.get("", response_model=CameraListResponse)
async def get_cameras(
    status: str = None,
    db: Session = Depends(get_db)
):
    """获取所有摄像头"""
    repo = CameraRepository(db)
    
    if status == "active":
        cameras = repo.get_active()
    elif status == "running":
        cameras = repo.get_running()
    else:
        cameras = repo.get_all()
    
    return CameraListResponse(
        success=True,
        total=len(cameras),
        cameras=[
            CameraResponse(
                id=cam.id,
                name=cam.name,
                location=cam.location,
                source=cam.source,
                resolution=cam.resolution,
                fps=cam.fps,
                status=cam.status,
                enabled=cam.enabled,
                created_at=cam.created_at.isoformat() if cam.created_at else None,
                updated_at=cam.updated_at.isoformat() if cam.updated_at else None
            )
            for cam in cameras
        ]
    )


@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(
    camera_id: int,
    db: Session = Depends(get_db)
):
    """获取指定摄像头"""
    repo = CameraRepository(db)
    camera = repo.get_by_id(camera_id)
    
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"摄像头 {camera_id} 不存在"
        )
    
    return CameraResponse(
        id=camera.id,
        name=camera.name,
        location=camera.location,
        source=camera.source,
        resolution=camera.resolution,
        fps=camera.fps,
        status=camera.status,
        enabled=camera.enabled,
        created_at=camera.created_at.isoformat() if camera.created_at else None,
        updated_at=camera.updated_at.isoformat() if camera.updated_at else None
    )


@router.post("", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
async def create_camera(
    camera: CameraCreate,
    db: Session = Depends(get_db)
):
    """创建新摄像头"""
    repo = CameraRepository(db)
    
    existing = repo.get_by_id(camera.id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"摄像头 {camera.id} 已存在"
        )
    
    new_camera = repo.create(
        id=camera.id,
        name=camera.name,
        location=camera.location,
        source=camera.source,
        resolution=camera.resolution,
        fps=camera.fps,
        enabled=camera.enabled
    )
    
    return CameraResponse(
        id=new_camera.id,
        name=new_camera.name,
        location=new_camera.location,
        source=new_camera.source,
        resolution=new_camera.resolution,
        fps=new_camera.fps,
        status=new_camera.status,
        enabled=new_camera.enabled,
        created_at=new_camera.created_at.isoformat() if new_camera.created_at else None,
        updated_at=new_camera.updated_at.isoformat() if new_camera.updated_at else None
    )


@router.put("/{camera_id}", response_model=CameraResponse)
async def update_camera(
    camera_id: int,
    camera: CameraUpdate,
    db: Session = Depends(get_db)
):
    """更新摄像头"""
    repo = CameraRepository(db)
    
    existing = repo.get_by_id(camera_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"摄像头 {camera_id} 不存在"
        )
    
    update_data = camera.model_dump(exclude_unset=True)
    updated_camera = repo.update(camera_id, **update_data)
    
    return CameraResponse(
        id=updated_camera.id,
        name=updated_camera.name,
        location=updated_camera.location,
        source=updated_camera.source,
        resolution=updated_camera.resolution,
        fps=updated_camera.fps,
        status=updated_camera.status,
        enabled=updated_camera.enabled,
        created_at=updated_camera.created_at.isoformat() if updated_camera.created_at else None,
        updated_at=updated_camera.updated_at.isoformat() if updated_camera.updated_at else None
    )


@router.delete("/{camera_id}", response_model=dict)
async def delete_camera(
    camera_id: int,
    db: Session = Depends(get_db)
):
    """删除摄像头"""
    repo = CameraRepository(db)
    
    existing = repo.get_by_id(camera_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"摄像头 {camera_id} 不存在"
        )
    
    success = repo.delete(camera_id)
    
    return {
        "success": success,
        "message": f"摄像头 {camera_id} 已删除"
    }


@router.post("/{camera_id}/start", response_model=CameraActionResponse)
async def start_camera(
    camera_id: int,
    db: Session = Depends(get_db)
):
    """启动摄像头"""
    repo = CameraRepository(db)
    manager = get_camera_manager()
    
    camera = repo.get_by_id(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"摄像头 {camera_id} 不存在"
        )
    
    if not camera.enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"摄像头 {camera_id} 未启用"
        )
    
    success = manager.start_camera(camera_id)
    
    if success:
        repo.update_status(camera_id, CameraStatus.RUNNING)
    
    return CameraActionResponse(
        success=success,
        camera_id=camera_id,
        action="start",
        message=f"摄像头 {camera_id} {'已启动' if success else '启动失败'}"
    )


@router.post("/{camera_id}/stop", response_model=CameraActionResponse)
async def stop_camera(
    camera_id: int,
    db: Session = Depends(get_db)
):
    """停止摄像头"""
    repo = CameraRepository(db)
    manager = get_camera_manager()
    
    camera = repo.get_by_id(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"摄像头 {camera_id} 不存在"
        )
    
    success = manager.stop_camera(camera_id)
    
    if success:
        repo.update_status(camera_id, CameraStatus.STOPPED)
    
    return CameraActionResponse(
        success=success,
        camera_id=camera_id,
        action="stop",
        message=f"摄像头 {camera_id} {'已停止' if success else '停止失败'}"
    )


@router.post("/start-all", response_model=dict)
async def start_all_cameras(db: Session = Depends(get_db)):
    """启动所有摄像头"""
    manager = get_camera_manager()
    repo = CameraRepository(db)
    
    results = manager.start_all()
    
    for camera_id, success in results.items():
        if success:
            repo.update_status(camera_id, CameraStatus.RUNNING)
    
    return {
        "success": True,
        "results": results,
        "started": sum(1 for v in results.values() if v),
        "failed": sum(1 for v in results.values() if not v)
    }


@router.post("/stop-all", response_model=dict)
async def stop_all_cameras(db: Session = Depends(get_db)):
    """停止所有摄像头"""
    manager = get_camera_manager()
    repo = CameraRepository(db)
    
    results = manager.stop_all()
    
    for camera_id, success in results.items():
        if success:
            repo.update_status(camera_id, CameraStatus.STOPPED)
    
    return {
        "success": True,
        "results": results,
        "stopped": sum(1 for v in results.values() if v),
        "failed": sum(1 for v in results.values() if not v)
    }
