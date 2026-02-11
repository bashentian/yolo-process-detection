"""视频流路由模块

提供视频流的API端点。
"""
from fastapi import APIRouter, Response, HTTPException, status, Query
from fastapi.responses import StreamingResponse

from ..services.stream import get_streamer, cleanup_streamers
from ..services.camera import get_camera_manager


router = APIRouter(prefix="/stream", tags=["视频流"])


@router.get("/{camera_id}")
async def video_stream(camera_id: int):
    """获取摄像头视频流"""
    manager = get_camera_manager()
    camera = manager.get_camera(camera_id)
    
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"摄像头 {camera_id} 不存在"
        )
    
    if camera.status != "running":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"摄像头 {camera_id} 未运行"
        )
    
    streamer = get_streamer(camera_id)
    await streamer.start()
    
    return StreamingResponse(
        streamer.generate_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/mjpeg/{camera_id}")
async def mjpeg_stream(camera_id: int):
    """获取MJPEG视频流"""
    manager = get_camera_manager()
    camera = manager.get_camera(camera_id)
    
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"摄像头 {camera_id} 不存在"
        )
    
    if camera.status != "running":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"摄像头 {camera_id} 未运行"
        )
    
    streamer = get_streamer(camera_id)
    await streamer.start()
    
    return StreamingResponse(
        streamer.generate_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.post("/cleanup")
async def cleanup_streams():
    """清理所有视频流"""
    cleanup_streamers()
    return {
        "success": True,
        "message": "所有视频流已清理"
    }
