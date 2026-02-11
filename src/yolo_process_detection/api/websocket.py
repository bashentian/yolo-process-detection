"""WebSocket路由模块

提供WebSocket连接端点。
"""
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from ..services.websocket import websocket_endpoint, get_connection_manager


router = APIRouter(prefix="/ws", tags=["WebSocket"])


@router.websocket("/connect")
async def websocket_connect(
    websocket: WebSocket,
    client_id: Optional[str] = Query(default=None),
    camera_id: Optional[int] = Query(default=None)
):
    """WebSocket连接端点"""
    if client_id is None:
        import uuid
        client_id = str(uuid.uuid4())
    
    await websocket_endpoint(websocket, client_id, camera_id)


@router.websocket("/camera/{camera_id}")
async def camera_websocket(
    websocket: WebSocket,
    camera_id: int,
    client_id: Optional[str] = Query(default=None)
):
    """摄像头专用WebSocket端点"""
    if client_id is None:
        import uuid
        client_id = str(uuid.uuid4())
    
    await websocket_endpoint(websocket, client_id, camera_id)


@router.get("/status")
async def websocket_status():
    """获取WebSocket连接状态"""
    manager = get_connection_manager()
    return {
        "success": True,
        "connection_count": manager.get_connection_count(),
        "status": "running"
    }
