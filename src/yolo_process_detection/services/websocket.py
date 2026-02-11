"""WebSocket管理模块

提供WebSocket连接管理和实时数据推送功能。
"""
import asyncio
import json
from typing import Dict, Set, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect


class MessageType(str, Enum):
    """消息类型"""
    DETECTION = "detection"
    CAMERA_STATUS = "camera_status"
    STATISTICS = "statistics"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"


@dataclass
class WebSocketMessage:
    """WebSocket消息"""
    type: MessageType
    data: Any
    timestamp: str
    camera_id: Optional[int] = None
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class ConnectionManager:
    """WebSocket连接管理器
    
    管理所有WebSocket连接，支持广播和定向发送。
    """
    
    def __init__(self):
        self._connections: Dict[str, WebSocket] = {}
        self._camera_subscriptions: Dict[int, Set[str]] = {}
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """连接WebSocket"""
        await websocket.accept()
        
        async with self._lock:
            self._connections[client_id] = websocket
        
        await self.send_message(
            client_id,
            WebSocketMessage(
                type=MessageType.PONG,
                data={"status": "connected"},
                timestamp=datetime.now().isoformat()
            )
        )
    
    async def disconnect(self, client_id: str):
        """断开WebSocket连接"""
        async with self._lock:
            if client_id in self._connections:
                del self._connections[client_id]
            
            for camera_id in list(self._camera_subscriptions.keys()):
                if client_id in self._camera_subscriptions[camera_id]:
                    self._camera_subscriptions[camera_id].remove(client_id)
                    if not self._camera_subscriptions[camera_id]:
                        del self._camera_subscriptions[camera_id]
    
    async def subscribe_camera(self, client_id: str, camera_id: int):
        """订阅摄像头"""
        async with self._lock:
            if camera_id not in self._camera_subscriptions:
                self._camera_subscriptions[camera_id] = set()
            self._camera_subscriptions[camera_id].add(client_id)
    
    async def unsubscribe_camera(self, client_id: str, camera_id: int):
        """取消订阅摄像头"""
        async with self._lock:
            if camera_id in self._camera_subscriptions:
                self._camera_subscriptions[camera_id].discard(client_id)
                if not self._camera_subscriptions[camera_id]:
                    del self._camera_subscriptions[camera_id]
    
    async def send_message(
        self,
        client_id: str,
        message: WebSocketMessage
    ) -> bool:
        """发送消息给指定客户端"""
        websocket = self._connections.get(client_id)
        if not websocket:
            return False
        
        try:
            await websocket.send_text(message.to_json())
            return True
        except Exception as e:
            print(f"发送消息失败: {e}")
            await self.disconnect(client_id)
            return False
    
    async def broadcast(self, message: WebSocketMessage):
        """广播消息给所有客户端"""
        async with self._lock:
            disconnected_clients = []
            
            for client_id, websocket in self._connections.items():
                try:
                    await websocket.send_text(message.to_json())
                except Exception as e:
                    print(f"广播消息失败: {e}")
                    disconnected_clients.append(client_id)
            
            for client_id in disconnected_clients:
                await self.disconnect(client_id)
    
    async def broadcast_to_camera(
        self,
        camera_id: int,
        message: WebSocketMessage
    ):
        """广播消息给订阅指定摄像头的客户端"""
        async with self._lock:
            if camera_id not in self._camera_subscriptions:
                return
            
            disconnected_clients = []
            
            for client_id in self._camera_subscriptions[camera_id]:
                websocket = self._connections.get(client_id)
                if not websocket:
                    continue
                
                try:
                    await websocket.send_text(message.to_json())
                except Exception as e:
                    print(f"广播消息失败: {e}")
                    disconnected_clients.append(client_id)
            
            for client_id in disconnected_clients:
                await self.disconnect(client_id)
    
    async def send_detection(
        self,
        camera_id: int,
        detection_data: dict
    ):
        """发送检测结果"""
        message = WebSocketMessage(
            type=MessageType.DETECTION,
            data=detection_data,
            timestamp=datetime.now().isoformat(),
            camera_id=camera_id
        )
        await self.broadcast_to_camera(camera_id, message)
    
    async def send_camera_status(
        self,
        camera_id: int,
        status: str,
        extra_data: Optional[dict] = None
    ):
        """发送摄像头状态"""
        data = {
            "camera_id": camera_id,
            "status": status
        }
        if extra_data:
            data.update(extra_data)
        
        message = WebSocketMessage(
            type=MessageType.CAMERA_STATUS,
            data=data,
            timestamp=datetime.now().isoformat(),
            camera_id=camera_id
        )
        await self.broadcast(message)
    
    async def send_statistics(self, statistics: dict):
        """发送统计信息"""
        message = WebSocketMessage(
            type=MessageType.STATISTICS,
            data=statistics,
            timestamp=datetime.now().isoformat()
        )
        await self.broadcast(message)
    
    async def send_error(
        self,
        client_id: Optional[str],
        error: str,
        error_code: Optional[str] = None
    ):
        """发送错误消息"""
        data = {
            "error": error
        }
        if error_code:
            data["error_code"] = error_code
        
        message = WebSocketMessage(
            type=MessageType.ERROR,
            data=data,
            timestamp=datetime.now().isoformat()
        )
        
        if client_id:
            await self.send_message(client_id, message)
        else:
            await self.broadcast(message)
    
    async def handle_ping(self, client_id: str):
        """处理ping消息"""
        message = WebSocketMessage(
            type=MessageType.PONG,
            data={"status": "alive"},
            timestamp=datetime.now().isoformat()
        )
        await self.send_message(client_id, message)
    
    def get_connection_count(self) -> int:
        """获取连接数"""
        return len(self._connections)
    
    def get_camera_subscriber_count(self, camera_id: int) -> int:
        """获取摄像头订阅者数量"""
        return len(self._camera_subscriptions.get(camera_id, set()))


_connection_manager: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """获取连接管理器单例"""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager


async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str,
    camera_id: Optional[int] = None
):
    """WebSocket端点
    
    处理WebSocket连接和消息。
    """
    manager = get_connection_manager()
    
    try:
        await manager.connect(websocket, client_id)
        
        if camera_id is not None:
            await manager.subscribe_camera(client_id, camera_id)
        
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            message_type = message_data.get("type")
            
            if message_type == "ping":
                await manager.handle_ping(client_id)
            elif message_type == "subscribe":
                cam_id = message_data.get("camera_id")
                if cam_id is not None:
                    await manager.subscribe_camera(client_id, cam_id)
            elif message_type == "unsubscribe":
                cam_id = message_data.get("camera_id")
                if cam_id is not None:
                    await manager.unsubscribe_camera(client_id, cam_id)
    
    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        print(f"WebSocket错误: {e}")
        await manager.disconnect(client_id)
