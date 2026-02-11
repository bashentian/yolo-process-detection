"""主路由模块

整合所有API路由。
"""
from fastapi import APIRouter
from .cameras import router as cameras_router
from .detection import router as detection_router
from .websocket import router as websocket_router
from .stream import router as stream_router
from .users import router as users_router
from .system import router as system_router
from .actions import router as actions_router


router = APIRouter()

router.include_router(cameras_router)
router.include_router(detection_router)
router.include_router(websocket_router)
router.include_router(stream_router)
router.include_router(users_router)
router.include_router(system_router)
router.include_router(actions_router)
