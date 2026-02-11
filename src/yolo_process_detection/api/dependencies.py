from functools import lru_cache
from typing import Annotated
from fastapi import Depends

from ..models.detector import YOLODetector


@lru_cache
def get_detector() -> YOLODetector:
    """获取检测器实例（单例）"""
    return YOLODetector()


# 依赖类型别名，方便在路由中使用
DetectorDep = Annotated[YOLODetector, Depends(get_detector)]
