"""检测API路由模块

提供检测相关的API端点。
"""
import asyncio
import cv2
import numpy as np
from io import BytesIO
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from sqlalchemy.orm import Session
from PIL import Image

from ..database import get_db, DetectionRepository
from ..database.models import Detection
from ..models.detector import YOLODetector
from ..models.tracker import ObjectTracker
from ..core.config import get_settings
from ..schemas.detection import (
    DetectionResponse,
    BatchDetectionResponse,
    DetectionHistoryResponse,
    DetectionStatisticsResponse
)


router = APIRouter(prefix="/detection", tags=["检测"])

_detector: Optional[YOLODetector] = None
_tracker: Optional[ObjectTracker] = None


def get_detector() -> YOLODetector:
    """获取检测器单例"""
    global _detector
    if _detector is None:
        settings = get_settings()
        _detector = YOLODetector(
            model_name=settings.model_name,
            confidence_threshold=settings.confidence_threshold,
            iou_threshold=settings.iou_threshold,
            device=settings.device
        )
    return _detector


def get_tracker() -> ObjectTracker:
    """获取跟踪器单例"""
    global _tracker
    if _tracker is None:
        settings = get_settings()
        _tracker = ObjectTracker(
            max_age=settings.max_age,
            min_hits=settings.min_hits
        )
    return _tracker


@router.post("/detect", response_model=DetectionResponse)
async def detect_image(
    image: UploadFile = File(...),
    confidence: Optional[float] = Query(None, ge=0, le=1),
    iou: Optional[float] = Query(None, ge=0, le=1),
    db: Session = Depends(get_db)
):
    """检测图像中的对象"""
    try:
        contents = await image.read()
        image_array = np.asarray(Image.open(BytesIO(contents)))
        frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        detector = get_detector()
        
        original_conf = detector.confidence_threshold
        original_iou = detector.iou_threshold
        
        if confidence is not None:
            detector.confidence_threshold = confidence
        if iou is not None:
            detector.iou_threshold = iou
        
        import time
        start_time = time.time()
        detections = detector.detect(frame)
        inference_time = time.time() - start_time
        
        detector.confidence_threshold = original_conf
        detector.iou_threshold = original_iou
        
        detection_results = [
            {
                "bbox": {
                    "x_min": float(det.bbox[0]),
                    "y_min": float(det.bbox[1]),
                    "x_max": float(det.bbox[2]),
                    "y_max": float(det.bbox[3])
                },
                "confidence": det.confidence,
                "class_id": det.class_id,
                "class_name": det.class_name,
                "track_id": None
            }
            for det in detections
        ]
        
        repo = DetectionRepository(db)
        for det in detections:
            repo.create(
                camera_id=0,
                frame_number=0,
                x_min=float(det.bbox[0]),
                y_min=float(det.bbox[1]),
                x_max=float(det.bbox[2]),
                y_max=float(det.bbox[3]),
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name
            )
        
        return DetectionResponse(
            success=True,
            detections=detection_results,
            image_width=frame.shape[1],
            image_height=frame.shape[0],
            inference_time=inference_time,
            model_name=detector.model_name
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"检测失败: {str(e)}"
        )


@router.post("/detect/batch", response_model=BatchDetectionResponse)
async def detect_batch(
    images: List[UploadFile] = File(...),
    max_concurrent: int = Query(default=5, ge=1, le=20),
    db: Session = Depends(get_db)
):
    """批量检测图像"""
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []
    errors = []
    
    async def process_image(image: UploadFile):
        async with semaphore:
            try:
                contents = await image.read()
                image_array = np.asarray(Image.open(BytesIO(contents)))
                frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                
                detector = get_detector()
                
                import time
                start_time = time.time()
                detections = detector.detect(frame)
                inference_time = time.time() - start_time
                
                detection_results = [
                    {
                        "bbox": {
                            "x_min": float(det.bbox[0]),
                            "y_min": float(det.bbox[1]),
                            "x_max": float(det.bbox[2]),
                            "y_max": float(det.bbox[3])
                        },
                        "confidence": det.confidence,
                        "class_id": det.class_id,
                        "class_name": det.class_name,
                        "track_id": None
                    }
                    for det in detections
                ]
                
                return DetectionResponse(
                    success=True,
                    detections=detection_results,
                    image_width=frame.shape[1],
                    image_height=frame.shape[0],
                    inference_time=inference_time,
                    model_name=detector.model_name
                )
            
            except Exception as e:
                errors.append({
                    "filename": image.filename,
                    "error": str(e)
                })
                return None
    
    tasks = [process_image(img) for img in images]
    results = await asyncio.gather(*tasks)
    results = [r for r in results if r is not None]
    
    return BatchDetectionResponse(
        success=len(errors) == 0,
        total_images=len(images),
        successful_images=len(results),
        failed_images=len(errors),
        total_detections=sum(len(r.detections) for r in results),
        total_time=sum(r.inference_time for r in results),
        results=results,
        errors=errors
    )


@router.get("/history", response_model=DetectionHistoryResponse)
async def get_detection_history(
    camera_id: Optional[int] = Query(None),
    class_name: Optional[str] = Query(None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db)
):
    """获取检测历史记录"""
    repo = DetectionRepository(db)
    
    if camera_id is not None:
        detections = repo.get_by_camera(camera_id, limit, offset)
    elif class_name is not None:
        detections = repo.get_by_class(class_name, limit)
    else:
        detections = repo.get_by_camera(0, limit, offset)
    
    detection_results = [
        {
            "id": det.id,
            "camera_id": det.camera_id,
            "timestamp": det.timestamp.isoformat() if det.timestamp else None,
            "bbox": {
                "x_min": det.x_min,
                "y_min": det.y_min,
                "x_max": det.x_max,
                "y_max": det.y_max
            },
            "confidence": det.confidence,
            "class_id": det.class_id,
            "class_name": det.class_name,
            "track_id": det.track_id,
            "image_path": det.image_path,
            "thumbnail_path": det.thumbnail_path
        }
        for det in detections
    ]
    
    return DetectionHistoryResponse(
        success=True,
        total=len(detection_results),
        limit=limit,
        offset=offset,
        detections=detection_results
    )


@router.get("/statistics", response_model=DetectionStatisticsResponse)
async def get_detection_statistics(
    camera_id: Optional[int] = Query(None),
    days: int = Query(default=7, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """获取检测统计信息"""
    from datetime import datetime, timedelta
    
    repo = DetectionRepository(db)
    
    start_time = datetime.utcnow() - timedelta(days=days)
    end_time = datetime.utcnow()
    
    stats = repo.get_statistics(camera_id, start_time, end_time)
    
    return DetectionStatisticsResponse(
        success=True,
        summary=stats,
        time_range={
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "days": days
        }
    )
