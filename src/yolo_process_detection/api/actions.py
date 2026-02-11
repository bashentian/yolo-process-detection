"""动作识别API路由模块

提供连贯动作识别的API端点。
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..database import get_db, ActionSegmentRepository, ScrewDetectionRepository
from ..models.action_recognizer import SequenceValidator, WorkflowGraph, ActionPredictor
from ..schemas.detection import DetectionHistoryItem, DetectionHistoryResponse
from ..core.auth import require_user
from ..core.errors import ValidationError


router = APIRouter(prefix="/actions", tags=["动作识别"])


class SequenceRequest:
    """序列验证请求"""
    expected_sequence: List[str]
    actual_sequence: List[str]


@router.post("/validate", response_model=DetectionHistoryResponse)
async def validate_sequence(
    request: SequenceRequest,
    db: Session = Depends(get_db),
    current_user = Depends(require_user())
):
    """验证操作序列"""
    validator = SequenceValidator(request.expected_sequence)
    result = validator.validate_sequence(request.actual_sequence)
    
    return DetectionHistoryResponse(
        success=result.is_valid,
        total=len(request.actual_sequence),
        detections=[
            DetectionHistoryItem(
                id=i,
                camera_id=0,
                timestamp="",
                class_name=action.value,
                confidence=1.0,
                bbox=[]
            )
            for i, action in enumerate(request.actual_sequence)
        ]
    )


@router.post("/predict", response_model=dict)
async def predict_next_actions(
    camera_id: int,
    history_length: int = 10,
    db: Session = Depends(get_db),
    current_user = Depends(require_user())
):
    """预测接下来的动作"""
    predictor = ActionPredictor(lookahead=5)
    
    # 获取历史数据
    repo = ActionSegmentRepository(db)
    recent_segments = repo.get_recent_segments(camera_id, limit=history_length)
    
    # 提取动作序列
    action_sequence = [seg.action_type for seg in recent_segments]
    predictor.update_history(action_sequence[-1])
    
    # 预测
    predictions = predictor.predict_next_actions()
    
    return {
        "success": True,
        "predictions": [action.value for action in predictions],
        "confidence": 0.7
    }


@router.get("/history/{camera_id}", response_model=DetectionHistoryResponse)
async def get_action_history(
    camera_id: int,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user = Depends(require_user())
):
    """获取动作历史"""
    repo = ActionSegmentRepository(db)
    segments = repo.get_recent_segments(camera_id, limit=limit)
    
    return DetectionHistoryResponse(
        success=True,
        total=len(segments),
        detections=[
            DetectionHistoryItem(
                id=seg.id,
                camera_id=camera_id,
                timestamp=seg.start_time,
                class_name=seg.action_type.value,
                confidence=seg.confidence,
                bbox=[]
            )
            for seg in segments
        ]
    )
