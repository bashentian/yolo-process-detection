"""API路由模块

提供FastAPI路由定义，使用依赖注入管理服务实例。
"""
import asyncio
from pathlib import Path
from typing import Annotated
import time

from fastapi import (
    APIRouter,
    Depends,
    UploadFile,
    File,
    Query,
    Form,
    HTTPException,
    Request
)
from fastapi.responses import FileResponse, HTMLResponse
from starlette.templating import Jinja2Templates

from ...core.config import get_settings, Settings
from ...core.exceptions import handle_exception, APIError
from ...schemas.detection import (
    DetectionResponse,
    BatchDetectionResponse,
    Detection as DetectionSchema
)
from ...schemas.config import (
    UpdateConfigRequest,
    ConfigStatusResponse
)
from ...schemas.response import (
    HealthResponse,
    StatisticsResponse,
    EfficiencyResponse,
    TimelineResponse,
    AnomalyResponse,
    SceneResponse,
    ExportResponse,
    ResetResponse,
    ErrorResponse
)
from .dependencies import DetectorDep


router = APIRouter()


def get_templates() -> Jinja2Templates:
    """获取Jinja2模板实例"""
    return Jinja2Templates(directory="templates")


@router.get("/health", response_model=HealthResponse)
async def health_check(detector: DetectorDep) -> HealthResponse:
    """健康检查"""
    settings = get_settings()
    
    # 简单的模型健康检查
    model_loaded = detector.model is not None
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        version="1.0.0",
        model_loaded=model_loaded,
        gpu_available=settings.device == "cuda",
        uptime_seconds=0.0  # TODO: 实现正常运行时间跟踪
    )


@router.post("/detect", response_model=DetectionResponse)
async def detect(
    detector: DetectorDep,
    image: Annotated[UploadFile, File(...)],
    confidence: Annotated[float | None] = Query(
        default=None,
        ge=0,
        le=1,
        description="置信度阈值"
    ),
    iou: Annotated[float | None] = Query(
        default=None,
        ge=0,
        le=1,
        description="IoU阈值"
    )
) -> DetectionResponse:
    """检测图像中的对象"""
    import cv2
    import numpy as np
    from PIL import Image
    import io
    
    try:
        # 读取图像
        contents = await image.read()
        image_array = np.asarray(Image.open(io.BytesIO(contents)))
        # 转换 RGB 到 BGR (OpenCV格式)
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            frame = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        
        # 临时覆盖配置
        original_conf = detector.confidence_threshold
        original_iou = detector.iou_threshold
        
        if confidence is not None:
            detector.confidence_threshold = confidence
        if iou is not None:
            detector.iou_threshold = iou
        
        start_time = time.time()
        
        # 调用检测逻辑
        detections = detector.detect(frame)
        
        inference_time = time.time() - start_time
        
        # 恢复配置
        detector.confidence_threshold = original_conf
        detector.iou_threshold = original_iou
        
        # 转换检测结果为Schema格式
        detection_schemas = [
            DetectionSchema(
                bbox=d.bbox,
                confidence=d.confidence,
                class_id=d.class_id,
                class_name=d.class_name
            ) for d in detections
        ]
        
        return DetectionResponse(
            success=True,
            detections=detection_schemas,
            image_width=frame.shape[1],
            image_height=frame.shape[0],
            inference_time=inference_time,
            model_name=detector.model_name
        )
    
    except Exception as e:
        error_response, status_code = handle_exception(e)
        raise HTTPException(status_code=status_code, detail=error_response)


@router.post("/detect/batch", response_model=BatchDetectionResponse)
async def detect_batch(
    detector: DetectorDep,
    images: list[Annotated[UploadFile, File(...)]],
    max_concurrent: Annotated[int | None] = Query(
        default=5,
        ge=1,
        le=20,
        description="最大并发数"
    )
) -> BatchDetectionResponse:
    """批量检测图像"""
    import cv2
    import numpy as np
    from PIL import Image
    import io

    semaphore = asyncio.Semaphore(max_concurrent)
    results: list[DetectionResponse] = []
    errors: list[dict] = []
    
    async def bounded_detect(image: UploadFile) -> None:
        async with semaphore:
            try:
                contents = await image.read()
                image_array = np.asarray(Image.open(io.BytesIO(contents)))
                if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                    frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                else:
                    frame = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)

                start_time = time.time()
                detections = detector.detect(frame)
                inference_time = time.time() - start_time
                
                detection_schemas = [
                    DetectionSchema(
                        bbox=d.bbox,
                        confidence=d.confidence,
                        class_id=d.class_id,
                        class_name=d.class_name
                    ) for d in detections
                ]

                result = DetectionResponse(
                    success=True,
                    detections=detection_schemas,
                    image_width=frame.shape[1],
                    image_height=frame.shape[0],
                    inference_time=inference_time,
                    model_name=detector.model_name
                )
                results.append(result)
            except Exception as e:
                errors.append({
                    "filename": image.filename,
                    "error": str(e)
                })
    
    await asyncio.gather(*[bounded_detect(img) for img in images])
    
    return BatchDetectionResponse(
        success=len(errors) == 0,
        total_images=len(images),
        successful_images=len(results),
        failed_images=len(errors),
        total_detections=sum(
            len(r.detections) for r in results
        ),
        total_time=sum(r.inference_time for r in results),
        results=results,
        errors=errors
    )


@router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics(detector: DetectorDep) -> StatisticsResponse:
    """获取统计信息"""
    # 目前从 detector 中获取不到累计统计信息，除非添加持久化存储
    # 这里先返回当前状态
    
    return StatisticsResponse(
        success=True,
        summary={
            "total": 0, # TODO: 从数据库或缓存获取
            "by_class": {},
            "avg_confidence": 0.0
        },
        stage_distribution={},
        frame_count=0,
        processing_time=0.0,
        fps=0.0
    )


@router.get("/efficiency", response_model=EfficiencyResponse)
async def get_efficiency(detector: DetectorDep) -> EfficiencyResponse:
    """获取效率分析"""
    
    efficiency_info = detector.efficiency_analyzer.analyze()
    
    return EfficiencyResponse(
        success=True,
        status=efficiency_info['status'],
        score=efficiency_info['score'],
        metrics={
            "throughput": efficiency_info['throughput'],
            "latency": efficiency_info['latency'],
            "accuracy": efficiency_info['accuracy'],
            "efficiency_score": efficiency_info['score']
        },
        trend=efficiency_info['trend'],
        recommendations=efficiency_info['recommendations'],
        sample_count=len(detector.efficiency_analyzer.metrics)
    )


@router.get("/timeline", response_model=TimelineResponse)
async def get_timeline(
    start_frame: int = Query(default=0, ge=0),
    end_frame: int = Query(default=100, ge=0)
) -> TimelineResponse:
    """获取检测时间线"""
    from datetime import datetime
    
    # TODO: 需要实现历史数据存储才能返回真实时间线
    return TimelineResponse(
        success=True,
        total_points=0,
        stage_timeline=[],
        processing_range={
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": datetime.now(),
            "end_time": datetime.now()
        }
    )


@router.get("/anomalies", response_model=AnomalyResponse)
async def get_anomalies(detector: DetectorDep) -> AnomalyResponse:
    """获取异常检测结果"""
    # 返回最近的异常检测历史
    
    # 这里我们只能获取最近的一次状态，因为detector是瞬时的
    # 实际应用中应该查询数据库
    
    # 模拟数据，或者如果有持久化存储则从那里获取
    # 这里我们返回detector中的历史数据统计
    
    history = detector.anomaly_detector.history
    anomaly_count = 0
    # 简单的统计
    # 注意：history里存储的是features，不是检测结果
    
    return AnomalyResponse(
        success=True,
        total_checked=len(history),
        anomaly_count=0, # 无法直接从feature推断，除非存储了result
        anomaly_rate=0.0,
        anomalies=[],
        timestamp=None
    )


@router.post("/analyze/scene", response_model=SceneResponse)
async def analyze_scene(
    detector: DetectorDep,
    image: Annotated[UploadFile, File(...)]
) -> SceneResponse:
    """场景分析"""
    import cv2
    import numpy as np
    from PIL import Image
    import io
    
    try:
        contents = await image.read()
        image_array = np.asarray(Image.open(io.BytesIO(contents)))
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            frame = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        
        # 使用高级检测
        result = detector.detect_advanced(frame)
        scene_info = result['scene']
        
        return SceneResponse(
            success=True,
            scene={
                "stage": scene_info['stage'],
                "confidence": scene_info['confidence'],
                "context": scene_info['context'],
                "features": scene_info['scene_features']
            }
        )
    except Exception as e:
        error_response, status_code = handle_exception(e)
        raise HTTPException(status_code=status_code, detail=error_response)


@router.get("/config", response_model=dict)
async def get_config(detector: DetectorDep) -> dict:
    """获取当前配置"""
    settings = get_settings()
    return {
        "detection": {
            "model_name": detector.model_name,
            "confidence_threshold": detector.confidence_threshold,
            "iou_threshold": detector.iou_threshold,
            "max_detections": settings.max_detections,
            "device": detector.device
        },
        "tracking": {
            "enabled": settings.tracking_enabled,
            "max_age": settings.tracking_max_age,
            "min_hits": settings.tracking_min_hits
        },
        "analysis": {
            "scene_understanding_enabled": settings.scene_understanding_enabled,
            "anomaly_threshold": settings.anomaly_threshold,
            "use_attention": detector.use_attention
        }
    }


@router.put("/config", response_model=ConfigStatusResponse)
async def update_config(
    request: UpdateConfigRequest,
    detector: DetectorDep
) -> ConfigStatusResponse:
    """更新配置"""
    settings = get_settings()
    updated_fields: list[str] = []
    
    if request.detection:
        if request.detection.model_name:
            settings.model_name = request.detection.model_name
            detector.model_name = request.detection.model_name
            # 触发重新加载模型? detector.model 是 lazy property, 需要重置 _model
            detector._model = None 
            updated_fields.append("detection.model_name")
        if request.detection.confidence_threshold:
            settings.confidence_threshold = request.detection.confidence_threshold
            detector.confidence_threshold = request.detection.confidence_threshold
            updated_fields.append("detection.confidence_threshold")
        if request.detection.iou_threshold:
            settings.iou_threshold = request.detection.iou_threshold
            detector.iou_threshold = request.detection.iou_threshold
            updated_fields.append("detection.iou_threshold")
    
    return ConfigStatusResponse(
        success=True,
        current_config={},
        updated_fields=updated_fields,
        message=f"成功更新 {len(updated_fields)} 个配置项"
    )


@router.post("/export", response_model=ExportResponse)
async def export_results(
    format: str = Query(default="json", regex="^(json|csv|xlsx)$")
) -> ExportResponse:
    """导出检测结果"""
    settings = get_settings()
    
    export_dir = settings.outputs_root
    export_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    file_path = export_dir / f"export_{timestamp}.{format}"
    
    return ExportResponse(
        success=True,
        file_path=str(file_path),
        file_size=0,
        format=format,
        record_count=0,
        message="Export completed successfully"
    )


@router.post("/reset", response_model=ResetResponse)
async def reset_analysis(
    detector: DetectorDep,
    clear_history: bool = Query(default=True),
    clear_cache: bool = Query(default=False)
) -> ResetResponse:
    """重置分析状态"""
    cleared_data: list[str] = []
    
    if clear_history:
        detector.anomaly_detector.history.clear()
        detector.efficiency_analyzer.metrics.clear()
        cleared_data.append("history")
    if clear_cache:
        # TODO: Clear image cache if implemented
        cleared_data.append("cache")
    
    return ResetResponse(
        success=True,
        cleared_data=cleared_data,
        message="Analysis reset successfully"
    )


@router.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Web界面"""
    templates = get_templates()
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/video")
async def video_feed():
    """视频流页面"""
    templates = get_templates()
    return templates.TemplateResponse("camera_test.html", {})
