from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import cv2
import numpy as np
from pathlib import Path
import tempfile
import json
from datetime import datetime
import io
import base64
from PIL import Image
import asyncio
import uvicorn
import logging

from config import ProcessDetectionConfig
from detector import ProcessDetector
from video_processor import VideoProcessor
from analyzer import ProcessAnalyzer


app = FastAPI(
    title="YOLO Process Detection API",
    description="Industrial-grade process detection system with YOLO",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = ProcessDetectionConfig()
detector = ProcessDetector(config)
processor = VideoProcessor(config)
analyzer = ProcessAnalyzer(config)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectionResult(BaseModel):
    bbox: List[float]
    confidence: float
    class_id: int
    class_name: str


class ImageDetectionResponse(BaseModel):
    success: bool
    image_info: Dict[str, Any]
    detections: List[DetectionResult]
    process_stage: str
    inference_time: float
    timestamp: str


class BatchDetectionResponse(BaseModel):
    success: bool
    total_images: int
    results: List[ImageDetectionResponse]
    summary: Dict[str, Any]


class ProcessStatistics(BaseModel):
    total_detections: int
    avg_confidence: float
    class_distribution: Dict[str, int]
    stage_distribution: Dict[str, int]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str


class ModelInfo(BaseModel):
    model_name: str
    classes: Dict[int, str]
    stages: List[str]
    input_size: List[int]
    confidence_threshold: float


@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "service": "YOLO Process Detection API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=detector.model is not None,
        device=config.DEVICE,
        timestamp=datetime.now().isoformat()
    )


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    return ModelInfo(
        model_name=config.MODEL_NAME,
        classes=config.CLASS_NAMES,
        stages=config.PROCESS_STAGES,
        input_size=list(config.FRAME_RESIZE),
        confidence_threshold=config.CONFIDENCE_THRESHOLD
    )


@app.post("/api/detect/single", response_model=ImageDetectionResponse)
async def detect_single(
    file: UploadFile = File(..., description="Image file for detection"),
    draw_boxes: bool = Form(False, description="Draw bounding boxes on result")
):
    try:
        start_time = cv2.getTickCount()
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        detections, processed_frame = detector.detect_frame(image)
        process_stage = detector.analyze_process_stage(detections)
        
        end_time = cv2.getTickCount()
        inference_time = (end_time - start_time) / cv2.getTickFrequency() * 1000
        
        detection_results = [
            DetectionResult(
                bbox=detection.bbox,
                confidence=detection.confidence,
                class_id=detection.class_id,
                class_name=detection.class_name
            )
            for detection in detections
        ]
        
        response_data = ImageDetectionResponse(
            success=True,
            image_info={
                "filename": file.filename,
                "width": image.shape[1],
                "height": image.shape[0],
                "channels": image.shape[2]
            },
            detections=detection_results,
            process_stage=process_stage,
            inference_time=round(inference_time, 2),
            timestamp=datetime.now().isoformat()
        )
        
        if draw_boxes:
            annotated_frame = detector.draw_detections(image, detections, process_stage)
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            img_str = base64.b64encode(buffer).decode('utf-8')
            response_data.image_info["annotated_image"] = img_str
        
        logger.info(f"Detected {len(detections)} objects in {file.filename}")
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error in detect_single: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/api/detect/batch", response_model=BatchDetectionResponse)
async def detect_batch(
    files: List[UploadFile] = File(..., description="List of image files (max 10)"),
    draw_boxes: bool = Form(False, description="Draw bounding boxes on results")
):
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images allowed")
    
    try:
        results = []
        total_detections = 0
        class_counts = {}
        stage_counts = {}
        total_time = 0
        
        for file in files:
            try:
                contents = await file.read()
                nparr = np.frombuffer(contents, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    results.append({
                        "filename": file.filename,
                        "error": "Invalid image file",
                        "success": False
                    })
                    continue
                
                start_time = cv2.getTickCount()
                detections, _ = detector.detect_frame(image)
                process_stage = detector.analyze_process_stage(detections)
                end_time = cv2.getTickCount()
                inference_time = (end_time - start_time) / cv2.getTickFrequency() * 1000
                
                total_detections += len(detections)
                total_time += inference_time
                
                for det in detections:
                    class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
                
                stage_counts[process_stage] = stage_counts.get(process_stage, 0) + 1
                
                detection_results = [
                    DetectionResult(
                        bbox=detection.bbox,
                        confidence=detection.confidence,
                        class_id=detection.class_id,
                        class_name=detection.class_name
                    )
                    for detection in detections
                ]
                
                results.append(ImageDetectionResponse(
                    success=True,
                    image_info={
                        "filename": file.filename,
                        "width": image.shape[1],
                        "height": image.shape[0],
                        "channels": image.shape[2]
                    },
                    detections=detection_results,
                    process_stage=process_stage,
                    inference_time=round(inference_time, 2),
                    timestamp=datetime.now().isoformat()
                ))
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                results.append({
                    "filename": file.filename,
                    "error": str(e),
                    "success": False
                })
        
        response = BatchDetectionResponse(
            success=True,
            total_images=len(files),
            results=results,
            summary={
                "total_detections": total_detections,
                "avg_inference_time": round(total_time / len(files), 2) if files else 0,
                "class_distribution": class_counts,
                "stage_distribution": stage_counts
            }
        )
        
        logger.info(f"Processed {len(files)} images, detected {total_detections} objects")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in detect_batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch detection failed: {str(e)}")


@app.post("/api/detect/video")
async def detect_video(
    file: UploadFile = File(..., description="Video file for detection"),
    save_video: bool = Form(True, description="Save processed video"),
    fps: int = Form(30, description="Output FPS")
):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name
        
        logger.info(f"Processing video: {file.filename}")
        
        output_path = None
        if save_video:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"outputs/video_detection_{timestamp}.mp4"
        
        def process_callback(frame, detections, stage):
            pass
        
        processor.process_video(temp_file_path, output_path, process_callback)
        
        Path(temp_file_path).unlink()
        
        stats = analyzer.calculate_statistics()
        
        result = {
            "success": True,
            "video_info": {
                "filename": file.filename,
                "output_path": output_path,
                "save_video": save_video,
                "fps": fps
            },
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
        
        if output_path and Path(output_path).exists():
            return FileResponse(
                output_path,
                media_type="video/mp4",
                filename=f"processed_{file.filename}"
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in detect_video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video detection failed: {str(e)}")


@app.get("/api/statistics", response_model=ProcessStatistics)
async def get_statistics():
    try:
        stats = analyzer.calculate_statistics()
        
        return ProcessStatistics(
            total_detections=stats.get('total_detections', 0),
            avg_confidence=stats.get('average_confidence', 0.0),
            class_distribution=stats.get('class_distribution', {}),
            stage_distribution=stats.get('stage_distribution', {})
        )
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@app.get("/api/efficiency")
async def get_efficiency():
    try:
        efficiency = analyzer.analyze_process_efficiency()
        
        return {
            "success": True,
            "efficiency": efficiency,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting efficiency: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get efficiency: {str(e)}")


@app.get("/api/timeline")
async def get_timeline():
    try:
        timeline = analyzer.get_stage_timeline()
        
        return {
            "success": True,
            "timeline": timeline,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting timeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get timeline: {str(e)}")


@app.get("/api/anomalies")
async def get_anomalies():
    try:
        anomalies = analyzer.detect_anomalies()
        
        return {
            "success": True,
            "anomalies": anomalies,
            "count": len(anomalies),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting anomalies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get anomalies: {str(e)}")


@app.post("/api/export/results")
async def export_results(format: str = Form("json")):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            output_path = f"outputs/export_{timestamp}.json"
            analyzer.export_results(output_path)
            
            return FileResponse(
                output_path,
                media_type="application/json",
                filename=f"results_{timestamp}.json"
            )
        
        elif format == "csv":
            output_path = f"outputs/export_{timestamp}.csv"
            
            import pandas as pd
            df = pd.DataFrame(analyzer.detection_history)
            df.to_csv(output_path, index=False)
            
            return FileResponse(
                output_path,
                media_type="text/csv",
                filename=f"results_{timestamp}.csv"
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
            
    except Exception as e:
        logger.error(f"Error exporting results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@app.post("/api/reset")
async def reset_analysis():
    try:
        analyzer.reset()
        
        return {
            "success": True,
            "message": "Analysis data has been reset",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error resetting analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/api/config/update")
async def update_config(
    confidence_threshold: Optional[float] = Form(None, ge=0.0, le=1.0),
    iou_threshold: Optional[float] = Form(None, ge=0.0, le=1.0),
    max_detections: Optional[int] = Form(None, ge=1, le=1000)
):
    try:
        if confidence_threshold is not None:
            config.CONFIDENCE_THRESHOLD = confidence_threshold
            detector.model.conf = confidence_threshold
        
        if iou_threshold is not None:
            config.IOU_THRESHOLD = iou_threshold
            detector.model.iou = iou_threshold
        
        if max_detections is not None:
            config.MAX_DETECTIONS = max_detections
        
        return {
            "success": True,
            "config": {
                "confidence_threshold": config.CONFIDENCE_THRESHOLD,
                "iou_threshold": config.IOU_THRESHOLD,
                "max_detections": config.MAX_DETECTIONS
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error updating config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Config update failed: {str(e)}")


@app.get("/api/config")
async def get_config():
    return {
        "success": True,
        "config": {
            "model_name": config.MODEL_NAME,
            "confidence_threshold": config.CONFIDENCE_THRESHOLD,
            "iou_threshold": config.IOU_THRESHOLD,
            "max_detections": config.MAX_DETECTIONS,
            "device": config.DEVICE,
            "frame_resize": config.FRAME_RESIZE,
            "classes": config.CLASS_NAMES,
            "stages": config.PROCESS_STAGES
        },
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )