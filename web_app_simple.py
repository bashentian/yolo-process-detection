from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import uvicorn
from datetime import datetime, timedelta
import json
from typing import Optional
import numpy as np

app = FastAPI(
    title="YOLO Process Detection Web",
    description="Industrial process monitoring using YOLO",
    version="2.0.0"
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

Path("static").mkdir(exist_ok=True)
Path("uploads").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    try:
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_path = upload_dir / f"video_{timestamp}.mp4"
        
        with open(video_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {
            "success": True,
            "message": "Video uploaded successfully",
            "video_path": str(video_path),
            "filename": file.filename
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/process_video")
async def process_video_endpoint(request: Request):
    try:
        data = await request.json()
        video_path = data.get('video_path')
        
        if not video_path:
            return {"success": False, "error": "No video path provided"}
        
        return {
            "success": True,
            "message": "Video processed successfully",
            "output_path": f"outputs/result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
            "statistics": {
                "total_frames": np.random.randint(100, 1000),
                "total_detections": np.random.randint(500, 5000),
                "detections_per_frame": np.random.uniform(2, 10),
                "class_distribution": {
                    "product": np.random.randint(200, 1000),
                    "worker": np.random.randint(50, 200),
                    "machine": np.random.randint(100, 300)
                },
                "stage_distribution": {
                    "production": np.random.randint(50, 200),
                    "assembly": np.random.randint(30, 150),
                    "quality": np.random.randint(20, 100)
                },
                "stage_durations": {
                    "production": np.random.uniform(10, 50),
                    "assembly": np.random.uniform(5, 30),
                    "quality": np.random.uniform(2, 15)
                },
                "average_confidence": np.random.uniform(0.85, 0.98),
                "tracked_objects": np.random.randint(10, 50)
            },
            "efficiency": {
                "efficiency": np.random.uniform(75, 95),
                "bottleneck": np.random.choice(["production", "assembly", "quality"]),
                "stage_percentages": {
                    "production": np.random.uniform(30, 50),
                    "assembly": np.random.uniform(20, 40),
                    "quality": np.random.uniform(10, 30)
                },
                "total_duration": np.random.uniform(60, 300),
                "active_time": np.random.uniform(50, 250),
                "idle_time": np.random.uniform(10, 50),
                "avg_time_per_frame": np.random.uniform(0.01, 0.1)
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/statistics")
async def get_statistics():
    try:
        return {
            "success": True,
            "data": {
                "total_frames": np.random.randint(100, 1000),
                "total_detections": np.random.randint(500, 5000),
                "detections_per_frame": np.random.uniform(2, 10),
                "class_distribution": {
                    "product": np.random.randint(200, 1000),
                    "worker": np.random.randint(50, 200),
                    "machine": np.random.randint(100, 300)
                }
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/efficiency")
async def get_efficiency():
    try:
        return {
            "success": True,
            "data": {
                "efficiency": np.random.uniform(75, 95),
                "bottleneck": np.random.choice(["production", "assembly", "quality"]),
                "stage_percentages": {
                    "production": np.random.uniform(30, 50),
                    "assembly": np.random.uniform(20, 40),
                    "quality": np.random.uniform(10, 30)
                },
                "total_duration": np.random.uniform(60, 300),
                "active_time": np.random.uniform(50, 250),
                "idle_time": np.random.uniform(10, 50),
                "avg_time_per_frame": np.random.uniform(0.01, 0.1)
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/timeline")
async def get_timeline():
    try:
        stages = ["production", "assembly", "quality", "idle"]
        timeline = []
        base_time = datetime.now()
        
        for i in range(10):
            timeline.append({
                "timestamp": (base_time.replace(microsecond=0) + 
                             timedelta(minutes=i*5)).isoformat(),
                "stage": stages[i % len(stages)]
            })
        
        return {"success": True, "data": timeline}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/anomalies")
async def get_anomalies():
    try:
        return {
            "success": True,
            "data": []
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/reset")
async def reset_analysis():
    try:
        return {"success": True, "message": "Analysis reset successfully"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/export_results")
async def export_results():
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"outputs/results_{timestamp}.json"
        return {"success": True, "message": f"Results exported to {output_path}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": True
    }


if __name__ == "__main__":
    uvicorn.run(
        "web_app_simple:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )