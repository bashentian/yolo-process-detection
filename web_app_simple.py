"""
简化版Web应用 - 已弃用
请使用 web_interface.py 或 web_app.py
"""

import warnings
warnings.warn(
    "web_app_simple.py 已弃用。请使用 web_interface.py 或 web_app.py。",
    DeprecationWarning,
    stacklevel=2
)

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import uvicorn
from datetime import datetime
import os

app = FastAPI(
    title="YOLO Process Detection Web (Deprecated)",
    description="Industrial process monitoring using YOLO - PLEASE USE web_interface.py INSTEAD",
    version="2.0.0-deprecated"
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

Path("static").mkdir(exist_ok=True)
Path("uploads").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "deprecated": True,
        "message": "此版本已弃用，请使用 web_interface.py"
    })


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "deprecated",
        "message": "Please use web_interface.py or web_app.py instead",
        "alternative": {
            "api": "http://localhost:8000/docs",
            "web": "http://localhost:5000"
        }
    }


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """视频上传接口"""
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
            "filename": file.filename,
            "note": "This is a deprecated endpoint"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "note": "Please use web_interface.py instead"
            }
        )


@app.get("/api/statistics")
async def get_statistics():
    """统计信息 - 提示用户使用完整版本"""
    return {
        "success": False,
        "message": "This endpoint requires web_interface.py",
        "solution": "Run: python web_interface.py"
    }


@app.get("/api/efficiency")
async def get_efficiency():
    """效率信息 - 提示用户使用完整版本"""
    return {
        "success": False,
        "message": "This endpoint requires web_interface.py",
        "solution": "Run: python web_interface.py"
    }


@app.get("/api/timeline")
async def get_timeline():
    """时间线 - 提示用户使用完整版本"""
    return {
        "success": False,
        "message": "This endpoint requires web_interface.py",
        "solution": "Run: python web_interface.py"
    }


@app.get("/api/anomalies")
async def get_anomalies():
    """异常检测 - 提示用户使用完整版本"""
    return {
        "success": False,
        "message": "This endpoint requires web_interface.py",
        "solution": "Run: python web_interface.py"
    }


@app.post("/api/reset")
async def reset_analysis():
    """重置分析"""
    return {
        "success": True,
        "message": "Analysis reset",
        "note": "For full functionality, use web_interface.py"
    }


@app.post("/api/export_results")
async def export_results():
    """导出结果"""
    return {
        "success": False,
        "message": "This endpoint requires web_interface.py",
        "solution": "Run: python web_interface.py"
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("WARNING: web_app_simple.py is deprecated!")
    print("Please use web_interface.py instead:")
    print("  python web_interface.py")
    print("="*60 + "\n")
    
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
