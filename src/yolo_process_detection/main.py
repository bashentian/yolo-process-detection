"""主入口模块

提供CLI和Web服务启动入口。
"""
import sys
from pathlib import Path
from typing import Annotated

import typer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from .api import router
from .core.config import get_settings
from .core.exceptions import handle_exception
from .database import init_database


def create_app() -> FastAPI:
    """创建FastAPI应用"""
    settings = get_settings()
    
    # 初始化数据库
    init_database()
    
    app = FastAPI(
        title="YOLO Process Detection API",
        description="Industrial process monitoring using YOLO",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 挂载静态文件和模板
    templates = Jinja2Templates(directory="templates")
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    # 注册路由
    app.include_router(router, prefix="/api")
    
    # 异常处理器
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc: Exception):
        error_response, status_code = handle_exception(exc)
        return error_response
    
    # 启动事件
    @app.on_event("startup")
    async def startup_event():
        """应用启动事件"""
        print("Starting YOLO Process Detection System...")
        print(f"Model: {settings.model_name}")
        print(f"Device: {settings.device}")
        print(f"Confidence: {settings.confidence_threshold}")
    
    # 关闭事件
    @app.on_event("shutdown")
    async def shutdown_event():
        """应用关闭事件"""
        from .services.camera import get_camera_manager
        from .services.stream import cleanup_streamers
        
        print("Shutting down YOLO Process Detection System...")
        cleanup_streamers()
        manager = get_camera_manager()
        manager.stop_all()
    
    return app


app = create_app()


@app.get("/", include_in_schema=False)
async def root():
    """根路径重定向到API文档"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


def main():
    """CLI主入口"""
    typer.echo("YOLO Process Detection System")
    typer.echo("=" * 50)
    
    settings = get_settings()
    typer.echo(f"Model: {settings.model_name}")
    typer.echo(f"Device: {settings.device}")
    typer.echo(f"Confidence: {settings.confidence_threshold}")
    typer.echo()


def run_web(host: str = "0.0.0.0", port: int = 5000, reload: bool = False):
    """启动Web服务"""
    settings = get_settings()
    
    print(f"\n{'='*60}")
    print(f"YOLO Process Detection Web Interface")
    print(f"{'='*60}")
    print(f"Access: http://{host}:{port}")
    print(f"API Docs: http://{host}:{port}/docs")
    print(f"{'='*60}\n")
    
    uvicorn.run(
        "yolo_process_detection.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


def run_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """启动API服务"""
    settings = get_settings()
    
    print(f"\n{'='*60}")
    print(f"YOLO Process Detection API")
    print(f"{'='*60}")
    print(f"Access: http://{host}:{port}")
    print(f"API Docs: http://{host}:{port}/docs")
    print(f"{'='*60}\n")
    
    uvicorn.run(
        "yolo_process_detection.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
