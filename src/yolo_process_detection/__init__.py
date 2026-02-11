"""
YOLO工序检测系统

基于Ultralytics YOLO的工业视觉检测系统，支持实时目标检测、工序分析、异常检测和效率评估。

## 功能特性

- **实时目标检测**：基于YOLOv11/v12的精确目标检测
- **多目标跟踪**：基于IoU的稳定目标跟踪算法
- **场景理解**：自动识别当前工序阶段
- **异常检测**：基于历史数据的实时异常检测
- **效率分析**：全面的性能监控和优化建议
- **Web接口**：FastAPI提供的RESTful API
- **视频处理**：支持视频文件和网络摄像头

## 快速开始

```bash
# 安装依赖
pip install -e .

# 启动Web服务
python -m yolo_process_detection.web_interface

# 使用API
curl -X POST -F "image=@test.jpg" http://localhost:5000/api/detect
```

## 目录结构

```
yolo_process_detection/
├── core/           # 核心配置和工具
├── models/         # 检测和跟踪模型
├── services/       # 业务逻辑服务
├── api/           # FastAPI路由
├── schemas/       # Pydantic模式定义
└── web_interface/ # Web界面
"""

__version__ = "1.0.0"
__author__ = "YOLO Process Detection Team"
