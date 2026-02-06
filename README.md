# YOLO工序检测系统

基于YOLOv8的工业生产工序检测与监控系统，支持实时视频流处理、智能目标跟踪、工序阶段识别和效率分析。

## 核心功能

- 实时目标检测：使用YOLOv8模型检测工人、机器、产品、工具等关键元素
- 多目标跟踪：基于IoU匹配的目标跟踪算法，跟踪生产过程中的关键对象
- 工序识别：根据检测到的对象组合自动识别当前工序阶段
- 效率分析：计算生产效率、识别瓶颈、分析工序时间分布
- 异常检测：自动检测生产过程中的异常情况
- Web界面：提供友好的Web界面进行视频上传、处理和结果查看

## 项目结构

```
yolo_process_detection/
├── config.py              # 配置文件
├── detector.py            # 目标检测模块
├── tracker.py             # 多目标跟踪模块
├── video_processor.py     # 视频处理模块
├── analyzer.py            # 数据分析模块
├── data_utils.py          # 数据准备工具
├── web_interface.py       # Web界面
├── main.py                # 主程序入口
├── train.py               # 模型训练脚本
├── example_usage.py       # 使用示例
├── requirements.txt       # 依赖包
├── templates/
│   └── index.html        # Web界面模板
├── data/                  # 数据目录
├── models/               # 模型目录
└── outputs/             # 输出目录
```

## 安装

1. 创建虚拟环境（推荐）：

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 处理视频文件

```bash
python main.py video input_video.mp4 -o output_video.mp4
```

参数说明：
- `input_video.mp4`: 输入视频路径
- `-o/--output`: 输出视频路径（可选）
- `--no-display`: 不显示处理窗口

### 2. 实时摄像头检测

```bash
python main.py webcam -c 0
```

参数说明：
- `-c/--camera`: 摄像头索引，默认为0

### 3. 分析视频（不显示窗口）

```bash
python main.py analyze input_video.mp4
```

### 4. 启动Web界面

```bash
python web_interface.py
```

然后在浏览器中打开 http://localhost:5000

### 5. 运行示例程序

```bash
python example_usage.py basic       # 基本检测示例
python example_usage.py video       # 视频处理示例
python example_usage.py realtime    # 实时检测示例
python example_usage.py batch       # 批量处理示例
python example_usage.py web         # Web API示例
python example_usage.py data        # 数据准备示例
```

## 自定义配置

编辑 `config.py` 文件来自定义配置：

```python
class ProcessDetectionConfig:
    MODEL_NAME = "yolov8n.pt"  # 模型大小：n, s, m, l, x
    CONFIDENCE_THRESHOLD = 0.5  # 置信度阈值
    IOU_THRESHOLD = 0.45       # IoU阈值
    DEVICE = "cuda"             # 设备：cuda 或 cpu
    
    CLASS_NAMES = {
        0: "worker",
        1: "machine",
        2: "product",
        3: "tool",
        4: "material"
    }
    
    PROCESS_STAGES = [
        "preparation",
        "processing",
        "assembly",
        "quality_check",
        "packaging"
    ]
```

## 训练自定义模型

1. 准备数据集（YOLO格式）：

```
data/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```

2. 创建 `data.yaml` 文件：

```yaml
path: /path/to/data
train: train/images
val: val/images

nc: 5  # 类别数量
names: ['worker', 'machine', 'product', 'tool', 'material']
```

3. 开始训练：

```bash
python train.py data/data.yaml --epochs 100 --size n
```

4. 导出模型为ONNX格式：

```bash
python train.py data/data.yaml --export
```

## API接口

Web界面提供以下API端点：

- `POST /api/upload_video`: 上传视频文件
- `POST /api/process_video`: 处理上传的视频
- `GET /api/statistics`: 获取检测统计信息
- `GET /api/efficiency`: 获取效率分析
- `GET /api/timeline`: 获取工序时间线
- `GET /api/anomalies`: 获取异常检测结果
- `POST /api/export_results`: 导出分析结果
- `POST /api/reset`: 重置分析

## 数据分析

系统提供以下分析功能：

- 检测统计：总检测数、每帧平均检测数、各类别分布
- 工序分析：工序时间分布、效率计算、瓶颈识别
- 异常检测：基于统计的异常检测
- 轨迹分析：对象轨迹、移动距离、停留时间

## 性能优化

- 使用GPU加速（CUDA）
- 调整模型大小（n/s/m/l/x）
- 优化置信度和IoU阈值
- 启用视频帧跳过处理
- 使用多线程处理

## 常见问题

1. **CUDA内存不足**：使用更小的模型（yolov8n.pt）或降低批处理大小
2. **检测速度慢**：使用GPU或减小输入图像尺寸
3. **跟踪丢失**：调整跟踪参数（max_age, min_hits）
4. **工序识别不准**：调整工序识别规则或增加类别

## 技术栈

- 检测：Ultralytics YOLOv8
- 跟踪：自定义IoU跟踪器
- 视频处理：OpenCV
- Web界面：Flask + HTML/CSS/JavaScript
- 数据处理：NumPy, Pandas

## 许可证

MIT License

## 贡献

欢迎提交问题和拉取请求！
