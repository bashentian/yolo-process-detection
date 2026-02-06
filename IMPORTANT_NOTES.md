# YOLO工序检测系统 - 重要注意事项

## 项目补充内容说明

本项目已经实现了完整的YOLO工序检测功能，以下是需要注意和补充的重要内容：

## 1. 首次使用步骤

### 初始化项目
```bash
cd yolo_process_detection
python setup.py
```

这会创建必要的目录结构、配置文件模板和.gitignore文件。

### 安装依赖
```bash
pip install -r requirements.txt
```

### 快速测试
```bash
python quick_start.py
```

## 2. 重要注意事项

### 2.1 模型管理
- **首次运行**：系统会自动下载YOLOv8模型到 `models/` 目录
- **模型大小**：根据硬件选择合适的模型大小
  - `yolov8n.pt`: 最小最快，适合CPU和移动设备
  - `yolov8s.pt`: 小型，平衡速度和精度
  - `yolov8m.pt`: 中型，更高精度
  - `yolov8l.pt`: 大型，高精度但速度较慢
  - `yolov8x.pt`: 超大型，最高精度

### 2.2 硬件要求
**最低配置：**
- CPU: Intel i5 或同等性能
- 内存: 8GB RAM
- 存储: 10GB 可用空间

**推荐配置：**
- CPU: Intel i7 或同等性能
- GPU: NVIDIA GPU (支持CUDA)
- 内存: 16GB RAM
- 存储: 20GB 可用空间

### 2.3 数据准备
训练自定义模型需要准备数据集：

**目录结构：**
```
data/
├── train/
│   ├── images/  # 训练图像
│   └── labels/  # YOLO格式标签
├── val/
│   ├── images/  # 验证图像
│   └── labels/  # YOLO格式标签
└── data.yaml    # 数据配置文件
```

**标签格式（YOLO）：**
```
<class_id> <x_center> <y_center> <width> <height>
```
所有值都是归一化的（0-1之间）。

### 2.4 配置说明

**config.py 主要参数：**
- `MODEL_NAME`: 模型文件名
- `CONFIDENCE_THRESHOLD`: 检测置信度阈值（0-1）
- `IOU_THRESHOLD`: 非极大值抑制的IoU阈值
- `DEVICE`: 运行设备（"cuda" 或 "cpu"）
- `CLASS_NAMES`: 自定义类别名称
- `PROCESS_STAGES`: 工序阶段定义

## 3. 常见问题解决

### 3.1 CUDA内存不足
```python
# 解决方案1：使用更小的模型
MODEL_NAME = "yolov8n.pt"

# 解决方案2：降低批处理大小
# 在train.py中修改batch_size参数
model.train(..., batch=8, ...)

# 解决方案3：降低输入图像尺寸
# 在train.py中修改imgsz参数
model.train(..., imgsz=512, ...)
```

### 3.2 检测速度慢
```python
# 使用GPU加速
DEVICE = "cuda"

# 降低输入分辨率
FRAME_RESIZE = (480, 480)

# 使用更小的模型
MODEL_NAME = "yolov8n.pt"
```

### 3.3 跟踪丢失
```python
# 在config.py中调整跟踪参数
TRACKING_MAX_AGE = 50      # 增加最大存活帧数
TRACKING_MIN_HITS = 1      # 降低最小命中次数
```

### 3.4 工序识别不准确
**解决方案：**
1. 调整 `detector.py` 中的 `analyze_process_stage` 方法
2. 添加更多类别或修改类别组合规则
3. 增加检测置信度阈值

## 4. 性能优化建议

### 4.1 检测优化
- 使用GPU加速（CUDA）
- 选择合适的模型大小
- 调整输入图像尺寸
- 启用批处理

### 4.2 跟踪优化
- 适当调整跟踪参数
- 减少不必要的轨迹记录
- 使用更高效的匹配算法

### 4.3 视频处理优化
- 跳帧处理（每N帧处理一帧）
- 使用多线程/多进程
- 优化视频编解码参数

## 5. 部署说明

### 5.1 Docker部署
```bash
# 构建镜像
docker-compose build

# 启动服务（GPU版本）
docker-compose up

# 启动服务（CPU版本）
docker-compose --profile cpu up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 5.2 Web界面部署
```bash
# 启动Web服务
python web_interface.py

# 访问界面
# 浏览器打开: http://localhost:5000
```

### 5.3 生产环境部署
**注意事项：**
1. 使用生产级Web服务器（Gunicorn + Nginx）
2. 配置HTTPS
3. 设置适当的日志级别
4. 实施监控和告警
5. 配置自动备份

## 6. 模型训练

### 6.1 准备数据集
```bash
# 提取视频帧
python -c "
from data_utils import DatasetPreparer
preparer = DatasetPreparer('data/my_dataset')
preparer.extract_frames_from_video('video.mp4', 'data/my_dataset/images', 30)
"

# 标注数据
python -c "
from data_utils import AnnotationTool
annotator = AnnotationTool('data/my_dataset/images', ['worker', 'machine'])
annotator.annotate_images()
"

# 划分数据集
python -c "
from data_utils import DatasetPreparer
preparer = DatasetPreparer('data/my_dataset')
preparer.split_dataset(0.8)
"
```

### 6.2 训练模型
```bash
python train.py data/data.yaml --epochs 100 --size n --export
```

### 6.3 评估模型
```python
from evaluator import ModelEvaluator, ResultVisualizer
from config import ProcessDetectionConfig

config = ProcessDetectionConfig()
evaluator = ModelEvaluator(config)

results = evaluator.evaluate_on_dataset('data/val/images', 'data/val/labels')
evaluator.print_evaluation_report(results)

# 可视化结果
visualizer = ResultVisualizer()
visualizer.plot_metrics_over_time(results)
```

## 7. 安全注意事项

### 7.1 数据安全
- 不要将敏感数据上传到公共仓库
- 使用环境变量存储配置信息
- 定期备份重要数据

### 7.2 API安全
- 实施速率限制
- 添加认证机制
- 验证用户输入
- 使用HTTPS

### 7.3 隐私保护
- 面部识别时注意隐私法规
- 必要时使用匿名化处理
- 遵守GDPR等数据保护法规

## 8. 扩展功能建议

### 8.1 实时告警
- 邮件通知
- 短信通知
- Webhook集成
- 消息队列集成（RabbitMQ、Kafka）

### 8.2 数据存储
- 数据库集成（PostgreSQL、MongoDB）
- 对象存储（S3、MinIO）
- 时序数据库（InfluxDB、TimescaleDB）

### 8.3 高级分析
- 机器学习预测
- 趋势分析
- 预测性维护
- 质量预测

### 8.4 集成其他系统
- ERP系统集成
- MES系统集成
- SCADA系统集成
- IoT平台集成

## 9. 维护和更新

### 9.1 定期维护
- 更新依赖包
- 清理缓存文件
- 检查日志文件大小
- 备份配置和数据

### 9.2 版本更新
- 遵循语义化版本控制
- 维护变更日志
- 提供迁移指南
- 充分测试新版本

## 10. 学习资源

### 10.1 官方文档
- Ultralytics YOLOv8: https://docs.ultralytics.com/
- OpenCV文档: https://docs.opencv.org/
- Flask文档: https://flask.palletsprojects.com/

### 10.2 推荐阅读
- 计算机视觉基础
- 深度学习理论
- 目标检测算法
- 多目标跟踪算法

### 10.3 实践项目
- 尝试不同的工业场景
- 优化模型性能
- 开发自定义功能
- 参与开源项目

## 11. 故障排查清单

- [ ] 检查Python版本（推荐3.8+）
- [ ] 检查依赖包是否正确安装
- [ ] 检查CUDA版本（如果使用GPU）
- [ ] 检查文件路径是否正确
- [ ] 检查日志文件中的错误信息
- [ ] 验证数据集格式是否正确
- [ ] 检查磁盘空间是否充足
- [ ] 测试硬件资源使用情况
- [ ] 尝试降低配置要求
- [ ] 查看社区支持和文档

## 12. 联系和支持

如果遇到问题：
1. 查看日志文件（outputs/logs/）
2. 检查本文档的常见问题部分
3. 运行单元测试验证系统状态
4. 查看官方文档和社区资源

---

**最后更新：2025年**

祝您使用愉快！
