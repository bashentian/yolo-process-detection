# NVIDIA Jetson Nano 部署指南

## 概述

本指南专门针对NVIDIA Jetson Nano边缘设备，提供完整的模型量化、TensorRT优化和部署方案。

## 一、Jetson Nano 硬件特性

### 1.1 硬件规格
- **GPU**: NVIDIA Tegra X1 (128 CUDA cores)
- **CPU**: Quad-core ARM Cortex-A57
- **内存**: 4GB 64-bit LPDDR4
- **架构**: ARM64 (aarch64)
- **AI加速**: 支持TensorRT 8.x、CUDA 10.2、cuDNN 8.0

### 1.2 计算能力
- **FP32**: 完整支持
- **FP16**: 硬件加速（推荐）
- **INT8**: 支持但需要校准
- **INT4**: 不支持

### 1.3 性能特点
- 功耗：5W-10W
- 适合实时推理（30+ FPS）
- 需要模型优化才能获得良好性能

## 二、模型优化流程

### 2.1 优化路线图

```
PyTorch (.pt)
    ↓
ONNX (.onnx)
    ↓
ONNX 量化 (可选)
    ↓
TensorRT Engine (.engine)
    ↓
Jetson部署
```

### 2.2 快速优化命令

**在PC上准备模型：**
```bash
# 1. 训练模型
python train.py data/data.yaml --epochs 100 --size n

# 2. 优化为Jetson版本
python jetson_deployment.py --model models/custom_process_detection/weights/best.pt \
    --mode optimize \
    --fp16 \
    --quantize-onnx
```

**在Jetson Nano上运行：**
```bash
# 传输优化后的模型到Jetson
scp -r jetson_optimized/ jetson@nano.local:/home/jetson/

# 在Jetson上运行
ssh jetson@nano.local
cd jetson_optimized
python inference_jetson.py
```

## 三、模型量化

### 3.1 量化类型对比

| 量化类型 | 精度损失 | 速度提升 | 模型大小 | 适用场景 |
|---------|-----------|---------|---------|---------|
| FP32    | 0%       | 1x      | 100%    | 精度要求高 |
| FP16     | <1%       | 2-3x    | 50%     | 推荐使用 |
| INT8     | 1-3%      | 3-5x    | 25%     | 需要校准 |

### 3.2 FP16 量化（推荐）

**优势：**
- 硬件加速，无需校准
- 精度损失极小
- 模型大小减半
- 推理速度提升2-3倍

**使用方法：**
```python
from jetson_deployment import JetsonModelOptimizer

optimizer = JetsonModelOptimizer("model.pt", config)

# FP16优化
results = optimizer.optimize_for_jetson(
    "model.pt",
    fp16=True,      # 启用FP16
    int8=False,     # 不使用INT8
    quantize_onnx=False  # 不量化ONNX
)
```

### 3.3 INT8 量化

**需要校准数据：**
```python
from jetson_deployment import JetsonModelOptimizer

optimizer = JetsonModelOptimizer("model.pt", config)

# 准备校准数据（100-200张图像）
calibration_images = ["data/calib/img1.jpg", "data/calib/img2.jpg", ...]

# INT8优化
results = optimizer.optimize_for_jetson(
    "model.pt",
    fp16=True,      # FP16 + INT8
    int8=True,       # 启用INT8
    quantize_onnx=True
)
```

**校准数据准备：**
```python
from jetson_deployment import JetsonDeploymentManager
import cv2

manager = JetsonDeploymentManager(config)

# 自动从目录加载校准图像
results = manager.prepare_jetson_deployment(
    "model.pt",
    calibration_images_dir="data/calibration_images",  # 校准图像目录
    fp16=True,
    int8=True
)
```

### 3.4 ONNX 动态量化

**简单快速，无需校准：**
```python
optimizer = JetsonModelOptimizer("model.pt", config)

# 转换为ONNX
onnx_path = optimizer.convert_to_onnx("model.pt")

# INT8量化
quantized_path = optimizer.quantize_onnx(
    onnx_path,
    quantization_mode="int8"  # 或 "int16"
)

print(f"Quantized model: {quantized_path}")
```

## 四、TensorRT 优化

### 4.1 TensorRT 优势

- **层融合**: 合并多个操作
- **内核优化**: 针对GPU架构优化
- **精度校准**: 自动选择最佳精度
- **内存优化**: 减少内存访问

### 4.2 TensorRT Engine 构建

**基础构建（FP16）：**
```python
optimizer = JetsonModelOptimizer("model.pt", config)

# 转换流程
onnx_path = optimizer.convert_to_onnx("model.pt")
engine_path = optimizer.convert_to_tensorrt(
    onnx_path,
    fp16=True,      # FP16加速
    int8=False,     # 不使用INT8
    max_batch_size=1,
    max_workspace_size=1<<30  # 1GB工作空间
)
```

**高级构建（FP16 + INT8）：**
```python
# 准备校准数据
calibration_data = []
for i in range(100):
    img = cv2.imread(f"calib/calib_{i}.jpg")
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    calibration_data.append(img)

# 构建TensorRT引擎
engine_path = optimizer.convert_to_tensorrt(
    onnx_path,
    fp16=True,
    int8=True,          # 启用INT8
    calibration_data=calibration_data,  # 提供校准数据
    max_batch_size=1
)
```

### 4.3 动态批处理

**支持可变输入大小：**
```python
# 转换ONNX时启用动态批处理
onnx_path = optimizer.convert_to_onnx(
    "model.pt",
    dynamic_batch=True  # 启用动态批处理
)
```

## 五、Jetson Nano 部署

### 5.1 环境配置

**安装JetPack SDK：**
```bash
# JetPack 4.6+ 已包含所需组件
# 检查版本
cat /etc/nv_tegra_release
```

**安装Python依赖：**
```bash
# 在Jetson Nano上执行
sudo apt update
sudo apt install python3-pip libopencv-python3 python3-opencv

pip3 install numpy==1.19.5
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/l4t
pip3 install onnx onnxruntime-gpu
pip3 install tensorrt
```

### 5.2 推理引擎使用

**加载TensorRT引擎：**
```python
from jetson_deployment import JetsonInferenceEngine
from config import ProcessDetectionConfig

config = ProcessDetectionConfig()

# 使用TensorRT引擎
engine = JetsonInferenceEngine(
    "jetson_optimized/model.engine",  # TensorRT引擎文件
    config,
    use_tensorrt=True  # 强制使用TensorRT
)

# 推理
image = cv2.imread("test.jpg")
detections = engine.infer(image)

for det in detections:
    print(f"{det.class_name}: {det.confidence:.2f}")
```

**性能基准测试：**
```python
# 运行基准测试
results = engine.benchmark(num_iterations=100)

print(f"推理时间: {results['avg_inference_time_ms']:.2f} ms")
print(f"FPS: {results['fps']:.2f}")
```

### 5.3 实时视频处理

**完整部署脚本：**
```python
import cv2
from jetson_deployment import JetsonInferenceEngine
from video_processor import VideoProcessor
from config import ProcessDetectionConfig

config = ProcessDetectionConfig()
config.DEVICE = "cuda"

# 使用优化的TensorRT引擎
engine = JetsonInferenceEngine(
    "jetson_optimized/model.engine",
    config,
    use_tensorrt=True
)

def jetson_process_callback(frame, detections, stage):
    display_frame = cv2.resize(frame, (1280, 720))
    
    cv2.putText(display_frame, f"Stage: {stage}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display_frame, f"FPS: {engine.get_fps():.1f}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Jetson Detection', display_frame)

# 启动摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    detections = engine.infer(frame)
    stage = analyze_process_stage(detections)
    
    jetson_process_callback(frame, detections, stage)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 六、性能优化策略

### 6.1 Jetson Nano 优化技巧

**1. 启用最大性能模式：**
```bash
# 设置为最大性能模式
sudo nvpmodel -m 0
sudo jetson_clocks
```

**2. 降低分辨率：**
```python
# 使用较小的输入尺寸
config.FRAME_RESIZE = (416, 416)  # 从640降低到416
```

**3. 使用FP16混合精度：**
```python
# 在推理中启用FP16
torch.set_float32_matmul_precision('high')
```

**4. 内存优化：**
```python
# 清理GPU缓存
torch.cuda.empty_cache()

# 使用较小的批处理
batch_size = 1
```

### 6.2 性能基准对比

**运行完整基准测试：**
```bash
python jetson_deployment.py --model models/custom_process_detection/weights/best.pt \
    --mode benchmark
```

**预期性能提升：**
```
模型格式        推理时间    FPS      内存占用    模型大小
PyTorch        45ms       22       1.2GB      6MB
ONNX           30ms       33       0.8GB      5MB
TensorRT FP16   15ms       67       0.4GB      3MB
TensorRT INT8   12ms       83       0.3GB      1.5MB
```

## 七、故障排查

### 7.1 常见问题

**问题1：TensorRT转换失败**
```bash
# 解决方案：检查ONNX版本兼容性
pip install onnx==1.13.0
pip install onnxruntime-gpu==1.13.0

# 使用正确的opset版本
# 在jetson_deployment.py中设置opset_version=17
```

**问题2：INT8精度下降过大**
```python
# 解决方案：增加校准数据数量
calibration_images = list(Path("calib").glob("*.jpg"))[:200]  # 使用200张

# 使用混合精度
fp16=True, int8=True  # FP16 + INT8混合
```

**问题3：内存不足**
```python
# 解决方案：减小工作空间
optimizer.convert_to_tensorrt(
    onnx_path,
    max_workspace_size=1<<28  # 从1GB降到256MB
)

# 或降低输入分辨率
config.FRAME_RESIZE = (416, 416)
```

**问题4：推理速度慢**
```bash
# 解决方案：启用最大性能
sudo nvpmodel -m 0
sudo jetson_clocks

# 检查GPU使用情况
tegrastats
```

### 7.2 调试技巧

**启用详细日志：**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 或在TensorRT中启用详细日志
logger = trt.Logger(trt.Logger.VERBOSE)
```

**监控资源使用：**
```bash
# 实时监控
tegrastats --interval 1000

# 内存使用
sudo jtop  # 需要安装jtop
```

## 八、部署检查清单

### 8.1 部署前检查

- [ ] 模型在PC上正常训练和验证
- [ ] 模型已转换为ONNX格式
- [ ] ONNX模型已优化（FP16/INT8）
- [ ] TensorRT引擎已成功构建
- [ ] 推理速度满足要求（>30 FPS）
- [ ] 内存使用在限制范围内（<4GB）
- [ ] 所有依赖包在Jetson上安装完成
- [ ] 摄像头正常工作
- [ ] 电源供应充足（5V 4A）

### 8.2 部署后检查

- [ ] 应用正常启动
- [ ] 摄像头画面正常显示
- [ ] 检测功能正常
- [ ] FPS满足实时要求
- [ ] 温度在正常范围（<60°C）
- [ ] 长时间运行稳定
- [ ] 异常情况正常处理

## 九、最佳实践

### 9.1 开发流程

**1. PC端开发和训练**
```bash
# 在PC上完成所有开发
python train.py data/data.yaml --epochs 100
python advanced_training.py --mode distill --teacher best.pt --student-size n
```

**2. 模型优化**
```bash
# 在PC上完成优化
python jetson_deployment.py --model distilled/final_student.pt \
    --mode optimize \
    --fp16 \
    --quantize-onnx
```

**3. Jetson端部署**
```bash
# 传输到Jetson
scp -r jetson_optimized/ jetson@nano.local:~

# 在Jetson上测试
ssh jetson@nano.local
cd jetson_optimized
python inference_jetson.py
```

### 9.2 持续优化

**定期重新校准：**
- 如果场景光照变化大，定期更新INT8校准数据
- 保存不同场景的校准数据

**模型迭代：**
- 收集Jetson上的实际数据
- 反馈到训练流程
- 重新优化模型

**监控和维护：**
- 定期检查GPU温度
- 监控内存使用
- 更新JetPack和驱动

## 十、参考资源

- [NVIDIA Jetson Nano官方文档](https://developer.nvidia.com/embedded/jetson-nano)
- [TensorRT开发指南](https://docs.nvidia.com/deeplearning/tensorrt/)
- [JetPack SDK下载](https://developer.nvidia.com/jetpack)
- [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-0-4-now-available/72048)

---

**相关文件：**
- [jetson_deployment.py](file:///d:/my_world/python_workspace/ai_workspace/yolo_process_detection/jetson_deployment.py) - Jetson优化工具
- [train.py](file:///d:/my_world/python_workspace/ai_workspace/yolo_process_detection/train.py) - 模型训练
- [advanced_training.py](file:///d:/my_world/python_workspace/ai_workspace/yolo_process_detection/advanced_training.py) - 高级训练
