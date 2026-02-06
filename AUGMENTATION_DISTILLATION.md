# 数据增强与模型蒸馏指南

## 概述

本项目现已支持完整的数据增强和YOLO模型蒸馏功能，可以显著提升模型性能和部署效率。

## 一、数据增强

### 1.1 基础数据增强

在 [train.py](file:///d:/my_world/python_workspace/ai_workspace/yolo_process_detection/train.py) 中已集成YOLOv8的内置数据增强：

**支持的增强方法：**
- `hsv_h`: 色调增强 (0.015)
- `hsv_s`: 饱和度增强 (0.7)
- `hsv_v`: 明度增强 (0.4)
- `degrees`: 旋转角度 (10.0°)
- `translate`: 平移 (0.1)
- `scale`: 缩放 (0.5)
- `shear`: 剪切 (2.0°)
- `perspective`: 透视变换 (0.0)
- `flipud`: 垂直翻转 (0.0)
- `fliplr`: 水平翻转 (0.5)
- `mosaic`: 马赛克增强 (1.0)
- `mixup`: 图像混合 (0.1)

### 1.2 高级数据增强

在 [advanced_training.py](file:///d:/my_world/python_workspace/ai_workspace/yolo_process_detection/advanced_training.py) 中提供了更高级的增强方法：

#### Mosaic增强
将4张图像组合成一张，增加训练样本多样性：
```python
from advanced_training import AdvancedDataAugmentation

augmentor = AdvancedDataAugmentation()

# 准备4张图像和对应的标签
images = [img1, img2, img3, img4]
labels = [label1, label2, label3, label4]

# 执行mosaic增强
mosaic_img, mosaic_labels = augmentor.mosaic_augmentation(
    images, labels, grid_size=2
)
```

#### Mixup增强
混合两张图像和标签：
```python
# 混合两张图像
mixed_img, mixed_labels = augmentor.mixup_augmentation(
    img1, label1,
    img2, label2,
    alpha=0.2  # 混合比例
)
```

#### 随机透视变换
```python
# 应用随机透视变换
transformed_img, transformed_labels = augmentor.random_perspective(
    image, labels,
    degree=5,      # 旋转角度
    translate=0.1,  # 平移比例
    scale=(0.9, 1.1),  # 缩放范围
    shear=2        # 剪切角度
)
```

### 1.3 自定义数据增强

在 [data_utils.py](file:///d:/my_world/python_workspace/ai_workspace/yolo_process_detection/data_utils.py) 中提供了基础增强工具：

```python
from data_utils import DataAugmentor

augmentor = DataAugmentor()

# 单一增强方法
rotated = augmentor._rotate(image, angle=15)
flipped = augmentor._flip(image, direction=1)
brightened = augmentor._adjust_brightness(image, factor=1.2)
noisy = augmentor._add_noise(image, noise_factor=0.1)

# 组合增强
augmented = augmentor.augment_image(
    image,
    methods=["rotate", "flip", "brightness"]
)

# 批量增强数据集
augmentor.augment_dataset(
    "data/train/images",
    "data/train/augmented",
    augmentation_factor=3
)
```

### 1.4 使用数据增强训练

**方法1：使用内置增强（推荐）**
```bash
# 默认启用数据增强
python train.py data/data.yaml --epochs 100 --size n

# 禁用数据增强
python train.py data/data.yaml --epochs 100 --no-augmentation
```

**方法2：使用高级增强**
```bash
# 使用高级增强训练
python advanced_training.py --mode augment --data data/data.yaml --epochs 100
```

## 二、模型蒸馏

### 2.1 什么是模型蒸馏？

模型蒸馏是一种模型压缩技术，通过将大型"教师"模型的知识转移到小型"学生"模型中，在保持精度的同时显著减少模型大小和推理时间。

### 2.2 蒸馏优势

- **模型大小减少**：学生模型通常比教师模型小2-10倍
- **推理速度提升**：小型模型推理速度更快
- **部署成本降低**：适合边缘设备和移动设备部署
- **精度保持**：通过知识蒸馏保持较高精度

### 2.3 YOLO蒸馏实现

在 [advanced_training.py](file:///d:/my_world/python_workspace/ai_workspace/yolo_process_detection/advanced_training.py) 中实现了完整的YOLO蒸馏功能：

#### 核心组件

**1. YOLODistiller类**
```python
from advanced_training import YOLODistiller

# 初始化蒸馏器
distiller = YOLODistiller(
    teacher_model=teacher,
    student_model=student,
    config=config
)
```

**2. 蒸馏损失函数**
- 蒸馏损失：KL散度损失（温度软化）
- 检测损失：标准YOLO检测损失
- 总损失：α × 蒸馏损失 + β × 检测损失

**3. 温度参数**
- 用于软化教师模型的输出分布
- 较高的温度值产生更平滑的概率分布
- 典型值：3.0-10.0

### 2.4 使用模型蒸馏

#### 步骤1：训练教师模型
```bash
# 训练大型教师模型
python train.py data/data.yaml --epochs 150 --size m
```

#### 步骤2：执行模型蒸馏
```bash
# 使用YOLOv8m作为教师，蒸馏到YOLOv8n
python advanced_training.py --mode distill \
    --teacher models/custom_process_detection/weights/best.pt \
    --student-size n \
    --data data/data.yaml \
    --epochs 50
```

#### 步骤3：评估蒸馏结果
```python
from advanced_training import YOLODistiller
from config import ProcessDetectionConfig
from torch.utils.data import DataLoader
from ultralytics import YOLO

config = ProcessDetectionConfig()

# 加载模型
teacher = YOLO("models/custom_process_detection/weights/best.pt")
student = YOLO("models/distilled/final_student.pt")

# 创建蒸馏器并比较
distiller = YOLODistiller(teacher, student, config)

# 准备测试数据
from advanced_training import YOLODataset
test_dataset = YOLODataset("data/test/images", "data/test/labels")
test_loader = DataLoader(test_dataset, batch_size=1)

# 比较模型性能
comparison = distiller.compare_models(test_loader)

print(f"Teacher mAP: {comparison['teacher']['mAP']:.4f}")
print(f"Student mAP: {comparison['student']['mAP']:.4f}")
print(f"Speed Improvement: {comparison['speed_improvement']:.2f}x")
print(f"Compression Ratio: {comparison['compression_ratio']:.2f}x")
```

### 2.5 蒸馏参数调优

```python
# 自定义蒸馏参数
distiller = YOLODistiller(teacher, student, config)

# 调整温度参数
distiller.temperature = 5.0  # 默认5.0

# 调整损失权重
distiller.alpha = 0.7   # 蒸馏损失权重
distiller.beta = 0.3    # 检测损失权重

# 执行蒸馏
history = distiller.distill(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    learning_rate=0.001,
    output_dir="models/distilled"
)
```

## 三、实战案例

### 案例1：小样本数据训练

```python
from advanced_training import train_with_augmentation

# 使用强数据增强训练小数据集
model = train_with_augmentation(
    data_yaml="data/data.yaml",
    use_advanced_augmentation=True,
    epochs=200,
    batch_size=8,
    img_size=640
)
```

### 案例2：模型压缩部署

```python
# 完整的模型压缩流程
# 1. 训练教师模型
python train.py data/data.yaml --epochs 200 --size l

# 2. 蒸馏到学生模型
python advanced_training.py --mode distill \
    --teacher models/custom_process_detection/weights/best.pt \
    --student-size n \
    --data data/data.yaml \
    --epochs 100

# 3. 导出为ONNX
python train.py models/distilled/final_student.pt --export

# 4. 部署到边缘设备
# 使用蒸馏后的模型，推理速度提升3-5倍
```

### 案例3：实时应用优化

```python
# 为实时应用优化的训练流程
from advanced_training import YOLODistiller, YOLODataset
from torch.utils.data import DataLoader

# 1. 准备数据集
train_dataset = YOLODataset(
    "data/train/images",
    "data/train/labels",
    augment=True  # 启用实时增强
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)

# 2. 执行蒸馏
distiller = YOLODistiller(teacher, student, config)
distiller.temperature = 3.0  # 较低温度用于实时应用
distiller.alpha = 0.6

history = distiller.distill(
    train_loader=train_loader,
    epochs=30,
    learning_rate=0.01  # 较高学习率
)
```

## 四、最佳实践

### 4.1 数据增强最佳实践

1. **逐步增加增强强度**
   ```python
   # 早期训练阶段
   augmentation_params = {
       'fliplr': 0.5,
       'degrees': 5.0,
       'scale': 0.3
   }
   
   # 后期训练阶段
   augmentation_params = {
       'fliplr': 0.5,
       'degrees': 15.0,
       'scale': 0.5,
       'mosaic': 1.0,
       'mixup': 0.2
   }
   ```

2. **根据场景调整增强**
   ```python
   # 工业场景（增强少，保持精确）
   augmentation_params = {
       'fliplr': 0.3,
       'degrees': 5.0,
       'scale': 0.3
   }
   
   # 自然场景（增强多，提升泛化）
   augmentation_params = {
       'fliplr': 0.5,
       'degrees': 15.0,
       'scale': 0.5,
       'mosaic': 1.0,
       'mixup': 0.3
   }
   ```

### 4.2 模型蒸馏最佳实践

1. **选择合适的教师-学生组合**
   ```
   教师模型      学生模型      压缩比    精度损失    速度提升
   YOLOv8x    YOLOv8n     10x       2-5%       8-12x
   YOLOv8l    YOLOv8n     6x        1-3%       5-8x
   YOLOv8m    YOLOv8n     4x        0.5-2%     3-5x
   YOLOv8s    YOLOv8n     2x        0-1%       1.5-2x
   ```

2. **调整蒸馏参数**
   ```python
   # 高精度要求
   temperature = 10.0
   alpha = 0.8
   beta = 0.2
   
   # 平衡精度和速度
   temperature = 5.0
   alpha = 0.7
   beta = 0.3
   
   # 高速度要求
   temperature = 3.0
   alpha = 0.5
   beta = 0.5
   ```

### 4.3 性能优化技巧

1. **使用混合精度训练**
   ```python
   # 在train.py中添加
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   
   with autocast():
       loss = model(images, targets)
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

2. **梯度累积**
   ```python
   # 模拟更大的batch size
   accumulation_steps = 4
   
   for i, (images, targets) in enumerate(train_loader):
       loss = model(images, targets) / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

## 五、常见问题

### Q1: 数据增强后精度下降？
**A**: 检查增强参数是否过于激进：
- 降低旋转角度和缩放范围
- 减少mixup和mosaic的使用概率
- 确保增强不会改变目标对象的本质特征

### Q2: 蒸馏后精度损失过大？
**A**: 调整蒸馏参数：
- 增加温度参数（5.0-10.0）
- 提高α值（0.7-0.9），更重视教师知识
- 增加蒸馏训练的epoch数

### Q3: 蒸馏训练不收敛？
**A**: 检查以下几点：
- 确保教师模型已充分训练
- 使用较低的学习率开始
- 检查学生模型是否过于简单
- 增加warmup epoch

### Q4: 如何选择最佳增强组合？
**A**: 遵循以下原则：
- 从简单增强开始（翻转、缩放）
- 逐步添加复杂增强（mosaic、mixup）
- 在验证集上测试不同组合
- 根据具体应用场景调整

## 六、总结

通过合理使用数据增强和模型蒸馏，可以：
1. **提升模型性能**：增强数据多样性，提高泛化能力
2. **加速训练收敛**：更好的数据质量
3. **减少部署成本**：模型压缩，适合边缘部署
4. **保持推理精度**：通过知识蒸馏保持性能

建议在实际项目中结合使用这两种技术，根据具体需求调整参数。

---

**相关文件：**
- [train.py](file:///d:/my_world/python_workspace/ai_workspace/yolo_process_detection/train.py) - 基础训练
- [advanced_training.py](file:///d:/my_world/python_workspace/ai_workspace/yolo_process_detection/advanced_training.py) - 高级训练
- [data_utils.py](file:///d:/my_world/python_workspace/ai_workspace/yolo_process_detection/data_utils.py) - 数据工具
