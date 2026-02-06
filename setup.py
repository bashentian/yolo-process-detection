import os
from pathlib import Path


def create_project_structure():
    base_dir = Path(__file__).parent
    
    directories = [
        base_dir / "data" / "images",
        base_dir / "data" / "labels",
        base_dir / "data" / "train" / "images",
        base_dir / "data" / "train" / "labels",
        base_dir / "data" / "val" / "images",
        base_dir / "data" / "val" / "labels",
        base_dir / "data" / "test" / "images",
        base_dir / "data" / "test" / "labels",
        base_dir / "models" / "pretrained",
        base_dir / "models" / "custom",
        base_dir / "models" / "exports",
        base_dir / "outputs" / "videos",
        base_dir / "outputs" / "images",
        base_dir / "outputs" / "analysis",
        base_dir / "outputs" / "logs",
        base_dir / "uploads",
        base_dir / "cache"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")
    
    print("\nProject structure created successfully!")
    print("\nNext steps:")
    print("1. Place your training data in data/train/ and data/val/")
    print("2. Create a data.yaml file for training")
    print("3. Run 'python quick_start.py' to test the system")


def create_gitignore():
    base_dir = Path(__file__).parent
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Project specific
data/train/
data/val/
data/test/
outputs/
uploads/
cache/
models/custom/
models/exports/

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# ML models
*.pt
*.onnx
*.engine
!models/pretrained/*.pt

# Temporary files
tmp/
temp/
"""
    
    gitignore_path = base_dir / ".gitignore"
    with open(gitignore_path, 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    
    print(f"Created .gitignore at {gitignore_path}")


def create_env_template():
    env_content = """# YOLO Model Configuration
MODEL_SIZE=n
MODEL_PATH=pretrained/yolov8n.pt

# Detection Parameters
CONFIDENCE_THRESHOLD=0.5
IOU_THRESHOLD=0.45
MAX_DETECTIONS=100

# Device Configuration
DEVICE=cpu

# Video Processing
VIDEO_WIDTH=640
VIDEO_HEIGHT=640
DISPLAY_WIDTH=1280
DISPLAY_HEIGHT=720
FPS_LIMIT=30

# Training Configuration
BATCH_SIZE=16
EPOCHS=100
IMG_SIZE=640
LEARNING_RATE=0.01

# API Configuration
API_HOST=0.0.0.0
API_PORT=5000
DEBUG=True

# Paths
DATA_DIR=data
MODELS_DIR=models
OUTPUTS_DIR=outputs
UPLOADS_DIR=uploads
"""
    
    base_dir = Path(__file__).parent
    env_path = base_dir / ".env.example"
    with open(env_path, 'w', encoding='utf-8') as f:
        f.write(env_content)
    
    print(f"Created .env.example at {env_path}")


def create_data_yaml_template():
    yaml_content = """# YOLO训练数据配置文件

# 数据集路径（绝对路径或相对于此文件的路径）
path: /path/to/dataset  # 替换为你的数据集路径

# 训练和验证集图像路径
train: train/images
val: val/images
test: test/images  # 可选

# 类别数量
nc: 5

# 类别名称（按类别索引顺序）
names:
  0: worker
  1: machine
  2: product
  3: tool
  4: material

# 下载URL（可选，用于下载数据集）
# download: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip

# 数据增强参数（可选）
# augmentation:
#   hsv_h: 0.015
#   hsv_s: 0.7
#   hsv_v: 0.4
#   degrees: 0.0
#   translate: 0.1
#   scale: 0.5
#   shear: 0.0
#   perspective: 0.0
#   flipud: 0.0
#   fliplr: 0.5
#   mosaic: 1.0
#   mixup: 0.0
"""
    
    base_dir = Path(__file__).parent
    yaml_path = base_dir / "data" / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"Created data.yaml template at {yaml_path}")


def main():
    print("=== YOLO工序检测系统 - 项目初始化 ===\n")
    
    print("1. Creating project directory structure...")
    create_project_structure()
    
    print("\n2. Creating .gitignore...")
    create_gitignore()
    
    print("\n3. Creating .env.example...")
    create_env_template()
    
    print("\n4. Creating data.yaml template...")
    create_data_yaml_template()
    
    print("\n" + "="*50)
    print("项目初始化完成！")
    print("="*50)
    print("\n请按照以下步骤继续：")
    print("1. 复制 .env.example 为 .env 并修改配置")
    print("2. 编辑 data/data.yaml 配置你的数据集")
    print("3. 运行 'python quick_start.py' 进行快速测试")
    print("4. 参考 README.md 了解详细使用方法")


if __name__ == "__main__":
    main()
