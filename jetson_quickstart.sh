#!/bin/bash

# NVIDIA Jetson Nano 快速部署脚本
# 使用方法: bash jetson_quickstart.sh

echo "=========================================="
echo "Jetson Nano 部署准备"
echo "=========================================="

# 检查Jetson环境
if [ ! -f /etc/nv_tegra_release ]; then
    echo "错误: 此脚本需要在NVIDIA Jetson设备上运行"
    exit 1
fi

echo "检测到Jetson设备:"
cat /etc/nv_tegra_release

# 设置最大性能模式
echo ""
echo "=========================================="
echo "设置最大性能模式"
echo "=========================================="
sudo nvpmodel -m 0
sudo jetson_clocks

# 更新系统包
echo ""
echo "=========================================="
echo "更新系统包"
echo "=========================================="
sudo apt update
sudo apt upgrade -y

# 安装Python依赖
echo ""
echo "=========================================="
echo "安装Python依赖"
echo "=========================================="
sudo apt install -y \
    python3-pip \
    libopencv-python3 \
    python3-opencv \
    python3-numpy \
    python3-pandas \
    python3-matplotlib \
    python3-scipy

# 升级pip
python3 -m pip install --upgrade pip

# 安装PyTorch for Jetson
echo ""
echo "=========================================="
echo "安装PyTorch for Jetson"
echo "=========================================="
# 根据JetPack版本选择PyTorch版本
JETPACK_VERSION=$(cat /etc/nv_tegra_release | grep 'JP_VERSION' | cut -d'=' -f2 | cut -d'.' -f1)

if [ "$JETPACK_VERSION" = "4" ]; then
    echo "安装PyTorch for JetPack 4.x"
    pip3 install torch==1.10.0 torchvision==0.11.1
else
    echo "警告: 未知的JetPack版本，使用最新PyTorch"
    pip3 install torch torchvision
fi

# 安装其他Python包
echo ""
echo "=========================================="
echo "安装其他Python包"
echo "=========================================="
pip3 install \
    onnx==1.13.0 \
    onnxruntime-gpu==1.13.0 \
    ultralytics>=8.0.0 \
    pyyaml>=6.0 \
    tqdm>=4.65.0 \
    psutil>=5.9.0 \
    scikit-learn>=1.3.0 \
    flask>=3.0.0

# 创建项目目录
echo ""
echo "=========================================="
echo "创建项目目录"
echo "=========================================="
mkdir -p ~/yolo_process_detection
cd ~/yolo_process_detection

# 传输项目文件（需要在PC上执行）
echo ""
echo "=========================================="
echo "项目文件传输"
echo "=========================================="
echo "请从PC执行以下命令传输项目文件:"
echo ""
echo "scp -r yolo_process_detection/ jetson@<JETSON_IP>:~/yolo_process_detection/"
echo ""
echo "将 <JETSON_IP> 替换为你的Jetson IP地址"
echo ""

# 安装系统监控工具
echo ""
echo "=========================================="
echo "安装系统监控工具"
echo "=========================================="
sudo apt install -y htop sysstat

# 安装jtop (Jetson性能监控)
echo "安装jtop (Jetson性能监控工具)"
pip3 install jetson-stats

# 创建启动脚本
cat > ~/yolo_process_detection/start_detection.sh << 'EOF'
#!/bin/bash

# 设置最大性能模式
sudo nvpmodel -m 0
sudo jetson_clocks

# 进入项目目录
cd ~/yolo_process_detection

# 启动检测程序
python3 inference_jetson.py
EOF

chmod +x ~/yolo_process_detection/start_detection.sh

# 完成提示
echo ""
echo "=========================================="
echo "Jetson Nano 环境准备完成!"
echo "=========================================="
echo ""
echo "后续步骤:"
echo "1. 从PC传输项目文件到Jetson"
echo "2. 传输优化后的模型到Jetson"
echo "3. 运行模型优化 (如需要)"
echo "4. 启动检测程序: ./start_detection.sh"
echo ""
echo "监控工具:"
echo "  - GPU监控: sudo tegrastats"
echo "  - 系统监控: htop"
echo "  - Jetson监控: jtop"
echo ""
