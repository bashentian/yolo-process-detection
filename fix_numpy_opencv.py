"""
修复 Numpy 和 OpenCV 兼容性问题的脚本
"""
import subprocess
import sys


def fix_numpy_opencv():
    """修复 Numpy 和 OpenCV 的兼容性问题"""
    print("开始修复 Numpy 和 OpenCV 兼容性问题...")
    
    # 1. 先卸载可能存在冲突的包
    print("\n1. 卸载冲突的包...")
    packages_to_uninstall = [
        "opencv-python",
        "opencv-python-headless",
        "numpy"
    ]
    
    for package in packages_to_uninstall:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package])
            print(f"   ✓ 已卸载 {package}")
        except Exception as e:
            print(f"   - {package} 可能未安装，跳过")
    
    # 2. 清理 pip 缓存
    print("\n2. 清理 pip 缓存...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "cache", "purge"])
        print("   ✓ 缓存已清理")
    except Exception:
        print("   - 缓存清理跳过")
    
    # 3. 安装兼容版本的 numpy
    print("\n3. 安装兼容版本的 numpy...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "numpy==1.26.4", "--force-reinstall", "--no-cache-dir"
    ])
    print("   ✓ numpy 1.26.4 安装完成")
    
    # 4. 安装兼容版本的 opencv-python
    print("\n4. 安装兼容版本的 opencv-python...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "opencv-python==4.8.1.78", "--force-reinstall", "--no-cache-dir"
    ])
    print("   ✓ opencv-python 4.8.1.78 安装完成")
    
    # 5. 验证安装
    print("\n5. 验证安装...")
    try:
        import numpy as np
        import cv2
        print(f"   ✓ NumPy 版本: {np.__version__}")
        print(f"   ✓ OpenCV 版本: {cv2.__version__}")
        
        # 测试基本功能
        test_array = np.array([1, 2, 3])
        print(f"   ✓ NumPy 数组创建成功")
        
        # 测试 OpenCV 的 numpy 功能
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', test_img)
        print(f"   ✓ OpenCV 图像编码成功")
        
        print("\n✅ 修复完成！所有测试通过。")
        return True
        
    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        return False


if __name__ == "__main__":
    success = fix_numpy_opencv()
    sys.exit(0 if success else 1)
