"""测试摄像头视频流功能

用于测试摄像头是否能正常启动和视频流是否能正常传输
"""

import cv2
import time
import sys


def test_camera(camera_index=0):
    """测试摄像头"""
    print(f"\n{'='*60}")
    print(f"测试摄像头 {camera_index}")
    print(f"{'='*60}\n")
    
    # 打开摄像头
    print(f"正在打开摄像头 {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 {camera_index}")
        return False
    
    print(f"✅ 摄像头 {camera_index} 打开成功")
    
    # 获取摄像头参数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\n摄像头参数:")
    print(f"  分辨率: {width}x{height}")
    print(f"  FPS: {fps}")
    
    # 读取测试帧
    print(f"\n正在读取测试帧...")
    ret, frame = cap.read()
    
    if not ret:
        print(f"❌ 无法读取帧")
        cap.release()
        return False
    
    print(f"✅ 成功读取帧")
    print(f"  帧形状: {frame.shape}")
    print(f"  数据类型: {frame.dtype}")
    
    # 测试连续读取
    print(f"\n测试连续读取 (10帧)...")
    success_count = 0
    start_time = time.time()
    
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            success_count += 1
        time.sleep(0.033)  # 约30fps
    
    elapsed = time.time() - start_time
    actual_fps = success_count / elapsed if elapsed > 0 else 0
    
    print(f"  成功读取: {success_count}/10 帧")
    print(f"  实际FPS: {actual_fps:.2f}")
    
    # 释放摄像头
    cap.release()
    print(f"\n✅ 摄像头测试完成")
    
    return True


def list_available_cameras(max_cameras=5):
    """列出所有可用的摄像头"""
    print(f"\n{'='*60}")
    print(f"扫描可用摄像头")
    print(f"{'='*60}\n")
    
    available_cameras = []
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"✅ 摄像头 {i}: {width}x{height}")
            available_cameras.append(i)
            cap.release()
        else:
            print(f"❌ 摄像头 {i}: 不可用")
    
    return available_cameras


def main():
    """主函数"""
    print("摄像头视频流测试工具")
    print("="*60)
    
    # 列出可用摄像头
    available_cameras = list_available_cameras()
    
    if not available_cameras:
        print("\n❌ 没有找到可用的摄像头")
        print("\n可能的解决方案:")
        print("  1. 检查摄像头是否正确连接")
        print("  2. 检查摄像头是否被其他程序占用")
        print("  3. 检查摄像头驱动是否正确安装")
        print("  4. 如果是虚拟机，请检查USB设备是否已连接")
        sys.exit(1)
    
    print(f"\n找到 {len(available_cameras)} 个可用摄像头")
    
    # 测试第一个可用摄像头
    camera_index = available_cameras[0]
    
    if test_camera(camera_index):
        print(f"\n{'='*60}")
        print(f"✅ 所有测试通过！")
        print(f"{'='*60}")
        print(f"\n您可以使用摄像头索引 {camera_index} 启动视频流")
        print(f"在Web界面中选择摄像头 {camera_index} 即可")
    else:
        print(f"\n{'='*60}")
        print(f"❌ 测试失败")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
