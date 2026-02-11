"""测试摄像头视频流功能

使用Python最佳实践重构的版本，提供健壮的摄像头测试功能。
"""
from dataclasses import dataclass, field
from typing import NamedTuple, Iterator, Optional
from contextlib import contextmanager
import logging
import time
import cv2


logger = logging.getLogger(__name__)


@dataclass
class CameraInfo:
    """摄像头信息数据类"""
    index: int
    width: int
    height: int
    fps: float
    is_available: bool = True


@dataclass
class TestResult:
    """测试结果数据类"""
    camera_index: int
    success: bool
    frame_shape: Optional[tuple[int, int, int]] = None
    actual_fps: float = 0.0
    error_message: Optional[str] = None
    elapsed_time: float = 0.0


class CameraError(Exception):
    """摄像头错误基类"""
    pass


class CameraNotFoundError(CameraError):
    """摄像头未找到错误"""
    pass


class CameraReadError(CameraError):
    """摄像头读取错误"""
    pass


class CameraFrameError(CameraError):
    """摄像头帧错误"""
    pass


@contextmanager
def camera_context(camera_index: int):
    """摄像头上下文管理器
    
    自动管理cv2.VideoCapture资源。
    
    Args:
        camera_index: 摄像头索引
        
    Yields:
        cv2.VideoCapture: 摄像头对象
        
    Raises:
        CameraNotFoundError: 如果无法打开摄像头
    """
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        raise CameraNotFoundError(f"无法打开摄像头 {camera_index}")
    
    try:
        yield cap
    finally:
        cap.release()


def get_camera_info(cap: cv2.VideoCapture) -> CameraInfo:
    """获取摄像头信息
    
    Args:
        cap: 摄像头对象
        
    Returns:
        CameraInfo: 摄像头信息
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    
    return CameraInfo(
        index=int(cap.get(cv2.CAP_PROP_BACKEND)),
        width=width,
        height=height,
        fps=fps,
        is_available=True
    )


def read_test_frame(cap: cv2.VideoCapture) -> tuple[bool, Optional[cv2.typing.MatLike]]:
    """读取测试帧
    
    Args:
        cap: 摄像头对象
        
    Returns:
        tuple: (是否成功, 帧对象)
    """
    ret, frame = cap.read()
    return ret, frame if ret else None


def measure_fps(cap: cv2.VideoCapture, frames: int = 10) -> dict[str, float]:
    """测量实际FPS
    
    Args:
        cap: 摄像头对象
        frames: 测试帧数
        
    Returns:
        dict: 包含fps和耗时
    """
    success_count = 0
    start_time = time.perf_counter()
    
    for _ in range(frames):
        ret, _ = cap.read()
        if ret:
            success_count += 1
        time.sleep(0.033)
    
    elapsed = time.perf_counter() - start_time
    actual_fps = success_count / elapsed if elapsed > 0 else 0.0
    
    return {
        "fps": actual_fps,
        "elapsed": elapsed
    }


def test_camera(camera_index: int = 0) -> TestResult:
    """测试摄像头
    
    Args:
        camera_index: 摄像头索引
        
    Returns:
        TestResult: 测试结果
    """
    logger.info(f"Testing camera {camera_index}")
    
    start_time = time.perf_counter()
    
    try:
        with camera_context(camera_index) as cap:
            camera_info = get_camera_info(cap)
            
            logger.info(
                f"Camera {camera_index} opened: "
                f"{camera_info.width}x{camera_info.height} @ {camera_info.fps:.2f}fps"
            )
            
            ret, frame = read_test_frame(cap)
            
            if not ret or frame is None:
                raise CameraReadError("无法读取测试帧")
            
            logger.info(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
            
            fps_info = measure_fps(cap)
            
            elapsed = time.perf_counter() - start_time
            
            return TestResult(
                camera_index=camera_index,
                success=True,
                frame_shape=frame.shape,
                actual_fps=fps_info["fps"],
                elapsed_time=elapsed
            )
            
    except CameraError as e:
        logger.error(f"Camera test failed: {e}")
        elapsed = time.perf_counter() - start_time
        return TestResult(
            camera_index=camera_index,
            success=False,
            frame_shape=None,
            actual_fps=0.0,
            error_message=str(e),
            elapsed_time=elapsed
        )


def list_available_cameras(max_cameras: int = 5) -> list[CameraInfo]:
    """列出所有可用的摄像头
    
    Args:
        max_cameras: 最大检查数量
        
    Returns:
        list[CameraInfo]: 可用摄像头列表
    """
    logger.info(f"Scanning for available cameras (max {max_cameras})")
    
    available_cameras: list[CameraInfo] = []
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            try:
                camera_info = get_camera_info(cap)
                logger.info(
                    f"Found camera {i}: "
                    f"{camera_info.width}x{camera_info.height} @ {camera_info.fps:.2f}fps"
                )
                available_cameras.append(camera_info)
            finally:
                cap.release()
        else:
            logger.debug(f"Camera {i} not available")
    
    return available_cameras


def format_test_result(result: TestResult) -> str:
    """格式化测试结果
    
    Args:
        result: 测试结果对象
        
    Returns:
        str: 格式化的结果字符串
    """
    if result.success:
        return (
            f"Camera {result.camera_index}: SUCCESS\n"
            f"  Resolution: {result.frame_shape[1]}x{result.frame_shape[0] if result.frame_shape else 'N/A'}\n"
            f"  Actual FPS: {result.actual_fps:.2f}\n"
            f"  Elapsed: {result.elapsed_time:.2f}s"
        )
    else:
        return (
            f"Camera {result.camera_index}: FAILED\n"
            f"  Error: {result.error_message}\n"
            f"  Elapsed: {result.elapsed_time:.2f}s"
        )


def print_header(title: str, width: int = 60) -> None:
    """打印标题
    
    Args:
        title: 标题文本
        width: 标题宽度
    """
    print(f"\n{'=' * width}")
    print(f"{title.center(width)}")
    print(f"{'=' * width}")


def print_suggestions() -> None:
    """打印故障排除建议"""
    suggestions = [
        "1. 检查摄像头是否正确连接",
        "2. 检查摄像头是否被其他程序占用",
        "3. 检查摄像头驱动是否正确安装",
        "4. 如果是虚拟机，请检查USB设备是否已连接",
        "5. 尝试使用不同的摄像头索引",
        "6. 检查系统权限设置"
    ]
    
    print("\n可能的解决方案:")
    for suggestion in suggestions:
        print(f"  {suggestion}")


def main() -> None:
    """主函数"""
    print_header("摄像头视频流测试工具")
    
    available_cameras = list_available_cameras()
    
    if not available_cameras:
        logger.error("No available cameras found")
        print("\n❌ 没有找到可用的摄像头")
        print_suggestions()
        return
    
    logger.info(f"Found {len(available_cameras)} available camera(s)")
    print(f"\n找到 {len(available_cameras)} 个可用摄像头")
    
    camera_index = available_cameras[0].index
    result = test_camera(camera_index)
    
    print_header("测试结果")
    
    if result.success:
        print(f"\n✅ 所有测试通过！")
        print(f"\n您可以使用摄像头索引 {camera_index} 启动视频流")
        print(f"在Web界面中选择摄像头 {camera_index} 即可")
    else:
        print(f"\n❌ 测试失败")
        print_suggestions()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    main()
