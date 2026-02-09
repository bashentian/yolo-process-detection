"""优化后的主程序

使用统一的性能监控、错误处理和配置验证功能。
"""

import asyncio
import argparse
import logging
from pathlib import Path
from typing import Optional

from config import ProcessDetectionConfig, get_config
from config_validator import validate_detection_config
from error_handler import (
    default_error_handler,
    VideoProcessingError,
    DetectionError,
    catch_errors,
    validate_file_path
)
from performance_monitor import default_performance_monitor
from utils import setup_logging, ProgressTracker
from detector import ProcessDetector
from video_processor import VideoProcessor
from analyzer import ProcessAnalyzer
import web_app
import api


@catch_errors("程序启动", None)
async def main():
    """主程序入口"""
    # 设置日志
    logger = setup_logging()
    logger.info("启动优化后的YOLO检测系统")

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="YOLO工序检测系统")
    parser.add_argument("mode", choices=["video", "webcam", "web", "api"],
                       help="运行模式")
    parser.add_argument("--config", type=str, default="production",
                       help="配置环境")
    parser.add_argument("--input", type=str,
                       help="输入文件路径（video模式）")
    parser.add_argument("--output", type=str,
                       help="输出文件路径")
    parser.add_argument("--device", type=str, default=None,
                       help="运行设备 (cpu/cuda)")
    parser.add_argument("--gpu", action="store_true",
                       help="强制使用GPU")

    args = parser.parse_args()

    # 获取配置并验证
    config = get_config(args.config)
    if args.device:
        config.DEVICE = args.device
    if args.gpu and config.DEVICE == "cpu":
        config.DEVICE = "cuda"

    # 验证配置
    validate_detection_config(config)

    # 验证模型文件
    if not validate_file_path(
        str(Path(__file__).parent.parent / "models" / config.MODEL_NAME),
        must_exist=True
    ):
        raise DetectionError(f"模型文件不存在: {config.MODEL_NAME}")

    logger.info(f"使用配置: {args.config}, 设备: {config.DEVICE}")

    # 初始化组件
    with default_performance_monitor.sync_measure("初始化组件"):
        detector = ProcessDetector(config)
        processor = VideoProcessor(config)
        analyzer = ProcessAnalyzer(config)

    if args.mode == "video":
        await process_video_mode(args, config, detector, processor, analyzer, logger)
    elif args.mode == "webcam":
        await process_webcam_mode(config, detector, processor, analyzer, logger)
    elif args.mode == "web":
        start_web_app(config, logger)
    elif args.mode == "api":
        start_api_server(config, logger)
    else:
        logger.error(f"未知的运行模式: {args.mode}")


async def process_video_mode(
    args,
    config,
    detector: ProcessDetector,
    processor: VideoProcessor,
    analyzer: ProcessAnalyzer,
    logger: logging.Logger
):
    """视频处理模式"""
    if not args.input:
        raise VideoProcessingError("视频模式需要指定输入文件 (--input)")

    if not validate_file_path(args.input, must_exist=True, extensions=['.mp4', '.avi', '.mov']):
        raise VideoProcessingError(f"无效的视频文件: {args.input}")

    logger.info(f"开始处理视频: {args.input}")

    # 设置输出路径
    output_path = args.output or f"outputs/processed_{Path(args.input).stem}.mp4"

    # 处理视频
    async def process_callback(frame, detections, stage):
        # 记录检测结果
        analyzer.record_detections(detections, processor.frame_count, None)

        # 更新工序阶段
        current_stage = analyzer.get_current_stage()
        if current_stage != stage:
            analyzer.record_stage_change(stage, None)

    # 进度跟踪
    progress = ProgressTracker(
        total=100,
        description="视频处理进度",
        logger=logger
    )

    try:
        # 处理视频（在单独的线程中）
        process_task = asyncio.to_thread(
            processor.process_video,
            args.input,
            output_path,
            process_callback
        )

        # 模拟进度更新
        while not processor.is_complete:
            progress.update(1)
            await asyncio.sleep(0.5)

        # 计算统计信息
        stats = analyzer.calculate_statistics()
        logger.info(f"视频处理完成，检测到 {stats.get('total_detections', 0)} 个目标")

        # 导出结果
        output_dir = Path(output_path).parent
        analyzer.export_results(str(output_dir / "analysis_results.json"))

    except Exception as e:
        default_error_handler.handle(
            e,
            "视频处理失败",
            severity=default_error_handler.ERROR
        )
        raise VideoProcessingError(f"视频处理失败: {e}")


async def process_webcam_mode(
    config,
    detector: ProcessDetector,
    processor: VideoProcessor,
    analyzer: ProcessAnalyzer,
    logger: logging.Logger
):
    """摄像头模式"""
    logger.info("启动摄像头检测模式")

    # 启动摄像头流管理器
    from web_app import camera_manager
    camera_manager.start(0, use_simulation=False)

    try:
        while True:
            frame = camera_manager.get_frame()
            if frame is not None:
                # 检测帧
                detections, processed_frame = detector.detect_frame(frame)
                stage = detector.analyze_process_stage(detections)

                # 记录数据
                analyzer.record_detections(detections, 0, None)

                # 显示结果
                annotated_frame = detector.draw_detections(
                    processed_frame, detections, stage
                )

                # 可以在这里添加更多显示逻辑

            await asyncio.sleep(0.033)  # 约30fps

    except KeyboardInterrupt:
        logger.info("用户中断程序")
    finally:
        camera_manager.stop()


def start_web_app(config, logger: logging.Logger):
    """启动Web应用"""
    logger.info("启动Web应用 (http://localhost:5000)")

    # 验证必要的目录
    for dir_name in ["outputs", "uploads", "static"]:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)

    web_app.app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )


def start_api_server(config, logger: logging.Logger):
    """启动API服务器"""
    logger.info("启动API服务器 (http://localhost:8000)")

    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


async def cleanup():
    """清理资源"""
    logger = logging.getLogger(__name__)
    logger.info("正在清理资源...")

    # 停止性能监控
    default_performance_monitor.disable()

    # 清理错误计数
    default_error_handler.clear_error_counts()

    logger.info("资源清理完成")


if __name__ == "__main__":
    try:
        # 运行主程序
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.critical(f"程序异常退出: {e}", exc_info=True)
        exit(1)
    finally:
        # 确保清理
        asyncio.run(cleanup())