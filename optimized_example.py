"""优化后的使用示例

展示如何使用优化后的检测系统，包括：
- 日志记录
- 错误处理
- 性能监控
- 内存管理
- 批处理
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List

# 导入优化后的模块
from utils import setup_logging, log_execution_time, handle_exceptions
from subpixel_detection import SubPixelDetector
from advanced_deployment import YOLODeployer, export_to_onnx
from performance_optimizer import (
    MemoryManager, BatchProcessor, ImageCache, 
    PerformanceMonitor, get_optimal_batch_size
)


def example_with_logging():
    """示例：使用日志记录"""
    print("\n" + "="*60)
    print("示例1: 使用日志记录")
    print("="*60)
    
    # 设置日志
    logger = setup_logging(log_dir="logs", log_level=20)  # INFO级别
    logger.info("开始处理任务")
    
    # 模拟处理
    logger.info("加载模型...")
    logger.info("模型加载完成")
    
    logger.warning("这是一个警告信息")
    logger.error("这是一个错误信息")
    
    print("日志文件已创建在 logs/ 目录")


@log_execution_time()
def example_with_performance_monitoring():
    """示例：使用性能监控"""
    print("\n" + "="*60)
    print("示例2: 使用性能监控")
    print("="*60)
    
    logger = setup_logging()
    monitor = PerformanceMonitor(log_interval=10, logger=logger)
    
    # 模拟处理
    import time
    for i in range(20):
        start = time.time()
        time.sleep(0.01)  # 模拟处理时间
        processing_time = time.time() - start
        monitor.record(processing_time)
    
    print("性能监控完成")


def example_with_memory_management():
    """示例：使用内存管理"""
    print("\n" + "="*60)
    print("示例3: 使用内存管理")
    print("="*60)
    
    logger = setup_logging()
    
    # 使用上下文管理器
    with MemoryManager(memory_threshold=80.0, logger=logger) as mm:
        # 获取初始内存状态
        stats = mm.get_memory_stats()
        print(f"初始内存使用: {stats.rss_mb:.1f}MB ({stats.percent:.1f}%)")
        
        # 创建一些大数组模拟内存使用
        arrays = []
        for i in range(5):
            arr = np.random.randn(1000, 1000)
            arrays.append(arr)
        
        # 检查内存
        if mm.check_memory():
            print("内存使用正常")
        else:
            print("内存使用过高！")
        
        # 手动优化内存
        del arrays
        mm.optimize_memory(force=True)
        
        final_stats = mm.get_memory_stats()
        print(f"最终内存使用: {final_stats.rss_mb:.1f}MB ({final_stats.percent:.1f}%)")
    
    print("内存管理完成")


def example_with_batch_processing():
    """示例：使用批处理"""
    print("\n" + "="*60)
    print("示例4: 使用批处理")
    print("="*60)
    
    logger = setup_logging()
    
    # 创建批处理器
    batch_processor = BatchProcessor(
        batch_size=4,
        max_batch_size=16,
        logger=logger
    )
    
    # 模拟图像列表
    images = [f"image_{i}.jpg" for i in range(20)]
    
    # 定义处理函数
    def process_batch(batch: List[str]):
        print(f"  处理批次，大小: {len(batch)}")
        return [f"processed_{img}" for img in batch]
    
    # 分批处理
    results = []
    for result in batch_processor.process_batches(images, process_batch):
        results.extend(result)
    
    print(f"批处理完成，共处理 {len(results)} 个图像")


def example_with_image_cache():
    """示例：使用图像缓存"""
    print("\n" + "="*60)
    print("示例5: 使用图像缓存")
    print("="*60)
    
    logger = setup_logging()
    
    # 创建图像缓存
    cache = ImageCache(max_size=10, max_memory_mb=100, logger=logger)
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 添加图像到缓存
    cache.put("image_1", test_image)
    cache.put("image_2", test_image)
    cache.put("image_3", test_image)
    
    # 获取缓存统计
    stats = cache.get_stats()
    print(f"缓存大小: {stats['size']}/{stats['max_size']}")
    print(f"内存使用: {stats['memory_usage_mb']:.1f}MB/{stats['max_memory_mb']:.1f}MB")
    
    # 从缓存获取图像
    cached_image = cache.get("image_1")
    if cached_image is not None:
        print(f"缓存命中！图像形状: {cached_image.shape}")
    
    # 清空缓存
    cache.clear()
    print("缓存已清空")


def example_with_subpixel_detection():
    """示例：使用亚像素级检测"""
    print("\n" + "="*60)
    print("示例6: 使用亚像素级检测")
    print("="*60)
    
    logger = setup_logging()
    
    # 注意：这里需要实际的模型文件
    model_path = "yolov8n.pt"  # 使用预训练模型
    
    try:
        # 创建检测器
        detector = SubPixelDetector(
            model_path=model_path,
            confidence_threshold=0.5,
            iou_threshold=0.45,
            logger=logger
        )
        
        # 创建测试图像（模拟缺陷）
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        # 添加模拟缺陷
        cv2.circle(test_image, (320, 240), 30, (50, 50, 50), -1)
        
        # 增强图像
        enhanced = detector.enhance_microscopic_image(test_image)
        if enhanced is not None:
            print("图像增强完成")
        
        # 检测
        detections, result_image = detector.detect_frame(test_image)
        print(f"检测到 {len(detections)} 个目标")
        
        for i, det in enumerate(detections):
            print(f"  {i+1}. {det.class_name}: 置信度={det.confidence:.3f}")
        
    except FileNotFoundError:
        print(f"模型文件不存在: {model_path}")
        print("请先下载模型或修改模型路径")
    except Exception as e:
        print(f"检测失败: {e}")


def example_with_deployment():
    """示例：使用部署器"""
    print("\n" + "="*60)
    print("示例7: 使用部署器")
    print("="*60)
    
    logger = setup_logging()
    
    model_path = "yolov8n.pt"
    
    try:
        # 创建Python原生部署器
        deployer = YOLODeployer(
            model_path=model_path,
            deploy_format="python",
            logger=logger
        )
        
        # 创建测试图像
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        # 执行检测
        result = deployer.detect(
            test_image,
            conf_threshold=0.5,
            iou_threshold=0.45
        )
        
        if result.success:
            print(f"检测成功！")
            print(f"  检测到 {len(result.detections)} 个目标")
            print(f"  推理时间: {result.inference_time*1000:.2f}ms")
            print(f"  FPS: {result.fps:.2f}")
        else:
            print(f"检测失败: {result.error_message}")
        
    except FileNotFoundError:
        print(f"模型文件不存在: {model_path}")
    except Exception as e:
        print(f"部署失败: {e}")


def example_complete_pipeline():
    """示例：完整处理流水线"""
    print("\n" + "="*60)
    print("示例8: 完整处理流水线")
    print("="*60)
    
    logger = setup_logging(log_dir="logs")
    
    # 1. 初始化内存管理
    with MemoryManager(logger=logger) as mm:
        logger.info("开始完整处理流水线")
        
        # 2. 创建图像缓存
        cache = ImageCache(max_size=50, logger=logger)
        
        # 3. 创建性能监控
        monitor = PerformanceMonitor(log_interval=5, logger=logger)
        
        # 4. 模拟处理多个图像
        image_paths = [f"image_{i}.jpg" for i in range(10)]
        
        for i, img_path in enumerate(image_paths):
            logger.info(f"处理图像 {i+1}/{len(image_paths)}: {img_path}")
            
            # 检查缓存
            cached = cache.get(img_path)
            if cached is not None:
                logger.info("  使用缓存图像")
                continue
            
            # 模拟处理
            import time
            start = time.time()
            time.sleep(0.05)  # 模拟处理时间
            
            # 创建模拟结果图像
            result_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # 添加到缓存
            cache.put(img_path, result_image)
            
            # 记录性能
            processing_time = time.time() - start
            monitor.record(processing_time)
            
            # 定期优化内存
            if i % 3 == 0:
                mm.optimize_memory()
        
        # 5. 获取缓存统计
        cache_stats = cache.get_stats()
        logger.info(f"缓存统计: {cache_stats}")
        
        logger.info("处理流水线完成")


def example_optimal_batch_size():
    """示例：计算最优批大小"""
    print("\n" + "="*60)
    print("示例9: 计算最优批大小")
    print("="*60)
    
    # 不同图像尺寸的最优批大小
    image_shapes = [
        (480, 640, 3),   # 标准分辨率
        (720, 1280, 3),  # HD
        (1080, 1920, 3), # Full HD
    ]
    
    target_memory = 1000.0  # 目标内存 1000MB
    
    for shape in image_shapes:
        batch_size = get_optimal_batch_size(shape, target_memory_mb=target_memory)
        print(f"图像尺寸 {shape}: 最优批大小 = {batch_size}")


def main():
    """主函数"""
    print("="*60)
    print("YOLO检测系统优化示例")
    print("="*60)
    
    # 运行所有示例
    examples = [
        ("日志记录", example_with_logging),
        ("性能监控", example_with_performance_monitoring),
        ("内存管理", example_with_memory_management),
        ("批处理", example_with_batch_processing),
        ("图像缓存", example_with_image_cache),
        ("亚像素级检测", example_with_subpixel_detection),
        ("部署器", example_with_deployment),
        ("完整流水线", example_complete_pipeline),
        ("最优批大小", example_optimal_batch_size),
    ]
    
    print("\n可用示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\n运行所有示例...\n")
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"示例 '{name}' 运行失败: {e}")
    
    print("\n" + "="*60)
    print("所有示例运行完成")
    print("="*60)


if __name__ == "__main__":
    main()