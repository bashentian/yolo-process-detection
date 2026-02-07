import cv2
import numpy as np
from pathlib import Path
from typing import List
import argparse

from subpixel_detection import SubPixelDetector
from advanced_deployment import YOLODeployer, export_to_onnx, export_to_tensorrt, benchmark_all_formats


def example_subpixel_detection(model_path: str, image_path: str, output_path: str = None):
    print("\n" + "="*60)
    print("亚像素级检测示例")
    print("="*60)
    
    detector = SubPixelDetector(model_path, confidence_threshold=0.5, iou_threshold=0.45)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    enhanced_image = detector.enhance_microscopic_image(image)
    
    detections, result_frame = detector.detect_frame(enhanced_image)
    
    print(f"\n检测到 {len(detections)} 个目标:")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det.class_name}: 置信度={det.confidence:.3f}, "
              f"中心=({det.center[0]:.2f}, {det.center[1]:.2f}), 面积={det.area:.0f}")
    
    if output_path:
        cv2.imwrite(output_path, result_frame)
        print(f"\n结果已保存到: {output_path}")
    
    cv2.imshow("Subpixel Detection", result_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def example_defect_simulation(image_path: str, output_path: str = None):
    print("\n" + "="*60)
    print("微缺陷模拟示例")
    print("="*60)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    detector = SubPixelDetector("yolov8n.pt")
    
    simulated_image = detector.simulate_micro_defects(
        image, 
        num_defects=10,
        defect_size_range=(5, 20)
    )
    
    print(f"\n已模拟添加 {10} 个微米级缺陷")
    
    if output_path:
        cv2.imwrite(output_path, simulated_image)
        print(f"模拟图像已保存到: {output_path}")
    
    combined = np.hstack([image, simulated_image])
    cv2.imshow("Original vs Simulated Defects", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def example_onnx_deployment(model_path: str, image_path: str, output_path: str = None):
    print("\n" + "="*60)
    print("ONNX部署示例")
    print("="*60)
    
    onnx_path = export_to_onnx(model_path, None, input_size=640, simplify=True, opset=12)
    
    deployer = YOLODeployer(onnx_path, deploy_format="onnx")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    detections, result_image = deployer.detect(image, conf_threshold=0.5, iou_threshold=0.45)
    
    print(f"\nONNX推理完成:")
    print(f"  检测到 {len(detections)} 个目标")
    for i, det in enumerate(detections[:5]):
        print(f"  {i+1}. 类别ID={det['class_id']}, 置信度={det['confidence']:.3f}")
    
    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"\n结果已保存到: {output_path}")
    
    cv2.imshow("ONNX Deployment", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def example_tensorrt_deployment(model_path: str, image_path: str, output_path: str = None):
    print("\n" + "="*60)
    print("TensorRT部署示例")
    print("="*60)
    
    try:
        engine_path = export_to_tensorrt(model_path, None, input_size=640, fp16=True, workspace=4)
        
        deployer = YOLODeployer(engine_path, deploy_format="tensorrt")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return
        
        detections, result_image = deployer.detect(image, conf_threshold=0.5, iou_threshold=0.45)
        
        print(f"\nTensorRT推理完成:")
        print(f"  检测到 {len(detections)} 个目标")
        for i, det in enumerate(detections[:5]):
            print(f"  {i+1}. 类别ID={det['class_id']}, 置信度={det['confidence']:.3f}")
        
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"\n结果已保存到: {output_path}")
        
        cv2.imshow("TensorRT Deployment", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"\nTensorRT部署失败（需要CUDA环境和TensorRT）: {e}")
        print("提示: 请确保已安装TensorRT并配置CUDA环境")


def example_benchmark(model_path: str, test_images_dir: str, num_images: int = 10):
    print("\n" + "="*60)
    print("性能基准测试示例")
    print("="*60)
    
    test_images = []
    image_files = list(Path(test_images_dir).glob("*.jpg")) + list(Path(test_images_dir).glob("*.png"))
    
    for img_file in image_files[:num_images]:
        img = cv2.imread(str(img_file))
        if img is not None:
            test_images.append(img)
    
    if not test_images:
        print(f"在目录 {test_images_dir} 中未找到测试图像")
        return
    
    print(f"加载了 {len(test_images)} 张测试图像")
    
    results = benchmark_all_formats(model_path, test_images, input_size=640)
    
    print("\n" + "="*60)
    print("性能对比总结")
    print("="*60)
    for format_name, result in results.items():
        print(f"\n{format_name.upper()}:")
        print(f"  平均推理时间: {result['avg_inference_time_ms']:.2f} ms")
        print(f"  平均FPS: {result['avg_fps']:.2f}")
        print(f"  标准差: {result['std_inference_time_ms']:.2f} ms")
    
    if 'onnx' in results:
        print(f"\n加速比（相对ONNX）:")
        baseline_time = results['onnx']['avg_inference_time_ms']
        for format_name, result in results.items():
            if format_name != 'onnx':
                speedup = baseline_time / result['avg_inference_time_ms']
                print(f"  {format_name.upper()}: {speedup:.2f}x")


def example_microscopic_pipeline(model_path: str, image_path: str, output_path: str = None):
    print("\n" + "="*60)
    print("显微图像处理完整流水线")
    print("="*60)
    
    detector = SubPixelDetector(model_path)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    print("\n步骤1: 图像增强...")
    enhanced_image = detector.enhance_microscopic_image(image)
    
    print("步骤2: 亚像素级检测...")
    detections, result_frame = detector.detect_frame(enhanced_image)
    
    print(f"\n检测统计:")
    print(f"  总检测数: {len(detections)}")
    print(f"  平均置信度: {np.mean([d.confidence for d in detections]):.3f}")
    print(f"  平均面积: {np.mean([d.area for d in detections]):.0f} 像素²")
    
    class_counts = {}
    for det in detections:
        class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
    print(f"  类别分布: {class_counts}")
    
    print("\n步骤3: 创建可视化结果...")
    comparison = np.hstack([
        cv2.putText(image.copy(), "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2),
        cv2.putText(enhanced_image.copy(), "Enhanced", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2),
        cv2.putText(result_frame.copy(), "Detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    ])
    
    if output_path:
        cv2.imwrite(output_path, comparison)
        print(f"\n结果已保存到: {output_path}")
    
    cv2.imshow("Microscopic Pipeline", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="YOLO高级功能使用示例")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO模型路径")
    parser.add_argument("--image", type=str, required=True, help="测试图像路径")
    parser.add_argument("--output", type=str, help="输出图像路径")
    parser.add_argument("--test-dir", type=str, help="测试图像目录（用于基准测试）")
    
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    subparsers.add_parser("subpixel", help="亚像素检测")
    subparsers.add_parser("simulate", help="缺陷模拟")
    subparsers.add_parser("onnx", help="ONNX部署")
    subparsers.add_parser("tensorrt", help="TensorRT部署")
    subparsers.add_parser("benchmark", help="性能基准测试")
    subparsers.add_parser("pipeline", help="显微图像处理流水线")
    
    args = parser.parse_args()
    
    if args.command == "subpixel":
        example_subpixel_detection(args.model, args.image, args.output)
    elif args.command == "simulate":
        example_defect_simulation(args.image, args.output)
    elif args.command == "onnx":
        example_onnx_deployment(args.model, args.image, args.output)
    elif args.command == "tensorrt":
        example_tensorrt_deployment(args.model, args.image, args.output)
    elif args.command == "benchmark":
        if args.test_dir:
            example_benchmark(args.model, args.test_dir)
        else:
            print("基准测试需要指定 --test-dir 参数")
    elif args.command == "pipeline":
        example_microscopic_pipeline(args.model, args.image, args.output)
    else:
        print("请指定命令: subpixel, simulate, onnx, tensorrt, benchmark, pipeline")
        print("\n示例:")
        print("  python advanced_usage_example.py --image test.jpg subpixel")
        print("  python advanced_usage_example.py --image test.jpg onnx --output result.jpg")
        print("  python advanced_usage_example.py --model yolov8n.pt --test-dir ./images benchmark")


if __name__ == "__main__":
    main()