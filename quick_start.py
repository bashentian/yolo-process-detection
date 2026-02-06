import cv2
import sys
from pathlib import Path
from config import ProcessDetectionConfig
from detector import ProcessDetector


def quick_test():
    print("=== YOLO工序检测系统 - 快速测试 ===\n")
    
    config = ProcessDetectionConfig()
    detector = ProcessDetector(config)
    
    print("系统配置:")
    print(f"  模型: {config.MODEL_NAME}")
    print(f"  置信度阈值: {config.CONFIDENCE_THRESHOLD}")
    print(f"  检测类别: {list(config.CLASS_NAMES.values())}")
    print(f"  工序阶段: {config.PROCESS_STAGES}")
    print()
    
    print("检查摄像头...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头，尝试使用测试图像...")
        
        test_image_path = Path("data/test_image.jpg")
        if test_image_path.exists():
            image = cv2.imread(str(test_image_path))
            detections = detector.detect(image)
            stage = detector.analyze_process_stage(detections)
            
            print(f"\n测试图像检测结果:")
            print(f"  检测数量: {len(detections)}")
            for det in detections:
                print(f"  - {det.class_name}: {det.confidence:.2f}")
            print(f"  当前工序: {stage}")
            
            annotated = detector.draw_detections(image, detections, stage)
            cv2.imwrite("outputs/test_result.jpg", annotated)
            print(f"\n结果已保存到 outputs/test_result.jpg")
        else:
            print("未找到测试图像，请将测试图像放在 data/test_image.jpg")
            sys.exit(1)
    else:
        print("摄像头连接成功！")
        print("\n开始实时检测（按 'q' 退出）...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            detections, processed_frame = detector.detect_frame(frame)
            stage = detector.analyze_process_stage(detections)
            
            annotated = detector.draw_detections(processed_frame, detections, stage)
            
            display_frame = cv2.resize(annotated, config.DISPLAY_SIZE)
            cv2.imshow('YOLO工序检测 - 快速测试', display_frame)
            
            print(f"\r检测: {len(detections)} 个对象 | 工序: {stage}", end="")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    print("\n\n测试完成！")
    print("\n下一步:")
    print("  1. 准备你的数据集（视频或图像）")
    print("  2. 根据需要修改 config.py 中的配置")
    print("  3. 运行 'python main.py video <视频文件>' 处理视频")
    print("  4. 或运行 'python main.py webcam' 启动实时检测")
    print("  5. 运行 'python web_interface.py' 启动Web界面")


if __name__ == "__main__":
    quick_start()
