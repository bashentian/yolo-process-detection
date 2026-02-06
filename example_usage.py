from video_processor import VideoProcessor
from analyzer import ProcessAnalyzer
from config import ProcessDetectionConfig
from datetime import datetime


def example_basic_detection():
    print("=== Basic Detection Example ===\n")
    
    config = ProcessDetectionConfig()
    detector = ProcessDetector(config)
    
    import cv2
    
    frame = cv2.imread("data/test_image.jpg")
    if frame is None:
        print("Please place a test image at data/test_image.jpg")
        return
    
    detections = detector.detect(frame)
    
    print(f"Found {len(detections)} detections:")
    for det in detections:
        print(f"  - {det.class_name}: {det.confidence:.2f} at {det.bbox}")


def example_video_processing():
    print("=== Video Processing Example ===\n")
    
    config = ProcessDetectionConfig()
    processor = VideoProcessor(config)
    analyzer = ProcessAnalyzer(config)
    
    input_video = "data/sample_video.mp4"
    
    def process_callback(frame, detections, stage):
        print(f"\rFrame: {processor.frame_count} | Stage: {stage} | Detections: {len(detections)}", end="")
        
        analyzer.record_detections(detections, processor.frame_count, datetime.now())
        current_stage = analyzer.get_current_stage()
        if current_stage != stage:
            analyzer.record_stage_change(stage, datetime.now())
    
    output_video = "outputs/result.mp4"
    processor.process_video(input_video, output_video, process_callback)
    
    print("\n\nAnalysis Results:")
    stats = analyzer.calculate_statistics()
    print(f"Total frames: {stats['total_frames']}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Average confidence: {stats['average_confidence']:.2f}")
    
    efficiency = analyzer.analyze_process_efficiency()
    print(f"\nEfficiency: {efficiency['efficiency']:.1f}%")
    print(f"Bottleneck: {efficiency['bottleneck']}")
    
    analyzer.export_results("outputs/analysis.json")


def example_real_time_detection():
    print("=== Real-time Detection Example ===\n")
    
    config = ProcessDetectionConfig()
    config.VIDEO_SOURCE = 0
    
    processor = VideoProcessor(config)
    
    print("Starting real-time detection from webcam...")
    print("Press 'q' to quit\n")
    
    processor.start_processing(0)


def example_custom_config():
    print("=== Custom Configuration Example ===\n")
    
    from config import ProcessDetectionConfig
    
    config = ProcessDetectionConfig()
    config.MODEL_NAME = "yolov8s.pt"
    config.CONFIDENCE_THRESHOLD = 0.7
    config.IOU_THRESHOLD = 0.5
    
    config.CLASS_NAMES = {
        0: "worker",
        1: "machine",
        2: "product",
        3: "tool",
        4: "material",
        5: "conveyor_belt"
    }
    
    print(f"Using model: {config.MODEL_NAME}")
    print(f"Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
    print(f"Classes: {list(config.CLASS_NAMES.values())}")


def example_batch_processing():
    print("=== Batch Processing Example ===\n")
    
    from pathlib import Path
    import cv2
    
    config = ProcessDetectionConfig()
    detector = ProcessDetector(config)
    
    image_dir = Path("data/images")
    output_dir = Path("outputs/annotated")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    print(f"Processing {len(image_files)} images...")
    
    for i, img_path in enumerate(image_files):
        frame = cv2.imread(str(img_path))
        detections = detector.detect(frame)
        stage = detector.analyze_process_stage(detections)
        
        annotated = detector.draw_detections(frame, detections, stage)
        
        output_path = output_dir / f"annotated_{img_path.name}"
        cv2.imwrite(str(output_path), annotated)
        
        print(f"\rProcessed {i+1}/{len(image_files)}", end="")
    
    print("\n\nBatch processing complete!")


def example_web_api():
    print("=== Web API Example ===\n")
    
    print("To start the web interface, run:")
    print("  python web_interface.py")
    print("\nThen open your browser to: http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  POST /api/upload_video - Upload a video file")
    print("  POST /api/process_video - Process uploaded video")
    print("  GET  /api/statistics - Get detection statistics")
    print("  GET  /api/efficiency - Get efficiency analysis")
    print("  GET  /api/timeline - Get process timeline")
    print("  GET  /api/stream_video - Stream live video")
    print("  POST /api/export_results - Export analysis results")


def example_data_preparation():
    print("=== Data Preparation Example ===\n")
    
    from data_utils import DatasetPreparer, AnnotationTool
    
    print("1. Prepare dataset structure:")
    preparer = DatasetPreparer("data/my_dataset")
    preparer.create_directories()
    
    print("\n2. Extract frames from video:")
    preparer.extract_frames_from_video(
        "data/source_video.mp4",
        "data/my_dataset/images",
        interval=30
    )
    
    print("\n3. Annotate images:")
    class_names = ["worker", "machine", "product", "tool", "material"]
    annotator = AnnotationTool("data/my_dataset/images", class_names)
    print("\nStarting annotation tool...")
    print("(This will open an interactive window)")
    annotator.annotate_images()
    
    print("\n4. Split dataset into train/val:")
    preparer.split_dataset(ratio=0.8)


if __name__ == "__main__":
    import sys
    
    examples = {
        "basic": example_basic_detection,
        "video": example_video_processing,
        "realtime": example_real_time_detection,
        "config": example_custom_config,
        "batch": example_batch_processing,
        "web": example_web_api,
        "data": example_data_preparation
    }
    
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        if example_name in examples:
            examples[example_name]()
        else:
            print(f"Unknown example: {example_name}")
            print(f"Available examples: {', '.join(examples.keys())}")
    else:
        print("Available examples:")
        for name in examples.keys():
            print(f"  python example_usage.py {name}")
