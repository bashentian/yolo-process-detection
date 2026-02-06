import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from detector import ProcessDetector
from tracker import ObjectTracker
from analyzer import ProcessAnalyzer
from video_processor import VideoProcessor
from config import ProcessDetectionConfig


def example_multi_camera_processing():
    print("=== 多摄像头并行处理 ===\n")
    
    camera_indices = [0, 1]
    
    processors = []
    analyzers = []
    
    for cam_idx in camera_indices:
        config = ProcessDetectionConfig()
        config.VIDEO_SOURCE = cam_idx
        
        processor = VideoProcessor(config)
        analyzer = ProcessAnalyzer(config)
        
        processors.append(processor)
        analyzers.append(analyzer)
    
    print(f"处理 {len(camera_indices)} 个摄像头...")
    
    try:
        while True:
            frames = []
            
            for i, processor in enumerate(processors):
                ret, frame = processor.video_source.read()
                if ret:
                    detections, processed_frame = processor.detector.detect_frame(frame)
                    stage = processor.detector.analyze_process_stage(detections)
                    
                    analyzers[i].record_detections(detections, processor.frame_count, datetime.now())
                    
                    annotated = processor.detector.draw_detections(
                        processed_frame, detections, f"Camera {i}: {stage}"
                    )
                    frames.append(annotated)
            
            if frames:
                combined = np.hstack(frames)
                combined = cv2.resize(combined, (1280, 360))
                cv2.imshow('Multi-Camera Detection', combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        for processor in processors:
            processor.release_resources()
        cv2.destroyAllWindows()


def example_efficiency_monitoring():
    print("=== 效率监控 ===\n")
    
    config = ProcessDetectionConfig()
    analyzer = ProcessAnalyzer(config)
    
    simulated_data = []
    stages = ["preparation", "processing", "assembly", "quality_check", "packaging"]
    
    for frame in range(100):
        stage = stages[frame % len(stages)]
        timestamp = datetime.now()
        
        stage_data = {
            "timestamp": timestamp.isoformat(),
            "stage": stage,
            "frame": frame
        }
        simulated_data.append(stage_data)
        
        analyzer.record_stage_change(stage, timestamp)
    
    efficiency = analyzer.analyze_process_efficiency()
    
    print("效率分析结果:")
    print(f"  整体效率: {efficiency['efficiency']:.1f}%")
    print(f"  瓶颈工序: {efficiency['bottleneck']}")
    print(f"  总时间: {efficiency['total_duration']:.1f}s")
    print(f"  工作时间: {efficiency['active_time']:.1f}s")
    print(f"  空闲时间: {efficiency['idle_time']:.1f}s")
    
    print("\n各工序占比:")
    for stage, percentage in efficiency['stage_percentages'].items():
        print(f"  {stage}: {percentage:.1f}%")


def example_anomaly_detection():
    print("=== 异常检测 ===\n")
    
    config = ProcessDetectionConfig()
    analyzer = ProcessAnalyzer(config)
    
    normal_detection_counts = [5, 6, 5, 7, 5, 6, 5, 5, 6, 5]
    anomaly_detection_counts = [20, 5, 5, 25, 5, 5, 30, 5, 5, 5]
    
    all_counts = normal_detection_counts + anomaly_detection_counts
    
    for frame_idx, count in enumerate(all_counts):
        detection_data = {
            "frame_number": frame_idx,
            "timestamp": datetime.now().isoformat(),
            "detections": [{"class_name": "object"} for _ in range(count)]
        }
        analyzer.detection_history.append(detection_data)
    
    anomalies = analyzer.detect_anomalies(threshold_std=2.0)
    
    print(f"检测到 {len(anomalies)} 个异常:")
    for anomaly in anomalies:
        print(f"  帧 {anomaly['frame_number']}: 预期 {anomaly['expected']:.1f}, "
              f"实际 {anomaly['actual']} (偏差: {anomaly['deviation']:+.1f})")


def example_trajectory_analysis():
    print("=== 轨迹分析 ===\n")
    
    config = ProcessDetectionConfig()
    processor = VideoProcessor(config)
    
    input_video = "data/sample_video.mp4"
    processor.initialize_video_source(input_video)
    
    frame_count = 0
    
    while True:
        ret, frame = processor.video_source.read()
        if not ret:
            break
        
        detections, _ = processor.detector.detect_frame(frame)
        tracked_objects = processor.tracker.update(detections)
        
        for track in tracked_objects:
            x, y = track.detection.center
            processor.analyzer.record_trajectory(track.track_id, (x, y))
        
        frame_count += 1
        
        if frame_count > 100:
            break
    
    processor.release_resources()
    
    obj_stats = processor.analyzer.get_object_statistics()
    
    print("轨迹分析结果:")
    for track_id, stats in obj_stats.items():
        if stats['trajectory_points'] > 10:
            print(f"  对象 ID {track_id}:")
            print(f"    轨迹点数: {stats['trajectory_points']}")
            print(f"    移动距离: {stats['distance_traveled']:.1f} 像素")


def example_real_time_alerts():
    print("=== 实时告警 ===\n")
    
    config = ProcessDetectionConfig()
    processor = VideoProcessor(config)
    
    alert_conditions = {
        "max_idle_time": 30,
        "min_workers": 1,
        "max_workers": 5
    }
    
    idle_counter = 0
    last_stage = None
    
    def process_callback(frame, detections, stage):
        nonlocal idle_counter, last_stage
        
        worker_count = sum(1 for d in detections if d.class_name == "worker")
        
        if stage == "idle":
            idle_counter += 1
        else:
            idle_counter = 0
        
        alerts = []
        
        if idle_counter > alert_conditions["max_idle_time"]:
            alerts.append(f"警告: 设备空闲 {idle_counter} 帧")
        
        if worker_count < alert_conditions["min_workers"]:
            alerts.append(f"警告: 工人不足 ({worker_count})")
        
        if worker_count > alert_conditions["max_workers"]:
            alerts.append(f"警告: 人员过多 ({worker_count})")
        
        if stage != last_stage:
            print(f"\n工序变更: {last_stage} -> {stage}")
            last_stage = stage
        
        for alert in alerts:
            print(f"  {alert}")
        
        annotated = processor.detector.draw_detections(frame, detections, stage)
        
        for alert in alerts:
            cv2.putText(annotated, alert[:50], (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Real-time Alerts', cv2.resize(annotated, config.DISPLAY_SIZE))
    
    print("启动实时监控（按 'q' 退出）...")
    processor.process_video(0, None, process_callback)


def example_process_flow_analysis():
    print("=== 工序流程分析 ===\n")
    
    config = ProcessDetectionConfig()
    analyzer = ProcessAnalyzer(config)
    
    stages = ["preparation", "processing", "processing", "assembly", 
              "quality_check", "packaging", "idle", "preparation"]
    
    timestamps = []
    for i in range(8):
        timestamp = datetime.now()
        timestamps.append(timestamp)
        analyzer.record_stage_change(stages[i], timestamp)
    
    flow = analyzer.visualize_process_flow()
    
    print("工序流转分析:")
    print(f"  转换次数: {flow['transition_count']}")
    
    print("\n转换序列:")
    for transition in flow['transitions']:
        print(f"  {transition['from']} -> {transition['to']} "
              f"({transition['timestamp']})")


def example_batch_processing_with_stats():
    print("=== 批量处理与统计 ===\n")
    
    config = ProcessDetectionConfig()
    detector = ProcessDetector(config)
    
    input_dir = Path("data/batch_images")
    output_dir = Path("outputs/batch_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    
    all_detections = []
    all_stages = []
    
    print(f"处理 {len(image_files)} 张图像...")
    
    for i, img_path in enumerate(image_files):
        image = cv2.imread(str(img_path))
        detections = detector.detect(image)
        stage = detector.analyze_process_stage(detections)
        
        all_detections.extend(detections)
        all_stages.append(stage)
        
        annotated = detector.draw_detections(image, detections, stage)
        cv2.imwrite(str(output_dir / f"result_{img_path.name}"), annotated)
        
        print(f"\r处理进度: {i+1}/{len(image_files)}", end="")
    
    print("\n\n批量处理完成！")
    
    class_counts = {}
    for det in all_detections:
        class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
    
    stage_counts = {}
    for stage in all_stages:
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    
    print("\n检测结果统计:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    
    print("\n工序分布:")
    for stage, count in stage_counts.items():
        print(f"  {stage}: {count} ({count/len(all_stages)*100:.1f}%)")


def example_custom_process_logic():
    print("=== 自定义工序逻辑 ===\n")
    
    from detector import ProcessDetector
    
    class CustomProcessDetector(ProcessDetector):
        def analyze_process_stage(self, detections):
            class_counts = {}
            for det in detections:
                class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
            
            if class_counts.get("worker", 0) >= 2 and class_counts.get("machine", 0) >= 1:
                if class_counts.get("product", 0) >= 2:
                    return "high_throughput"
                else:
                    return "setup"
            
            if class_counts.get("tool", 0) >= 3:
                return "maintenance"
            
            if class_counts.get("worker", 0) == 0:
                return "automated"
            
            return "unknown"
    
    config = ProcessDetectionConfig()
    detector = CustomProcessDetector(config)
    
    test_scenarios = [
        [{"class_name": "worker"}, {"class_name": "worker"}, 
         {"class_name": "machine"}, {"class_name": "product"}, {"class_name": "product"}],
        [{"class_name": "worker"}, {"class_name": "tool"}, 
         {"class_name": "tool"}, {"class_name": "tool"}],
        [{"class_name": "machine"}, {"class_name": "product"}]
    ]
    
    print("自定义工序逻辑测试:")
    for i, scenario in enumerate(test_scenarios):
        stage = detector.analyze_process_stage(scenario)
        classes = [s["class_name"] for s in scenario]
        print(f"  场景 {i+1}: {', '.join(classes)} -> {stage}")


if __name__ == "__main__":
    import sys
    
    examples = {
        "multi": example_multi_camera_processing,
        "efficiency": example_efficiency_monitoring,
        "anomaly": example_anomaly_detection,
        "trajectory": example_trajectory_analysis,
        "alerts": example_real_time_alerts,
        "flow": example_process_flow_analysis,
        "batch": example_batch_processing_with_stats,
        "custom": example_custom_process_logic
    }
    
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        if example_name in examples:
            examples[example_name]()
        else:
            print(f"Unknown example: {example_name}")
            print(f"Available examples: {', '.join(examples.keys())}")
    else:
        print("Available advanced examples:")
        for name in examples.keys():
            print(f"  python advanced_examples.py {name}")
