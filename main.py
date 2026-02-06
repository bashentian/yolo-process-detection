import argparse
from pathlib import Path
import cv2
from datetime import datetime

from config import ProcessDetectionConfig
from video_processor import VideoProcessor
from analyzer import ProcessAnalyzer


def process_video_file(input_path: str, output_path: str = None, 
                      show_display: bool = True):
    config = ProcessDetectionConfig()
    processor = VideoProcessor(config)
    analyzer = ProcessAnalyzer(config)
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/process_detection_{timestamp}.mp4"
    
    def process_callback(frame, detections, stage):
        analyzer.record_detections(detections, processor.frame_count, datetime.now())
        
        current_stage = analyzer.get_current_stage()
        if current_stage != stage:
            analyzer.record_stage_change(stage, datetime.now())
        
        if show_display:
            display_frame = cv2.resize(frame, config.DISPLAY_SIZE)
            cv2.imshow('Process Detection', display_frame)
        
        print(f"\rFrame: {processor.frame_count} | Stage: {stage} | "
              f"Detections: {len(detections)}", end="")
    
    print(f"Processing video: {input_path}")
    print(f"Output will be saved to: {output_path}")
    print("Press 'q' to quit")
    
    processor.process_video(input_path, output_path, process_callback)
    
    print("\n\nProcessing complete!")
    
    stats = analyzer.calculate_statistics()
    print("\nStatistics:")
    print(f"  Total frames: {stats['total_frames']}")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Average confidence: {stats['average_confidence']:.2f}")
    
    efficiency = analyzer.analyze_process_efficiency()
    print("\nEfficiency Analysis:")
    print(f"  Overall efficiency: {efficiency['efficiency']:.1f}%")
    print(f"  Bottleneck stage: {efficiency['bottleneck']}")
    print(f"  Active time: {efficiency['active_time']:.1f}s")
    print(f"  Idle time: {efficiency['idle_time']:.1f}s")
    
    anomalies = analyzer.detect_anomalies()
    if anomalies:
        print(f"\nFound {len(anomalies)} anomalies")
    
    output_json = output_path.replace('.mp4', '_analysis.json')
    analyzer.export_results(output_json)


def process_webcam(camera_index: int = 0):
    config = ProcessDetectionConfig()
    config.VIDEO_SOURCE = camera_index
    
    processor = VideoProcessor(config)
    analyzer = ProcessAnalyzer(config)
    
    def process_callback(frame, detections, stage):
        analyzer.record_detections(detections, processor.frame_count, datetime.now())
        
        current_stage = analyzer.get_current_stage()
        if current_stage != stage:
            analyzer.record_stage_change(stage, datetime.now())
        
        display_frame = cv2.resize(frame, config.DISPLAY_SIZE)
        
        stats_text = f"Stage: {stage} | Objects: {len(detections)}"
        cv2.putText(display_frame, stats_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Real-time Process Detection', display_frame)
    
    print(f"Starting webcam on camera {camera_index}")
    print("Press 'q' to quit")
    
    processor.process_video(camera_index, None, process_callback)
    
    print("\nReal-time detection stopped")


def analyze_video(input_path: str):
    config = ProcessDetectionConfig()
    processor = VideoProcessor(config)
    analyzer = ProcessAnalyzer(config)
    
    print(f"Analyzing video: {input_path}")
    
    processor.initialize_video_source(input_path)
    
    while True:
        ret, frame = processor.video_source.read()
        if not ret:
            break
        
        detections, _ = processor.detector.detect_frame(frame)
        stage = processor.detector.analyze_process_stage(detections)
        
        analyzer.record_detections(detections, processor.frame_count, datetime.now())
        current_stage = analyzer.get_current_stage()
        if current_stage != stage:
            analyzer.record_stage_change(stage, datetime.now())
        
        print(f"\rProcessing frame {processor.frame_count}: {stage}", end="")
    
    processor.release_resources()
    
    print("\n\nAnalysis complete!")
    
    stats = analyzer.calculate_statistics()
    print(f"\nTotal frames: {stats['total_frames']}")
    print(f"Total detections: {stats['total_detections']}")
    
    efficiency = analyzer.analyze_process_efficiency()
    print(f"\nEfficiency: {efficiency['efficiency']:.1f}%")
    print(f"Bottleneck: {efficiency['bottleneck']}")
    
    anomalies = analyzer.detect_anomalies()
    print(f"\nAnomalies found: {len(anomalies)}")
    
    output_path = f"outputs/analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    analyzer.export_results(output_path)


def main():
    parser = argparse.ArgumentParser(
        description='YOLO Process Detection System'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    video_parser = subparsers.add_parser('video', help='Process video file')
    video_parser.add_argument('input', help='Input video path')
    video_parser.add_argument('-o', '--output', help='Output video path')
    video_parser.add_argument('--no-display', action='store_true',
                            help='Disable display window')
    
    webcam_parser = subparsers.add_parser('webcam', help='Process webcam stream')
    webcam_parser.add_argument('-c', '--camera', type=int, default=0,
                              help='Camera index (default: 0)')
    
    analyze_parser = subparsers.add_parser('analyze', help='Analyze video without display')
    analyze_parser.add_argument('input', help='Input video path')
    
    args = parser.parse_args()
    
    if args.command == 'video':
        process_video_file(args.input, args.output, not args.no_display)
    elif args.command == 'webcam':
        process_webcam(args.camera)
    elif args.command == 'analyze':
        analyze_video(args.input)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
