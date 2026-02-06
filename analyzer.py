import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict, Counter

from detector import Detection, ProcessDetector
from tracker import TrackedObject
from config import ProcessDetectionConfig


class ProcessAnalyzer:
    def __init__(self, config: ProcessDetectionConfig):
        self.config = config
        self.detection_history: List[Dict] = []
        self.stage_history: List[Dict] = []
        self.object_trajectories: Dict[int, List[tuple]] = defaultdict(list)
        self.stage_durations: Dict[str, float] = defaultdict(float)
        self.current_stage_start: Optional[datetime] = None
        
    def record_detections(self, detections: List[Detection], frame_number: int,
                         timestamp: datetime):
        detection_data = {
            "frame_number": frame_number,
            "timestamp": timestamp.isoformat(),
            "detections": [
                {
                    "bbox": det.bbox,
                    "confidence": det.confidence,
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "center": det.center,
                    "area": det.area
                }
                for det in detections
            ]
        }
        self.detection_history.append(detection_data)
    
    def record_stage_change(self, stage: str, timestamp: datetime):
        if self.current_stage_start:
            duration = (timestamp - self.current_stage_start).total_seconds()
            self.stage_durations[stage] += duration
        
        stage_data = {
            "timestamp": timestamp.isoformat(),
            "stage": stage,
            "duration_since_last_change": duration if self.current_stage_start else 0
        }
        self.stage_history.append(stage_data)
        
        self.current_stage_start = timestamp
    
    def record_trajectory(self, track_id: int, position: tuple):
        self.object_trajectories[track_id].append(position)
    
    def calculate_statistics(self) -> Dict:
        total_detections = sum(len(d["detections"]) for d in self.detection_history)
        
        class_counts = Counter()
        for data in self.detection_history:
            for det in data["detections"]:
                class_counts[det["class_name"]] += 1
        
        stage_counts = Counter(s["stage"] for s in self.stage_history)
        
        avg_confidence = []
        for data in self.detection_history:
            for det in data["detections"]:
                avg_confidence.append(det["confidence"])
        
        avg_confidence = sum(avg_confidence) / len(avg_confidence) if avg_confidence else 0
        
        return {
            "total_frames": len(self.detection_history),
            "total_detections": total_detections,
            "detections_per_frame": total_detections / len(self.detection_history) if self.detection_history else 0,
            "class_distribution": dict(class_counts),
            "stage_distribution": dict(stage_counts),
            "stage_durations": dict(self.stage_durations),
            "average_confidence": avg_confidence,
            "tracked_objects": len(self.object_trajectories)
        }
    
    def generate_timeline(self) -> List[Dict]:
        timeline = []
        
        for i, stage_data in enumerate(self.stage_history):
            timeline.append({
                "timestamp": stage_data["timestamp"],
                "stage": stage_data["stage"],
                "frame_count": sum(
                    1 for d in self.detection_history 
                    if d["timestamp"] <= stage_data["timestamp"]
                )
            })
        
        return timeline
    
    def analyze_process_efficiency(self) -> Dict:
        total_duration = sum(self.stage_durations.values())
        
        if total_duration == 0:
            return {"efficiency": 0, "bottleneck": "unknown"}
        
        stage_percentages = {
            stage: (duration / total_duration) * 100
            for stage, duration in self.stage_durations.items()
        }
        
        bottleneck = max(stage_percentages.items(), key=lambda x: x[1])[0]
        
        active_time = sum(
            duration for stage, duration in self.stage_durations.items()
            if stage != "idle"
        )
        efficiency = (active_time / total_duration) * 100 if total_duration > 0 else 0
        
        return {
            "efficiency": efficiency,
            "bottleneck": bottleneck,
            "stage_percentages": stage_percentages,
            "total_duration": total_duration,
            "active_time": active_time,
            "idle_time": self.stage_durations.get("idle", 0)
        }
    
    def detect_anomalies(self, threshold_std: float = 2.0) -> List[Dict]:
        anomalies = []
        
        detection_counts = [len(d["detections"]) for d in self.detection_history]
        if detection_counts:
            mean_count = sum(detection_counts) / len(detection_counts)
            std_count = (sum((x - mean_count) ** 2 for x in detection_counts) / len(detection_counts)) ** 0.5
            
            for i, data in enumerate(self.detection_history):
                count = len(data["detections"])
                if abs(count - mean_count) > threshold_std * std_count:
                    anomalies.append({
                        "frame_number": data["frame_number"],
                        "timestamp": data["timestamp"],
                        "type": "detection_count_anomaly",
                        "expected": mean_count,
                        "actual": count,
                        "deviation": count - mean_count
                    })
        
        return anomalies
    
    def export_results(self, output_path: str):
        results = {
            "statistics": self.calculate_statistics(),
            "timeline": self.generate_timeline(),
            "efficiency_analysis": self.analyze_process_efficiency(),
            "anomalies": self.detect_anomalies(),
            "export_timestamp": datetime.now().isoformat()
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results exported to {output_path}")
    
    def visualize_process_flow(self) -> Dict:
        flow = []
        
        for i in range(len(self.stage_history) - 1):
            current_stage = self.stage_history[i]["stage"]
            next_stage = self.stage_history[i + 1]["stage"]
            
            if current_stage != next_stage:
                flow.append({
                    "from": current_stage,
                    "to": next_stage,
                    "timestamp": self.stage_history[i]["timestamp"]
                })
        
        return {
            "transitions": flow,
            "transition_count": len(flow)
        }
    
    def get_object_statistics(self) -> Dict:
        obj_stats = defaultdict(lambda: {
            "class_name": "",
            "first_seen": None,
            "last_seen": None,
            "duration": 0,
            "distance_traveled": 0.0,
            "trajectory_points": 0
        })
        
        for track_id, trajectory in self.object_trajectories.items():
            if len(trajectory) >= 2:
                distance = 0.0
                for i in range(1, len(trajectory)):
                    dx = trajectory[i][0] - trajectory[i-1][0]
                    dy = trajectory[i][1] - trajectory[i-1][1]
                    distance += (dx ** 2 + dy ** 2) ** 0.5
                
                obj_stats[track_id]["distance_traveled"] = distance
                obj_stats[track_id]["trajectory_points"] = len(trajectory)
        
        return dict(obj_stats)
    
    def reset(self):
        self.detection_history.clear()
        self.stage_history.clear()
        self.object_trajectories.clear()
        self.stage_durations.clear()
        self.current_stage_start = None
