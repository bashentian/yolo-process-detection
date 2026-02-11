import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

from detector import Detection


@dataclass
class TrackedObject:
    track_id: int
    detection: Detection
    frame_count: int
    hit_streak: int
    time_since_update: int
    history: deque
    
    def update(self, detection: Detection):
        self.detection = detection
        self.hit_streak += 1
        self.time_since_update = 0
        self.history.append(detection.center)
        if len(self.history) > 10:
            self.history.popleft()
    
    def predict(self) -> Tuple[float, float]:
        if len(self.history) > 1:
            recent_positions = list(self.history)[-2:]
            dx = recent_positions[1][0] - recent_positions[0][0]
            dy = recent_positions[1][1] - recent_positions[0][1]
            x, y = self.detection.center
            return (x + dx, y + dy)
        return self.detection.center
    
    def mark_missed(self):
        self.time_since_update += 1
        self.hit_streak = 0


class ObjectTracker:
    def __init__(self, max_age: int = 30, min_hits: int = 3, 
                 iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_track_id = 1
        self._track_id_pool: deque = field(default_factory=deque)
        
    def _get_track_id(self) -> int:
        """获取track ID，支持循环使用"""
        if self._track_id_pool:
            return self._track_id_pool.popleft()
        else:
            track_id = self.next_track_id
            self.next_track_id += 1
            return track_id
    
    def _recycle_track_id(self, track_id: int):
        """回收track ID"""
        if track_id < self.next_track_id - 1000:
            self._track_id_pool.append(track_id)
        
    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        if not self.tracks:
            for det in detections:
                self._create_new_track(det)
            return self._get_active_tracks()
        
        matched_indices = self._match_detections_vectorized(detections)
        
        unmatched_detections = []
        for i, det in enumerate(detections):
            if i not in matched_indices.get(0, []):
                unmatched_detections.append(det)
        
        unmatched_tracks = []
        for track_id in self.tracks:
            if track_id not in matched_indices.get(1, []):
                unmatched_tracks.append(track_id)
        
        for det_idx, track_id in zip(matched_indices[0], matched_indices[1]):
            self.tracks[track_id].update(detections[det_idx])
        
        for track_id in unmatched_tracks:
            self.tracks[track_id].mark_missed()
        
        for det in unmatched_detections:
            self._create_new_track(det)
        
        self._remove_old_tracks()
        
        return self._get_active_tracks()
    
    def _match_detections_vectorized(self, detections: List[Detection]) -> Dict[str, List[int]]:
        """向量化的检测匹配（性能优化）"""
        n_det = len(detections)
        n_track = len(self.tracks)
        
        if n_det == 0 or n_track == 0:
            return {0: [], 1: []}
        
        det_bboxes = np.array([det.bbox for det in detections], dtype=np.float32)
        track_bboxes = np.array([track.detection.bbox for track in self.tracks.values()], dtype=np.float32)
        
        iou_matrix = self._calculate_iou_vectorized(det_bboxes, track_bboxes)
        
        matched_detections = []
        matched_tracks = []
        
        threshold = self.iou_threshold
        while np.max(iou_matrix) >= threshold:
            max_iou_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            det_idx, track_idx = max_iou_idx
            
            track_id = list(self.tracks.keys())[track_idx]
            
            matched_detections.append(int(det_idx))
            matched_tracks.append(track_id)
            
            iou_matrix[det_idx, :] = -1
            iou_matrix[:, track_idx] = -1
            
            if len(matched_detections) >= min(n_det, n_track):
                break
        
        return {0: matched_detections, 1: matched_tracks}
    
    def _calculate_iou_vectorized(self, bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
        """向量化的IoU计算（性能优化）"""
        n1 = bboxes1.shape[0]
        n2 = bboxes2.shape[0]
        
        bboxes1 = bboxes1.reshape(n1, 1, 4)
        bboxes2 = bboxes2.reshape(1, n2, 4)
        
        x1_min = np.maximum(bboxes1[:, :, 0], bboxes2[:, :, 0])
        y1_min = np.maximum(bboxes1[:, :, 1], bboxes2[:, :, 1])
        x2_min = np.minimum(bboxes1[:, :, 2], bboxes2[:, :, 2])
        y2_min = np.minimum(bboxes1[:, :, 3], bboxes2[:, :, 3])
        
        inter_w = np.maximum(0, x2_min - x1_min)
        inter_h = np.maximum(0, y2_min - y1_min)
        inter_area = inter_w * inter_h
        
        area1 = (bboxes1[:, :, 2] - bboxes1[:, :, 0]) * (bboxes1[:, :, 3] - bboxes1[:, :, 1])
        area2 = (bboxes2[:, :, 2] - bboxes2[:, :, 0]) * (bboxes2[:, :, 3] - bboxes2[:, :, 1])
        union_area = area1 + area2 - inter_area
        
        iou = np.where(union_area > 0, inter_area / union_area, 0)
        
        return iou
    
    def _match_detections(self, detections: List[Detection]) -> Dict[str, List[int]]:
        """原始匹配方法（保留兼容）"""
        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        
        for i, det in enumerate(detections):
            for j, (track_id, track) in enumerate(self.tracks.items()):
                iou_matrix[i, j] = self._calculate_iou(det.bbox, track.detection.bbox)
        
        matched_detections = []
        matched_tracks = []
        
        while np.max(iou_matrix) >= self.iou_threshold:
            max_iou_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            det_idx, track_idx = max_iou_idx
            
            track_id = list(self.tracks.keys())[track_idx]
            
            matched_detections.append(det_idx)
            matched_tracks.append(track_id)
            
            iou_matrix[det_idx, :] = -1
            iou_matrix[:, track_idx] = -1
        
        return {0: matched_detections, 1: matched_tracks}
    
    def _calculate_iou(self, bbox1: tuple, bbox2: tuple) -> float:
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_x1 = max(x1_min, x2_min)
        inter_y1 = max(y1_min, y2_min)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _create_new_track(self, detection: Detection):
        track = TrackedObject(
            track_id=self.next_track_id,
            detection=detection,
            frame_count=0,
            hit_streak=1,
            time_since_update=0,
            history=deque([detection.center], maxlen=10)
        )
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1
    
    def _remove_old_tracks(self):
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track.time_since_update > self.max_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def _get_active_tracks(self) -> List[TrackedObject]:
        return [
            track for track in self.tracks.values()
            if track.hit_streak >= self.min_hits
        ]
    
    def get_tracks_by_class(self, class_name: str) -> List[TrackedObject]:
        return [
            track for track in self.tracks.values()
            if track.detection.class_name == class_name and 
               track.hit_streak >= self.min_hits
        ]
