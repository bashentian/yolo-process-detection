"""跟踪器模块

提供基于IoU的多目标跟踪功能。
"""
import numpy as np
from typing import NamedTuple
from dataclasses import dataclass, field
from collections import deque


class TrackedDetection(NamedTuple):
    """跟踪检测结果"""
    track_id: int
    detection: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    hit_streak: int
    time_since_update: int


@dataclass
class TrackedObject:
    """跟踪对象"""
    track_id: int
    detection: tuple[float, float, float, float]
    hit_streak: int = 0
    time_since_update: int = 0
    history: deque = field(default_factory=deque)
    
    def update(self, detection: tuple[float, float, float, float]) -> None:
        """更新检测"""
        self.detection = detection
        self.hit_streak += 1
        self.time_since_update = 0
        self.history.append(detection)
        if len(self.history) > 10:
            self.history.popleft()
    
    def predict(self) -> tuple[float, float]:
        """预测下一位置"""
        if len(self.history) > 1:
            recent = list(self.history)[-2:]
            dx = recent[1][0] - recent[0][0]
            dy = recent[1][1] - recent[0][1]
            x, y = (recent[1][0] + recent[1][2]) / 2, (recent[1][1] + recent[1][3]) / 2
            return (x + dx, y + dy)
        return self.center
    
    @property
    def center(self) -> tuple[float, float]:
        """获取中心点"""
        x1, y1, x2, y2 = self.detection
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def mark_missed(self) -> None:
        """标记为丢失"""
        self.time_since_update += 1
        self.hit_streak = 0


class IoUTracker:
    """基于IoU的目标跟踪器"""
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: dict[int, TrackedObject] = {}
        self._next_track_id: int = 1
        self._track_id_pool: deque = field(default_factory=deque)
    
    def _get_track_id(self) -> int:
        """获取新跟踪ID"""
        if self._track_id_pool:
            return self._track_id_pool.popleft()
        track_id = self._next_track_id
        self._next_track_id += 1
        return track_id
    
    def _recycle_track_id(self, track_id: int) -> None:
        """回收跟踪ID"""
        if track_id < self._next_track_id - 1000:
            self._track_id_pool.append(track_id)
    
    def _calculate_iou(
        self,
        bbox1: tuple[float, float, float, float],
        bbox2: tuple[float, float, float, float]
    ) -> float:
        """计算两个边界框的IoU"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_x1 = max(x1_min, x2_min)
        inter_y1 = max(y1_min, y2_min)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _match_detections(
        self,
        detections: list[tuple]
    ) -> tuple[list[int], list[int]]:
        """匹配检测和跟踪"""
        n_det = len(detections)
        n_track = len(self.tracks)
        
        if n_det == 0 or n_track == 0:
            return [], []
        
        iou_matrix = np.zeros((n_det, n_track))
        for i, det in enumerate(detections):
            for j, track in enumerate(self.tracks.values()):
                iou_matrix[i, j] = self._calculate_iou(det, track.detection)
        
        matched_det: list[int] = []
        matched_track: list[int] = []
        track_ids = list(self.tracks.keys())
        
        while np.max(iou_matrix) >= self.iou_threshold:
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            det_idx, track_idx = max_idx
            
            matched_det.append(det_idx)
            matched_track.append(track_ids[track_idx])
            
            iou_matrix[det_idx, :] = -1
            iou_matrix[:, track_idx] = -1
        
        return matched_det, matched_track
    
    def update(self, detections: list[TrackedDetection]) -> list[TrackedDetection]:
        """更新跟踪器"""
        if not self.tracks:
            for det in detections:
                track_id = self._get_track_id()
                self.tracks[track_id] = TrackedObject(
                    track_id=track_id,
                    detection=det.detection
                )
            return self._get_active_tracks()
        
        matched_det, matched_track = self._match_detections(
            [d.detection for d in detections]
        )
        
        unmatched_det_indices = set(range(len(detections))) - set(matched_det)
        unmatched_track_ids = set(self.tracks.keys()) - set(matched_track)
        
        for det_idx, track_id in zip(matched_det, matched_track):
            if track_id in self.tracks:
                self.tracks[track_id].update(detections[det_idx].detection)
        
        for track_id in unmatched_track_ids:
            self.tracks[track_id].mark_missed()
        
        for det_idx in unmatched_det_indices:
            det = detections[det_idx]
            track_id = self._get_track_id()
            self.tracks[track_id] = TrackedObject(
                track_id=track_id,
                detection=det.detection
            )
        
        self._remove_old_tracks()
        
        return self._get_active_tracks()
    
    def _remove_old_tracks(self) -> None:
        """移除丢失的跟踪"""
        track_ids_to_remove = [
            track_id for track_id, track in self.tracks.items()
            if track.time_since_update > self.max_age
        ]
        for track_id in track_ids_to_remove:
            self._recycle_track_id(track_id)
            del self.tracks[track_id]
    
    def _get_active_tracks(self) -> list[TrackedDetection]:
        """获取活跃跟踪"""
        return [
            TrackedDetection(
                track_id=track.track_id,
                detection=track.detection,
                confidence=0.0,
                class_id=0,
                class_name="unknown",
                hit_streak=track.hit_streak,
                time_since_update=track.time_since_update
            )
            for track in self.tracks.values()
            if track.hit_streak >= self.min_hits
        ]
    
    def reset(self) -> None:
        """重置跟踪器"""
        self.tracks.clear()
        self._next_track_id = 1
        self._track_id_pool.clear()


class ObjectTracker:
    """对象跟踪器（兼容旧接口）"""
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        self._tracker = IoUTracker(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold
        )
    
    def update(
        self,
        detections: list[tuple]
    ) -> list[TrackedDetection]:
        """更新跟踪"""
        tracked = self._tracker.update([
            TrackedDetection(
                track_id=0,
                detection=det,
                confidence=0.0,
                class_id=0,
                class_name="unknown",
                hit_streak=1,
                time_since_update=0
            )
            for det in detections
        ])
        return tracked
    
    def reset(self) -> None:
        """重置"""
        self._tracker.reset()
