"""测试跟踪器模块"""
import pytest
from unittest.mock import Mock
from pathlib import Path

from yolo_process_detection.models.tracker import (
    IoUTracker,
    ObjectTracker,
    TrackedObject
)


class TestTrackedObject:
    """测试TrackedObject类"""
    
    def test_create_tracked_object(self):
        """测试创建跟踪对象"""
        track = TrackedObject(
            track_id=1,
            detection=(100, 100, 200, 200)
        )
        
        assert track.track_id == 1
        assert track.detection == (100, 100, 200, 200)
        assert track.hit_streak == 0
        assert track.time_since_update == 0
    
    def test_update_detection(self):
        """测试更新检测"""
        track = TrackedObject(
            track_id=1,
            detection=(100, 100, 200, 200)
        )
        
        track.update((150, 150, 250, 250))
        
        assert track.detection == (150, 150, 250, 250)
        assert track.hit_streak == 1
        assert track.time_since_update == 0
    
    def test_center_calculation(self):
        """测试中心点计算"""
        track = TrackedObject(
            track_id=1,
            detection=(100, 100, 200, 200)
        )
        
        center = track.center
        
        assert center == (150.0, 150.0)
    
    def test_mark_missed(self):
        """测试标记丢失"""
        track = TrackedObject(
            track_id=1,
            detection=(100, 100, 200, 200)
        )
        track.hit_streak = 3
        
        track.mark_missed()
        
        assert track.time_since_update == 1
        assert track.hit_streak == 0
    
    def test_history_management(self):
        """测试历史记录管理"""
        track = TrackedObject(
            track_id=1,
            detection=(100, 100, 200, 200)
        )
        
        for i in range(15):
            track.update((100 + i, 100 + i, 200 + i, 200 + i))
        
        assert len(track.history) == 10  # 保留最近10个


class TestIoUTracker:
    """测试IoU跟踪器"""
    
    def test_empty_update(self):
        """测试空更新"""
        tracker = IoUTracker()
        tracker.update([])
        
        assert len(tracker.tracks) == 0
    
    def test_single_detection(self):
        """测试单检测更新"""
        from yolo_process_detection.models.tracker import TrackedDetection
        
        tracker = IoUTracker()
        
        detection = TrackedDetection(
            track_id=0,
            detection=(100, 100, 200, 200),
            confidence=0.95,
            class_id=0,
            class_name="worker",
            hit_streak=1,
            time_since_update=0
        )
        
        result = tracker.update([detection])
        
        assert len(tracker.tracks) == 1
        assert len(result) == 1
    
    def test_iou_calculation(self):
        """测试IoU计算"""
        tracker = IoUTracker()
        
        bbox1 = (100, 100, 200, 200)
        bbox2 = (150, 150, 250, 250)
        
        iou = tracker._calculate_iou(bbox1, bbox2)
        
        assert 0 < iou <= 1
        assert iou > 0.3  # 有重叠
    
    def test_no_overlap_iou(self):
        """测试无重叠IoU"""
        tracker = IoUTracker()
        
        bbox1 = (0, 0, 100, 100)
        bbox2 = (200, 200, 300, 300)
        
        iou = tracker._calculate_iou(bbox1, bbox2)
        
        assert iou == 0
    
    def test_identical_bbox_iou(self):
        """测试相同边界框IoU"""
        tracker = IoUTracker()
        
        bbox = (100, 100, 200, 200)
        
        iou = tracker._calculate_iou(bbox, bbox)
        
        assert iou == 1.0
    
    def test_track_id_recycling(self):
        """测试跟踪ID回收"""
        tracker = IoUTracker()
        
        ids = []
        for i in range(1005):
            tracker.tracks.clear()
            detection = TrackedDetection(
                track_id=0,
                detection=(100 + i, 100, 200 + i, 200),
                confidence=0.95,
                class_id=0,
                class_name="worker",
                hit_streak=1,
                time_since_update=0
            )
            tracker.update([detection])
            if tracker.tracks:
                ids.append(list(tracker.tracks.keys())[0])
        
        # 验证ID回收
        assert len(set(ids)) < 1005


class TestObjectTracker:
    """测试对象跟踪器（兼容接口）"""
    
    def test_create_tracker(self):
        """测试创建跟踪器"""
        tracker = ObjectTracker(
            max_age=30,
            min_hits=3,
            iou_threshold=0.3
        )
        
        assert tracker._tracker.max_age == 30
        assert tracker._tracker.min_hits == 3
        assert tracker._tracker.iou_threshold == 0.3
    
    def test_update_detections(self):
        """测试更新检测"""
        tracker = ObjectTracker()
        
        detections = [
            (100, 100, 200, 200),
            (300, 300, 400, 400)
        ]
        
        result = tracker.update(detections)
        
        assert len(tracker._tracker.tracks) == 2
    
    def test_reset(self):
        """测试重置"""
        tracker = ObjectTracker()
        
        tracker.update([(100, 100, 200, 200)])
        tracker.reset()
        
        assert len(tracker._tracker.tracks) == 0
