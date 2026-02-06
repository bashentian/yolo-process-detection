import unittest
import cv2
import numpy as np
from pathlib import Path
import tempfile

from detector import ProcessDetector, Detection
from config import ProcessDetectionConfig


class TestDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ProcessDetectionConfig()
        cls.config.DEVICE = "cpu"
        cls.detector = ProcessDetector(cls.config)
    
    def test_detector_initialization(self):
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.model)
    
    def test_detect_empty_frame(self):
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = self.detector.detect(empty_frame)
        
        self.assertIsInstance(detections, list)
    
    def test_detect_with_test_image(self):
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = self.detector.detect(test_image)
        
        self.assertIsInstance(detections, list)
        for det in detections:
            self.assertIsInstance(det, Detection)
            self.assertIsInstance(det.bbox, tuple)
            self.assertIsInstance(det.confidence, float)
            self.assertIsInstance(det.class_id, int)
            self.assertIsInstance(det.class_name, str)
    
    def test_detection_properties(self):
        det = Detection(
            bbox=(100.0, 100.0, 200.0, 200.0),
            confidence=0.95,
            class_id=0,
            class_name="test"
        )
        
        center = det.center
        self.assertAlmostEqual(center[0], 150.0)
        self.assertAlmostEqual(center[1], 150.0)
        
        area = det.area
        self.assertAlmostEqual(area, 10000.0)
    
    def test_analyze_process_stage(self):
        from detector import Detection
        
        test_cases = [
            ([], "idle"),
            ([Detection((0,0,100,100), 0.9, 0, "worker")], "idle"),
            ([
                Detection((0,0,100,100), 0.9, 0, "worker"),
                Detection((200,200,300,300), 0.8, 1, "machine"),
                Detection((400,400,500,500), 0.7, 2, "product")
            ], "processing")
        ]
        
        for detections, expected_stage in test_cases:
            stage = self.detector.analyze_process_stage(detections)
            self.assertEqual(stage, expected_stage)


class TestVideoProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ProcessDetectionConfig()
        cls.config.DEVICE = "cpu"
    
    def test_video_processor_initialization(self):
        from video_processor import VideoProcessor
        
        processor = VideoProcessor(self.config)
        self.assertIsNotNone(processor)
        self.assertIsNotNone(processor.detector)
        self.assertIsNotNone(processor.tracker)
    
    def test_process_frame(self):
        from video_processor import VideoProcessor
        
        processor = VideoProcessor(self.config)
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = processor.process_frame(test_frame)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, test_frame.shape)


class TestAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ProcessDetectionConfig()
        cls.analyzer = ProcessAnalyzer(cls.config)
    
    def test_record_detections(self):
        from detector import Detection
        from datetime import datetime
        
        detections = [
            Detection((100,100,200,200), 0.9, 0, "worker"),
            Detection((300,300,400,400), 0.8, 1, "machine")
        ]
        
        self.analyzer.record_detections(detections, 1, datetime.now())
        
        self.assertEqual(len(self.analyzer.detection_history), 1)
        self.assertEqual(len(self.analyzer.detection_history[0]['detections']), 2)
    
    def test_calculate_statistics(self):
        from detector import Detection
        from datetime import datetime
        
        detections = [
            Detection((100,100,200,200), 0.9, 0, "worker"),
            Detection((300,300,400,400), 0.8, 1, "machine")
        ]
        
        self.analyzer.record_detections(detections, 1, datetime.now())
        
        stats = self.analyzer.calculate_statistics()
        
        self.assertIn('total_frames', stats)
        self.assertIn('total_detections', stats)
        self.assertIn('average_confidence', stats)
        self.assertEqual(stats['total_frames'], 1)
        self.assertEqual(stats['total_detections'], 2)
    
    def test_analyze_process_efficiency(self):
        from datetime import datetime, timedelta
        
        now = datetime.now()
        self.analyzer.record_stage_change("processing", now)
        self.analyzer.record_stage_change("assembly", now + timedelta(seconds=10))
        
        efficiency = self.analyzer.analyze_process_efficiency()
        
        self.assertIn('efficiency', efficiency)
        self.assertIn('bottleneck', efficiency)
        self.assertIn('stage_durations', efficiency)


class TestDataUtils(unittest.TestCase):
    def test_dataset_preparer_creation(self):
        from data_utils import DatasetPreparer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            preparer = DatasetPreparer(tmpdir)
            preparer.create_directories()
            
            expected_dirs = ['images', 'labels', 'train', 'val']
            for dir_name in expected_dirs:
                self.assertTrue((Path(tmpdir) / dir_name).exists())
    
    def test_data_augmentor(self):
        from data_utils import DataAugmentor
        
        augmentor = DataAugmentor()
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        augmented = augmentor.augment_image(test_image, methods=['flip'])
        
        self.assertEqual(augmented.shape, test_image.shape)


if __name__ == '__main__':
    unittest.main()
