import sys
from unittest.mock import MagicMock

# Mock torch
mock_torch = MagicMock()
mock_torch.nn.Module = object
sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = mock_torch.nn
sys.modules["torch.nn.functional"] = mock_torch.nn.functional

# Mock ultralytics
mock_ultralytics = MagicMock()
sys.modules["ultralytics"] = mock_ultralytics

# Mock core.config to avoid pydantic_settings import issues and dependency on real config
mock_config_module = MagicMock()
# We need get_settings to be available
mock_config_module.get_settings = MagicMock()
sys.modules["src.yolo_process_detection.core.config"] = mock_config_module
# Also mock the relative import path if needed (though sys.modules should handle it if imported as absolute)
# But inside detector.py it is `from ..core.config import get_settings`
# We might need to ensure the parent package is importable?
# Actually, if we import `src.yolo_process_detection.models.detector`, it resolves `..core.config` to `src.yolo_process_detection.core.config`.

import pytest
import numpy as np
from unittest.mock import Mock, patch

# Now import the module under test
from src.yolo_process_detection.models.detector import (
    YOLODetector, 
    SceneUnderstanding, 
    AnomalyDetection, 
    EfficiencyAnalyzer,
    Detection,
    RegionalAttention
)

class TestSceneUnderstanding:
    def test_identify_stage_idle(self):
        scene = SceneUnderstanding()
        detections = []
        result = scene.identify_stage(detections, np.zeros((100, 100, 3)))
        assert result['stage'] == 'idle'
        assert result['confidence'] == 0.0

    def test_identify_stage_processing(self):
        scene = SceneUnderstanding()
        # machine + worker = processing
        detections = [
            Detection((0,0,10,10), 0.9, 0, 'worker'),
            Detection((20,20,30,30), 0.9, 1, 'machine')
        ]
        result = scene.identify_stage(detections, np.zeros((100, 100, 3)))
        assert result['stage'] == 'processing'
        assert '检测到工作人员' in result['context']

class TestAnomalyDetection:
    def test_normal_detection(self):
        anomaly = AnomalyDetection(history_size=10)
        # Fill history with normal data
        for _ in range(10):
            anomaly.update([Detection((0,0,10,10), 0.9, 0, 'worker')], 0.1)
        
        # Test normal case
        result = anomaly.detect([Detection((0,0,10,10), 0.9, 0, 'worker')], 0.1)
        assert not result['is_anomaly']

    def test_anomaly_object_count(self):
        anomaly = AnomalyDetection(history_size=10)
        # Fill history
        for _ in range(10):
            anomaly.update([Detection((0,0,10,10), 0.9, 0, 'worker')], 0.1)
        
        # Test anomaly (too many objects)
        many_detections = [Detection((0,0,10,10), 0.9, 0, 'worker')] * 100
        result = anomaly.detect(many_detections, 0.1)
        assert result['is_anomaly']
        assert result['type'] == 'object_count'

class TestEfficiencyAnalyzer:
    def test_analyze_insufficient_data(self):
        analyzer = EfficiencyAnalyzer()
        result = analyzer.analyze()
        assert result['status'] == 'collecting'

    def test_analyze_good_performance(self):
        analyzer = EfficiencyAnalyzer(window_size=10)
        # Simulate good performance
        for _ in range(10):
            # 100 objects in 0.1s = 1000 throughput (excellent)
            detections = [Detection((0,0,10,10), 0.9, 0, 'worker')] * 10
            analyzer.update(detections, 0.1)
            
        result = analyzer.analyze()
        assert result['score'] > 0.8
        assert result['status'] in ['good', 'excellent']

def test_yolo_detector_initialization():
    # Setup mock settings
    mock_settings = Mock()
    mock_settings.model_name = "yolo11n.pt"
    mock_settings.confidence_threshold = 0.5
    mock_settings.iou_threshold = 0.45
    mock_settings.device = "cpu"
    mock_settings.anomaly_history_size = 100
    mock_settings.efficiency_window_size = 50
    mock_settings.use_attention = False
    
    mock_config_module.get_settings.return_value = mock_settings

    detector = YOLODetector()
    assert detector.model_name == "yolo11n.pt"
    assert not detector.use_attention

@patch('src.yolo_process_detection.models.detector.get_settings')
def test_detect_advanced(mock_get_settings):
    # Setup mocks
    mock_settings = Mock()
    mock_settings.model_name = "yolo11n.pt"
    mock_settings.confidence_threshold = 0.5
    mock_settings.iou_threshold = 0.45
    mock_settings.device = "cpu"
    mock_settings.anomaly_history_size = 100
    mock_settings.efficiency_window_size = 50
    mock_settings.use_attention = False
    mock_get_settings.return_value = mock_settings
    
    detector = YOLODetector()
    
    # Mock the model instance
    mock_model_instance = MagicMock()
    # Mock result
    mock_box = Mock()
    mock_box.xyxy = [Mock(cpu=lambda: Mock(numpy=lambda: np.array([0, 0, 100, 100])))]
    mock_box.conf = [Mock(cpu=lambda: Mock(numpy=lambda: np.array(0.9)))]
    mock_box.cls = [Mock(cpu=lambda: Mock(numpy=lambda: np.array(0)))]
    
    mock_result = Mock()
    mock_result.boxes = [mock_box]
    mock_result.names = {0: 'worker'}
    mock_model_instance.return_value = [mock_result]
    
    # Inject mock model
    detector._model = mock_model_instance

    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    
    result = detector.detect_advanced(frame)
    
    assert 'detections' in result
    assert 'scene' in result
    assert 'anomaly' in result
    assert 'efficiency' in result
    assert len(result['detections']) == 1
    assert result['detections'][0].class_name == 'worker'
