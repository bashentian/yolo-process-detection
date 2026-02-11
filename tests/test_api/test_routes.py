"""测试API路由模块"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """测试健康检查端点"""
    
    def test_health_check(self, api_client):
        """测试健康检查"""
        response = api_client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "model_loaded" in data


class TestDetectionEndpoint:
    """测试检测端点"""
    
    def test_detect_no_image(self, api_client):
        """测试无图像请求"""
        response = api_client.post("/api/detect")
        
        assert response.status_code == 422  # Validation error
    
    def test_detect_with_image(self, api_client, mock_upload_file):
        """测试带图像请求"""
        with patch("builtins.open", Mock()):
            response = api_client.post(
                "/api/detect",
                files={"image": ("test.jpg", b"fake content", "image/jpeg")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "detections" in data
        assert "inference_time" in data
    
    def test_detect_with_confidence(self, api_client, mock_upload_file):
        """测试带置信度请求"""
        response = api_client.post(
            "/api/detect?confidence=0.8",
            files={"image": ("test.jpg", b"fake content", "image/jpeg")}
        )
        
        assert response.status_code == 200
    
    def test_batch_detect(self, api_client, mock_upload_file):
        """测试批量检测"""
        files = [
            ("images", ("test1.jpg", b"content1", "image/jpeg")),
            ("images", ("test2.jpg", b"content2", "image/jpeg")),
        ]
        
        response = api_client.post("/api/detect/batch", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_images"] == 2
        assert "results" in data


class TestStatisticsEndpoint:
    """测试统计端点"""
    
    def test_get_statistics(self, api_client):
        """测试获取统计"""
        response = api_client.get("/api/statistics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert "summary" in data


class TestEfficiencyEndpoint:
    """测试效率端点"""
    
    def test_get_efficiency(self, api_client):
        """测试获取效率"""
        response = api_client.get("/api/efficiency")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert "score" in data
        assert "status" in data


class TestTimelineEndpoint:
    """测试时间线端点"""
    
    def test_get_timeline(self, api_client):
        """测试获取时间线"""
        response = api_client.get("/api/timeline")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert "total_points" in data
    
    def test_get_timeline_with_params(self, api_client):
        """测试带参数获取时间线"""
        response = api_client.get("/api/timeline?start_frame=0&end_frame=100")
        
        assert response.status_code == 200


class TestAnomalyEndpoint:
    """测试异常检测端点"""
    
    def test_get_anomalies(self, api_client):
        """测试获取异常"""
        response = api_client.get("/api/anomalies")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert "total_checked" in data
        assert "anomaly_count" in data


class TestConfigEndpoint:
    """测试配置端点"""
    
    def test_get_config(self, api_client):
        """测试获取配置"""
        response = api_client.get("/api/config")
        
        assert response.status_code == 200
        data = response.json()
        assert "detection" in data
        assert "tracking" in data
        assert "analysis" in data
    
    def test_update_config(self, api_client):
        """测试更新配置"""
        response = api_client.put(
            "/api/config",
            json={
                "detection": {
                    "confidence_threshold": 0.7
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert "updated_fields" in data


class TestResetEndpoint:
    """测试重置端点"""
    
    def test_reset_analysis(self, api_client):
        """测试重置分析"""
        response = api_client.post("/api/reset?clear_history=true")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert "history" in data["cleared_data"]


class TestExportEndpoint:
    """测试导出端点"""
    
    def test_export_json(self, api_client):
        """测试导出JSON"""
        response = api_client.get("/api/export?format=json")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert data["format"] == "json"
    
    def test_export_csv(self, api_client):
        """测试导出CSV"""
        response = api_client.get("/api/export?format=csv")
        
        assert response.status_code == 200
