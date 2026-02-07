import requests
import time

def test_web_app():
    base_url = "http://localhost:5000"
    
    print("测试YOLO工序检测系统Web应用...")
    print("=" * 50)
    
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        if response.status_code == 200:
            print("✓ 健康检查通过")
            print(f"  响应: {response.json()}")
        else:
            print(f"✗ 健康检查失败: {response.status_code}")
    except Exception as e:
        print(f"✗ 无法连接到服务器: {e}")
        return False
    
    try:
        response = requests.get(f"{base_url}/api/statistics", timeout=5)
        if response.status_code == 200:
            print("✓ 统计API正常")
            print(f"  响应: {response.json()}")
        else:
            print(f"✗ 统计API失败: {response.status_code}")
    except Exception as e:
        print(f"✗ 统计API错误: {e}")
    
    try:
        response = requests.get(f"{base_url}/api/efficiency", timeout=5)
        if response.status_code == 200:
            print("✓ 效率API正常")
            print(f"  响应: {response.json()}")
        else:
            print(f"✗ 效率API失败: {response.status_code}")
    except Exception as e:
        print(f"✗ 效率API错误: {e}")
    
    try:
        response = requests.get(f"{base_url}/api/timeline", timeout=5)
        if response.status_code == 200:
            print("✓ 时间线API正常")
            print(f"  响应: {response.json()}")
        else:
            print(f"✗ 时间线API失败: {response.status_code}")
    except Exception as e:
        print(f"✗ 时间线API错误: {e}")
    
    try:
        response = requests.post(f"{base_url}/api/reset", timeout=5)
        if response.status_code == 200:
            print("✓ 重置API正常")
            print(f"  响应: {response.json()}")
        else:
            print(f"✗ 重置API失败: {response.status_code}")
    except Exception as e:
        print(f"✗ 重置API错误: {e}")
    
    print("=" * 50)
    print("测试完成！")
    print(f"访问Web界面: {base_url}/")
    print(f"访问API文档: {base_url}/docs")

if __name__ == "__main__":
    test_web_app()