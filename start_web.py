import uvicorn
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='YOLO工序检测系统 Web服务')
    parser.add_argument('--host', default='0.0.0.0', help='服务主机地址')
    parser.add_argument('--port', type=int, default=5000, help='服务端口')
    parser.add_argument('--reload', action='store_true', help='启用热重载')
    parser.add_argument('--workers', type=int, default=1, help='工作进程数')
    parser.add_argument('--log-level', default='info', choices=['debug', 'info', 'warning', 'error'], help='日志级别')
    
    args = parser.parse_args()
    
    Path('uploads').mkdir(exist_ok=True)
    Path('outputs').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    Path('static').mkdir(exist_ok=True)
    Path('templates').mkdir(exist_ok=True)
    
    print(f"启动 YOLO工序检测系统 Web服务...")
    print(f"服务地址: http://{args.host}:{args.port}")
    print(f"热重载: {'启用' if args.reload else '禁用'}")
    print(f"工作进程数: {args.workers}")
    print(f"日志级别: {args.log_level}")
    print("=" * 50)
    
    uvicorn.run(
        "web_app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level
    )

if __name__ == "__main__":
    main()