@echo off
echo ====================================
echo YOLO工序检测系统 Web服务启动
echo ====================================
echo.

python start_web.py --host 0.0.0.0 --port 5000 --reload --log-level info

pause