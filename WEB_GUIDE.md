# YOLO工序检测系统 Web界面使用指南

## 概述

本系统采用FastAPI + Bootstrap 5构建的现代化Web界面，提供了直观的工序检测和监控功能。

## 技术栈

### 后端
- **FastAPI**: 高性能异步Web框架
- **Uvicorn**: ASGI服务器
- **Jinja2**: 模板引擎
- **Python 3.10+**: 运行环境

### 前端
- **Bootstrap 5**: 响应式UI框架
- **Bootstrap Icons**: 图标库
- **原生JavaScript**: 交互逻辑
- **Fetch API**: 异步数据请求

## 快速启动

### 1. 安装依赖

```bash
pip install fastapi uvicorn[standard] python-multipart jinja2 numpy
```

### 2. 启动Web服务

#### 方式一：使用简化版（无需完整YOLO环境）
```bash
python web_app_simple.py
```

#### 方式二：使用完整版（需要完整的YOLO环境）
```bash
python web_app.py
```

#### 方式三：使用启动脚本
```bash
python start_web.py --host 0.0.0.0 --port 5000 --reload
```

#### 方式四：使用批处理文件（Windows）
```bash
start_web.bat
```

### 3. 访问Web界面

在浏览器中打开：`http://localhost:5000`

### 4. 访问API文档

FastAPI自动生成的交互式API文档：
- Swagger UI: `http://localhost:5000/docs`
- ReDoc: `http://localhost:5000/redoc`

## 功能说明

### 主要功能

#### 1. 视频上传与处理
- 支持拖拽上传视频文件
- 支持点击选择文件
- 实时显示上传进度
- 异步视频处理

#### 2. 实时监控
- 工序阶段显示
- 实时统计数据
- 处理进度跟踪
- 系统健康监控

#### 3. 统计分析
- 整体效率计算
- 检测总数统计
- 处理帧数统计
- 平均耗时分析

#### 4. 时间线展示
- 工序阶段变化记录
- 时间戳显示
- 历史回溯

#### 5. 数据导出
- JSON格式导出
- 完整分析结果
- 可用于后续分析

### API端点

#### 基础端点
- `GET /` - Web界面
- `GET /api/health` - 健康检查

#### 视频处理端点
- `POST /api/upload_video` - 上传视频
- `POST /api/process_video` - 处理视频

#### 数据获取端点
- `GET /api/statistics` - 获取统计数据
- `GET /api/efficiency` - 获取效率分析
- `GET /api/timeline` - 获取时间线
- `GET /api/anomalies` - 获取异常检测

#### 系统控制端点
- `POST /api/reset` - 重置分析
- `POST /api/export_results` - 导出结果

## 界面说明

### 顶部导航栏
- 系统标题：YOLO工序检测系统
- 版本信息：v2.0
- 在线状态：显示系统运行状态
- 当前时间：实时更新

### 主监控区域
- **视频显示区**：显示实时视频流或处理结果
- **上传区域**：拖拽或点击上传视频文件
- **控制按钮**：
  - 处理视频：开始处理上传的视频
  - 导出结果：导出分析结果
  - 重置分析：清空当前分析数据

### 统计卡片
- **整体效率**：工序整体效率百分比
- **检测总数**：累计检测到的对象数量
- **处理帧数**：处理的视频帧数
- **平均耗时**：每帧平均处理时间

### 工序时间线
- 显示工序阶段变化历史
- 每个阶段的时间戳
- 按时间顺序排列

### 系统状态
- **系统健康度**：系统运行状态百分比
- **处理进度**：当前处理进度百分比
- **消息通知**：系统消息和错误提示

## 使用流程

### 1. 上传视频
1. 点击上传区域或拖拽视频文件
2. 等待上传完成
3. 系统会显示上传成功消息

### 2. 处理视频
1. 点击"处理视频"按钮
2. 等待处理完成（显示加载动画）
3. 处理完成后自动显示结果

### 3. 查看结果
1. 查看统计数据更新
2. 查看效率分析结果
3. 查看工序时间线
4. 查看系统状态

### 4. 导出结果
1. 点击"导出结果"按钮
2. 结果以JSON格式保存到outputs目录
3. 文件名包含时间戳

### 5. 重置分析
1. 点击"重置分析"按钮
2. 清空所有统计数据
3. 准备进行新的分析

## 响应式设计

### 桌面端 (>992px)
- 完整功能显示
- 8列视频区域，4列统计区域
- 最佳用户体验

### 平板端 (768px-992px)
- 调整布局适配
- 统计卡片自适应
- 保持核心功能

### 移动端 (<768px)
- 单列布局
- 简化显示
- 触摸优化

## 自定义配置

### 修改端口
```bash
python web_app_simple.py --port 8080
```

### 修改主机地址
```bash
python web_app_simple.py --host 192.168.1.100
```

### 启用热重载（开发模式）
```bash
python web_app_simple.py --reload
```

### 修改日志级别
```bash
python web_app_simple.py --log-level debug
```

## 故障排除

### 问题1：无法启动服务
**解决方案**：
- 检查端口5000是否被占用
- 检查依赖是否完整安装
- 查看错误日志

### 问题2：视频上传失败
**解决方案**：
- 检查文件格式是否支持（MP4, AVI, MOV）
- 检查文件大小是否超过限制（16MB）
- 检查uploads目录权限

### 问题3：处理速度慢
**解决方案**：
- 使用GPU加速
- 降低视频分辨率
- 优化模型参数

### 问题4：统计数据不更新
**解决方案**：
- 检查API连接
- 刷新页面
- 查看浏览器控制台错误

## 性能优化

### 前端优化
- 使用CDN加载静态资源
- 实现数据缓存
- 减少不必要的API请求

### 后端优化
- 异步处理耗时操作
- 使用连接池
- 实现数据缓存

### 部署优化
- 使用Nginx反向代理
- 启用Gzip压缩
- 使用多进程部署

## 安全建议

1. **文件上传限制**
   - 限制文件类型
   - 限制文件大小
   - 验证文件内容

2. **API访问控制**
   - 添加身份认证
   - 实现速率限制
   - 记录访问日志

3. **数据保护**
   - 使用HTTPS
   - 敏感数据加密
   - 定期备份数据

## 扩展功能

### 添加新的API端点
```python
@app.get("/api/custom_endpoint")
async def custom_endpoint():
    return {"success": True, "data": "custom response"}
```

### 添加新的页面
```python
@app.get("/custom_page", response_class=HTMLResponse)
async def custom_page(request: Request):
    return templates.TemplateResponse("custom.html", {"request": request})
```

### 添加数据库支持
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine("sqlite:///database.db")
SessionLocal = sessionmaker(bind=engine)
```

## 维护建议

1. **定期清理**
   - 清理上传的视频文件
   - 清理临时文件
   - 清理日志文件

2. **监控日志**
   - 检查错误日志
   - 监控性能指标
   - 记录用户行为

3. **更新依赖**
   - 定期更新Python包
   - 测试更新后的兼容性
   - 备份当前环境

## 技术支持

如遇到问题，请检查：
1. 浏览器控制台错误
2. 服务器端日志
3. 网络连接状态
4. 依赖版本兼容性

## 许可证

本系统遵循项目主许可证。

## 更新日志

### v2.0 (2024)
- 迁移到FastAPI框架
- 使用Bootstrap 5重构界面
- 添加实时监控功能
- 优化性能和用户体验
- 添加响应式设计支持