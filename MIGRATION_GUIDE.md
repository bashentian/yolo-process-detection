# YOLO检测系统迁移指南

本指南帮助您将现有代码迁移到新的统一模块。

---

## 一、迁移概览

新的统一模块包括：
- `config_validator.py` - 配置验证
- `performance_monitor.py` - 性能监控
- `error_handler.py` - 错误处理
- `optimized_main.py` - 优化的主程序

---

## 二、逐步迁移

### 步骤 1: 更新导入语句

#### 旧代码
```python
from utils import PerformanceMonitor, log_execution_time
from logger import ErrorHandler
```

#### 新代码
```python
from performance_monitor import (
    PerformanceMonitor,
    default_performance_monitor,
    measure_sync,
    measure_async
)
from error_handler import (
    ErrorHandler,
    default_error_handler,
    catch_errors,
    AppException
)
from config_validator import validate_detection_config
```

### 步骤 2: 替换性能监控

#### 场景 1: 函数性能监控

#### 旧代码 (utils.py)
```python
from utils import log_execution_time

@log_execution_time()
def detect_objects():
    # 检测逻辑
    pass
```

#### 新代码
```python
from performance_monitor import measure_sync

@measure_sync("detect_objects")
def detect_objects():
    # 检测逻辑
    pass
```

#### 场景 2: 异步函数性能监控

#### 旧代码
```python
# 不支持异步
```

#### 新代码
```python
from performance_monitor import measure_async

@measure_async("detect_objects")
async def detect_objects():
    # 异步检测逻辑
    pass
```

#### 场景 3: 上下文管理器

#### 旧代码
```python
from utils import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start_timer("operation")
# 执行操作
monitor.end_timer("operation")
```

#### 新代码
```python
from performance_monitor import default_performance_monitor

with default_performance_monitor.sync_measure("operation"):
    # 执行操作
    pass
```

### 步骤 3: 替换错误处理

#### 场景 1: 异常捕获装饰器

#### 旧代码 (utils.py)
```python
from utils import handle_exceptions

@handle_exceptions()
def process_data():
    # 可能抛出异常的代码
    pass
```

#### 新代码
```python
from error_handler import catch_errors

@catch_errors("处理数据", default_return=None)
def process_data():
    # 可能抛出异常的代码
    pass
```

#### 场景 2: 异步异常处理

#### 旧代码
```python
# 不支持异步
```

#### 新代码
```python
from error_handler import catch_errors

@catch_errors("处理数据", default_return=None)
async def process_data():
    # 异步处理代码
    pass
```

#### 场景 3: 自定义异常

#### 旧代码
```python
try:
    raise Exception("模型加载失败")
except Exception as e:
    logger.error(e)
```

#### 新代码
```python
from error_handler import ModelLoadError, ErrorSeverity

try:
    raise ModelLoadError("模型加载失败", severity=ErrorSeverity.CRITICAL)
except ModelLoadError as e:
    logger.error(e.message)
```

### 步骤 4: 添加配置验证

#### 旧代码
```python
from config import ProcessDetectionConfig

config = ProcessDetectionConfig()
# 直接使用配置，没有验证
```

#### 新代码
```python
from config import ProcessDetectionConfig
from config_validator import validate_detection_config

config = ProcessDetectionConfig()
validate_detection_config(config)  # 验证配置
```

### 步骤 5: 更新文件验证

#### 旧代码 (utils.py 或 logger.py)
```python
from utils import validate_image_path, validate_video_path

if validate_image_path(image_path):
    # 处理图像
    pass
```

#### 新代码
```python
from error_handler import validate_image_path, validate_video_path

if validate_image_path(image_path):
    # 处理图像
    pass
```

---

## 三、模块对应关系

| 旧模块 | 新模块 | 说明 |
|--------|--------|------|
| `utils.py` 的 `PerformanceMonitor` | `performance_monitor.py` | 更强大的性能监控 |
| `logger.py` 的 `PerformanceMonitor` | `performance_monitor.py` | 统一的性能监控 |
| `logger.py` 的 `ErrorHandler` | `error_handler.py` | 扩展的错误处理 |
| `utils.py` 的 `log_execution_time` | `performance_monitor.py` 的 `measure_sync/async` | 支持异步 |
| `utils.py` 的 `handle_exceptions` | `error_handler.py` 的 `catch_errors` | 支持异步 |
| - | `config_validator.py` | 新增配置验证 |

---

## 四、API兼容性

### 保留的API

以下API保持兼容，可以直接替换导入：

```python
# utils.py
- validate_image_path() -> error_handler.validate_image_path()
- validate_video_path() -> error_handler.validate_video_path()
- safe_divide() -> error_handler.safe_divide()

# logger.py
- ErrorHandler.handle() -> error_handler.ErrorHandler.handle()
- ErrorHandler.validate_file_path() -> error_handler.validate_file_path()
```

### 新增API

新的统一模块提供了以下新功能：

```python
# 性能监控
- PerformanceMonitor.get_stats_summary()
- PerformanceMonitor.get_average_duration()
- PerformanceMonitor.get_recent_stats()
- measure_sync() 上下文管理器
- measure_async() 上下文管理器

# 错误处理
- AppException 基类及其子类
- ErrorSeverity 枚举
- ErrorHandler.get_error_summary()
- setup_global_exception_handler()

# 配置验证
- ConfigValidator.validate_config()
- validate_model_config()
- validate_process_config()
```

---

## 五、完整迁移示例

### 示例: API端点迁移

#### 旧代码 (api.py)
```python
from fastapi import FastAPI, HTTPException
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

@app.post("/api/detect")
async def detect(file):
    try:
        # 处理逻辑
        start_time = time.time()
        result = process(file)
        duration = time.time() - start_time
        logger.info(f"处理耗时: {duration:.2f}s")
        return result
    except Exception as e:
        logger.error(f"处理失败: {e}")
        raise HTTPException(status_code=500)
```

#### 新代码
```python
from fastapi import FastAPI, HTTPException
from performance_monitor import default_performance_monitor, measure_async
from error_handler import catch_errors, DetectionError, ErrorSeverity

app = FastAPI()

@app.post("/api/detect")
@measure_async("detect")
@catch_errors("检测端点", default_return=None)
async def detect(file):
    # 处理逻辑（性能监控和错误处理自动处理）
    result = process(file)
    return result

# 或者使用自定义异常处理
@app.post("/api/detect")
async def detect(file):
    try:
        with default_performance_monitor.async_measure("detect"):
            result = process(file)
        return result
    except Exception as e:
        raise DetectionError(
            f"检测失败: {e}",
            severity=ErrorSeverity.ERROR,
            context={"file": file.filename}
        )
```

### 示例: 视频处理迁移

#### 旧代码 (video_processor.py)
```python
import time
import logging

class VideoProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def process_video(self, video_path):
        start_time = time.time()
        try:
            # 处理逻辑
            for frame in frames:
                result = process_frame(frame)
                self.logger.debug(f"处理帧: {frame}")
        except Exception as e:
            self.logger.error(f"处理失败: {e}")
            raise
        finally:
            duration = time.time() - start_time
            self.logger.info(f"总耗时: {duration:.2f}s")
```

#### 新代码
```python
from performance_monitor import PerformanceMonitor
from error_handler import catch_errors, VideoProcessingError

class VideoProcessor:
    def __init__(self):
        self.monitor = PerformanceMonitor()

    @catch_errors("视频处理")
    def process_video(self, video_path):
        with self.monitor.sync_measure("process_video", log_threshold=0.5):
            # 处理逻辑
            for frame in frames:
                with self.monitor.sync_measure("process_frame"):
                    result = process_frame(frame)

        # 处理完成后获取统计信息
        summary = self.monitor.get_stats_summary()
        self.monitor.logger.info(f"处理统计: {summary}")
```

---

## 六、检查清单

迁移完成后，请检查以下事项：

- [ ] 所有 `utils.py` 的导入已更新
- [ ] 所有 `logger.py` 的导入已更新
- [ ] 所有性能监控使用新模块
- [ ] 所有错误处理使用新模块
- [ ] 添加了配置验证
- [ ] 测试所有功能正常运行
- [ ] 验证性能指标正常
- [ ] 检查日志输出格式正确

---

## 七、回滚方案

如果迁移后出现问题，可以快速回滚：

1. 保留旧的导入和代码注释
2. 使用条件导入：
```python
try:
    from performance_monitor import default_performance_monitor
    NEW_MODULES_AVAILABLE = True
except ImportError:
    from utils import PerformanceMonitor
    NEW_MODULES_AVAILABLE = False
```

3. 根据模块可用性选择实现

---

## 八、常见问题

### Q1: 新模块是否向后兼容？
A: 大部分API保持兼容，但建议逐步迁移到新API以获得更好的功能。

### Q2: 性能监控会影响性能吗？
A: 影响极小（微秒级），但可以通过设置阈值和禁用来进一步减少影响。

### Q3: 错误处理装饰器会影响异常类型吗？
A: 不会，异常类型和堆栈跟踪都会保留。

### Q4: 配置验证会修改配置吗？
A: 不会，验证只读取和检查配置值。

### Q5: 可以同时使用新旧模块吗？
A: 技术上可以，但不推荐，可能导致功能重复和不一致。

---

## 九、获取帮助

如果在迁移过程中遇到问题：
1. 查看 `OPTIMIZATION_REPORT.md` 了解优化详情
2. 查看模块的文档字符串
3. 运行测试用例验证功能
4. 检查日志输出获取更多信息

---

## 十、总结

迁移到新的统一模块将带来以下好处：
- ✅ 更好的代码组织和维护性
- ✅ 统一的错误处理和日志记录
- ✅ 全面的性能监控和分析
- ✅ 自动化的配置验证
- ✅ 更好的异步支持
- ✅ 更清晰的代码结构

建议分阶段进行迁移，先迁移非关键路径的代码，验证无误后再迁移核心功能。
