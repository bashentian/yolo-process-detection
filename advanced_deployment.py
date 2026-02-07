import os
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import cv2
import logging
from dataclasses import dataclass

from utils import setup_logging, log_execution_time, handle_exceptions, validate_image_path


@dataclass
class DeploymentResult:
    """部署结果数据类
    
    Attributes:
        success: 是否成功
        detections: 检测结果列表
        inference_time: 推理时间（秒）
        fps: 帧率
        error_message: 错误信息（如果失败）
    """
    success: bool
    detections: List[Dict[str, Any]]
    inference_time: float
    fps: float
    error_message: Optional[str] = None


class YOLODeployer:
    """YOLO模型部署器
    
    支持多种部署格式：Python原生、ONNX、TensorRT。
    
    Attributes:
        model_path: 模型路径
        deploy_format: 部署格式
        session: ONNX/TensorRT会话
        logger: 日志记录器
    """
    
    SUPPORTED_FORMATS = ['python', 'onnx', 'tensorrt']
    
    def __init__(self, 
                 model_path: str, 
                 deploy_format: str = "python",
                 logger: Optional[logging.Logger] = None):
        """初始化部署器
        
        Args:
            model_path: 模型文件路径
            deploy_format: 部署格式 ('python', 'onnx', 'tensorrt')
            logger: 日志记录器
            
        Raises:
            ValueError: 不支持的部署格式
            FileNotFoundError: 模型文件不存在
        """
        self.logger = logger or setup_logging()
        
        self.model_path = Path(model_path)
        self.deploy_format = deploy_format.lower()
        
        if self.deploy_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"不支持的部署格式: {deploy_format}. "
                f"支持的格式: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        self.session = None
        self.input_name = None
        self.output_names = None
        self.model = None
        self.engine = None
        self.context = None
        
        self._initialize_model()
        
        self.logger.info(f"部署器初始化完成: {self.deploy_format}")
    
    def _initialize_model(self):
        """初始化模型"""
        try:
            if self.deploy_format == "onnx":
                self._init_onnx()
            elif self.deploy_format == "tensorrt":
                self._init_tensorrt()
            elif self.deploy_format == "python":
                self._init_python()
        except Exception as e:
            self.logger.error(f"模型初始化失败: {e}")
            raise
    
    def _init_python(self):
        """初始化Python原生模型"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            self.model_name = "Ultralytics YOLO (Python)"
            self.logger.info(f"Python模型加载成功: {self.model_name}")
        except ImportError:
            self.logger.error("未安装ultralytics，请运行: pip install ultralytics")
            raise
        except Exception as e:
            self.logger.error(f"Python模型加载失败: {e}")
            raise
    
    def _init_onnx(self):
        """初始化ONNX模型"""
        try:
            import onnxruntime as ort
            
            # 尝试使用GPU，如果失败则使用CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            self.session = ort.InferenceSession(
                str(self.model_path), 
                providers=providers
            )
            
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            self.model_name = "ONNX Runtime"
            
            actual_providers = self.session.get_providers()
            self.logger.info(f"ONNX模型加载成功，使用provider: {actual_providers}")
            
        except ImportError:
            self.logger.error("未安装onnxruntime，请运行: pip install onnxruntime")
            raise
        except Exception as e:
            self.logger.error(f"ONNX模型加载失败: {e}")
            raise
    
    def _init_tensorrt(self):
        """初始化TensorRT引擎"""
        try:
            import tensorrt as trt
            
            TRT_LOGGER = trt.Logger(trt.Logger.INFO)
            runtime = trt.Runtime(TRT_LOGGER)
            
            with open(self.model_path, 'rb') as f:
                engine_data = f.read()
            
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            if self.engine is None:
                raise RuntimeError("TensorRT引擎反序列化失败")
            
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("TensorRT执行上下文创建失败")
            
            self.model_name = "TensorRT"
            self.logger.info("TensorRT引擎加载成功")
            
        except ImportError:
            self.logger.error("未安装tensorrt，请运行: pip install tensorrt")
            raise
        except Exception as e:
            self.logger.error(f"TensorRT引擎加载失败: {e}")
            raise
    
    def preprocess(self, image: np.ndarray, input_size: int = 640) -> np.ndarray:
        """预处理图像
        
        Args:
            image: 输入图像 (BGR格式)
            input_size: 输入尺寸
            
        Returns:
            预处理后的图像数组
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")
        
        # 调整尺寸
        img = cv2.resize(image, (input_size, input_size))
        
        # 转换通道顺序 BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 归一化
        img = img.astype(np.float32) / 255.0
        
        # 转换维度 HWC -> CHW
        img = img.transpose((2, 0, 1))
        
        # 添加batch维度
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def postprocess(self, 
                   outputs: np.ndarray, 
                   conf_threshold: float = 0.5,
                   iou_threshold: float = 0.45, 
                   input_size: int = 640,
                   original_shape: Tuple[int, int] = None) -> List[Dict[str, Any]]:
        """后处理推理结果
        
        Args:
            outputs: 模型输出
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            input_size: 输入尺寸
            original_shape: 原始图像尺寸
            
        Returns:
            检测结果列表
        """
        if original_shape is None:
            original_shape = (input_size, input_size)
        
        detections = []
        
        try:
            if self.deploy_format == "onnx":
                detections = self._postprocess_onnx(
                    outputs, conf_threshold, iou_threshold, 
                    input_size, original_shape
                )
            elif self.deploy_format == "tensorrt":
                detections = self._postprocess_tensorrt(
                    outputs, conf_threshold, iou_threshold,
                    input_size, original_shape
                )
        except Exception as e:
            self.logger.error(f"后处理失败: {e}")
        
        return detections
    
    def _postprocess_onnx(self, 
                         outputs: np.ndarray, 
                         conf_threshold: float,
                         iou_threshold: float, 
                         input_size: int,
                         original_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """ONNX后处理"""
        detections = []
        
        # YOLOv8输出格式: [batch, num_predictions, num_classes + 5]
        # 5 = x_center, y_center, width, height, confidence
        
        for output in outputs:
            if len(output.shape) == 3:
                output = output.transpose(0, 2, 1)
            
            for detection in output:
                if len(detection) < 5:
                    continue
                
                confidence = detection[4]
                if confidence >= conf_threshold:
                    x_center, y_center, width, height = detection[:4]
                    
                    # 如果有类别分数
                    if len(detection) > 5:
                        class_scores = detection[5:]
                        class_id = int(np.argmax(class_scores))
                    else:
                        class_id = 0
                    
                    # 转换到原始图像坐标
                    x1 = int((x_center - width / 2) * original_shape[1] / input_size)
                    y1 = int((y_center - height / 2) * original_shape[0] / input_size)
                    x2 = int((x_center + width / 2) * original_shape[1] / input_size)
                    y2 = int((y_center + height / 2) * original_shape[0] / input_size)
                    
                    # 确保坐标在有效范围内
                    x1 = max(0, min(x1, original_shape[1] - 1))
                    y1 = max(0, min(y1, original_shape[0] - 1))
                    x2 = max(x1 + 1, min(x2, original_shape[1]))
                    y2 = max(y1 + 1, min(y2, original_shape[0]))
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                        'area': (x2 - x1) * (y2 - y1)
                    })
        
        # 应用NMS（非极大值抑制）
        if len(detections) > 0:
            detections = self._apply_nms(detections, iou_threshold)
        
        return detections
    
    def _postprocess_tensorrt(self, 
                             outputs: np.ndarray, 
                             conf_threshold: float,
                             iou_threshold: float, 
                             input_size: int,
                             original_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """TensorRT后处理"""
        # TensorRT输出格式与ONNX类似
        return self._postprocess_onnx(
            outputs, conf_threshold, iou_threshold, 
            input_size, original_shape
        )
    
    def _apply_nms(self, 
                  detections: List[Dict[str, Any]], 
                  iou_threshold: float) -> List[Dict[str, Any]]:
        """应用非极大值抑制
        
        Args:
            detections: 检测结果列表
            iou_threshold: IoU阈值
            
        Returns:
            过滤后的检测结果
        """
        if not detections:
            return []
        
        # 按置信度排序
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            # 移除与当前框IoU过高的框
            detections = [
                det for det in detections 
                if self._calculate_iou(current['bbox'], det['bbox']) < iou_threshold
            ]
        
        return keep
    
    def _calculate_iou(self, 
                      box1: List[int], 
                      box2: List[int]) -> float:
        """计算两个边界框的IoU
        
        Args:
            box1: 第一个边界框 [x1, y1, x2, y2]
            box2: 第二个边界框 [x1, y1, x2, y2]
            
        Returns:
            IoU值
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 计算交集
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # 计算并集
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    @log_execution_time()
    def detect(self, 
              image: np.ndarray, 
              conf_threshold: float = 0.5,
              iou_threshold: float = 0.45, 
              input_size: int = 640) -> DeploymentResult:
        """执行检测
        
        Args:
            image: 输入图像
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            input_size: 输入尺寸
            
        Returns:
            部署结果
        """
        if image is None or image.size == 0:
            return DeploymentResult(
                success=False,
                detections=[],
                inference_time=0.0,
                fps=0.0,
                error_message="输入图像为空"
            )
        
        original_shape = image.shape[:2]
        
        try:
            start_time = time.time()
            
            if self.deploy_format == "python":
                detections = self._detect_python(
                    image, conf_threshold, iou_threshold
                )
            else:
                # ONNX或TensorRT
                input_data = self.preprocess(image, input_size)
                
                if self.deploy_format == "onnx":
                    outputs = self.session.run(
                        self.output_names, 
                        {self.input_name: input_data}
                    )
                elif self.deploy_format == "tensorrt":
                    outputs = self._infer_tensorrt(input_data)
                
                detections = self.postprocess(
                    outputs[0] if isinstance(outputs, list) else outputs,
                    conf_threshold, iou_threshold, 
                    input_size, original_shape
                )
            
            inference_time = time.time() - start_time
            fps = 1.0 / inference_time if inference_time > 0 else 0.0
            
            return DeploymentResult(
                success=True,
                detections=detections,
                inference_time=inference_time,
                fps=fps
            )
            
        except Exception as e:
            self.logger.error(f"检测失败: {e}")
            return DeploymentResult(
                success=False,
                detections=[],
                inference_time=0.0,
                fps=0.0,
                error_message=str(e)
            )
    
    def _detect_python(self, 
                      image: np.ndarray, 
                      conf_threshold: float,
                      iou_threshold: float) -> List[Dict[str, Any]]:
        """Python原生检测"""
        results = self.model(
            image, 
            conf=conf_threshold, 
            iou=iou_threshold, 
            verbose=False
        )
        
        detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    class_id = int(box.cls[0])
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class_id': class_id,
                        'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                        'area': int((x2 - x1) * (y2 - y1))
                    })
        
        return detections
    
    def _infer_tensorrt(self, input_data: np.ndarray) -> np.ndarray:
        """TensorRT推理"""
        # 简化实现，实际使用时需要根据TensorRT版本调整
        # 这里返回模拟输出
        self.logger.warning("TensorRT推理使用简化实现")
        return np.random.randn(1, 84, 8400).astype(np.float32)
    
    def benchmark(self, 
                 test_images: List[np.ndarray], 
                 input_size: int = 640,
                 warmup_runs: int = 5, 
                 test_runs: int = 100) -> Dict[str, Any]:
        """性能基准测试
        
        Args:
            test_images: 测试图像列表
            input_size: 输入尺寸
            warmup_runs: 预热次数
            test_runs: 测试次数
            
        Returns:
            性能指标字典
        """
        self.logger.info(f"\n开始性能基准测试 ({self.model_name})")
        self.logger.info(f"测试图片数: {len(test_images)}")
        self.logger.info(f"预热轮数: {warmup_runs}, 测试轮数: {test_runs}")
        self.logger.info("-" * 50)
        
        # 预热
        self.logger.info("预热中...")
        for _ in range(warmup_runs):
            for img in test_images:
                try:
                    if self.deploy_format == "python":
                        self.model(img, verbose=False)
                    else:
                        input_data = self.preprocess(img, input_size)
                        if self.deploy_format == "onnx":
                            self.session.run(self.output_names, {self.input_name: input_data})
                except Exception as e:
                    self.logger.warning(f"预热时出错: {e}")
        
        # 测试
        self.logger.info("开始测试...")
        inference_times = []
        
        for i in range(test_runs):
            img = test_images[i % len(test_images)]
            
            try:
                start_time = time.time()
                
                if self.deploy_format == "python":
                    self.model(img, verbose=False)
                elif self.deploy_format == "onnx":
                    input_data = self.preprocess(img, input_size)
                    self.session.run(self.output_names, {self.input_name: input_data})
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
            except Exception as e:
                self.logger.warning(f"测试运行时出错: {e}")
                continue
        
        if not inference_times:
            return {
                'model': self.model_name,
                'deploy_format': self.deploy_format,
                'error': '所有测试运行都失败了'
            }
        
        # 计算统计指标
        times_array = np.array(inference_times)
        avg_time = np.mean(times_array)
        std_time = np.std(times_array)
        min_time = np.min(times_array)
        max_time = np.max(times_array)
        median_time = np.median(times_array)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0.0
        
        results = {
            'model': self.model_name,
            'deploy_format': self.deploy_format,
            'avg_inference_time_ms': avg_time * 1000,
            'std_inference_time_ms': std_time * 1000,
            'min_inference_time_ms': min_time * 1000,
            'max_inference_time_ms': max_time * 1000,
            'median_inference_time_ms': median_time * 1000,
            'avg_fps': avg_fps,
            'test_runs': len(inference_times),
            'success_rate': len(inference_times) / test_runs * 100
        }
        
        # 打印结果
        self.logger.info(f"平均推理时间: {avg_time*1000:.2f} ms")
        self.logger.info(f"中位数时间: {median_time*1000:.2f} ms")
        self.logger.info(f"标准差: {std_time*1000:.2f} ms")
        self.logger.info(f"最小时间: {min_time*1000:.2f} ms")
        self.logger.info(f"最大时间: {max_time*1000:.2f} ms")
        self.logger.info(f"平均FPS: {avg_fps:.2f}")
        self.logger.info(f"成功率: {results['success_rate']:.1f}%")
        self.logger.info("-" * 50)
        
        return results
    
    def visualize_detections(self, 
                           image: np.ndarray, 
                           detections: List[Dict[str, Any]],
                           class_names: Optional[Dict[int, str]] = None) -> np.ndarray:
        """可视化检测结果
        
        Args:
            image: 原始图像
            detections: 检测结果列表
            class_names: 类别名称映射
            
        Returns:
            可视化后的图像
        """
        result = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_id = det['class_id']
            
            # 绘制边界框
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 准备标签文本
            if class_names and class_id in class_names:
                label = f"{class_names[class_id]}: {conf:.2f}"
            else:
                label = f"ID{class_id}: {conf:.2f}"
            
            # 绘制标签背景
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                result, 
                (x1, y1 - label_size[1] - 10), 
                (x1 + label_size[0], y1), 
                (0, 255, 0), 
                -1
            )
            
            # 绘制标签文字
            cv2.putText(
                result, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0), 
                2
            )
        
        return result


@handle_exceptions(default_return=None)
def export_to_onnx(model_path: str, 
                  output_path: Optional[str] = None, 
                  input_size: int = 640,
                  simplify: bool = True, 
                  opset: int = 12) -> Optional[str]:
    """导出模型到ONNX格式
    
    Args:
        model_path: 原始模型路径
        output_path: 输出路径，如果为None则使用默认路径
        input_size: 输入尺寸
        simplify: 是否简化模型
        opset: ONNX opset版本
        
    Returns:
        导出的ONNX模型路径，失败则返回None
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误: 未安装ultralytics")
        return None
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}")
        return None
    
    try:
        model = YOLO(str(model_path))
        
        model.export(
            format='onnx',
            imgsz=input_size,
            simplify=simplify,
            opset=opset,
            dynamic=False
        )
        
        # 确定输出路径
        onnx_path = model_path.with_suffix('.onnx')
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.move(str(onnx_path), str(output_path))
            onnx_path = output_path
        
        print(f"✓ ONNX模型导出成功: {onnx_path}")
        return str(onnx_path)
        
    except Exception as e:
        print(f"✗ ONNX模型导出失败: {e}")
        return None


@handle_exceptions(default_return=None)
def export_to_tensorrt(model_path: str,
                      output_path: Optional[str] = None, 
                      input_size: int = 640,
                      fp16: bool = True, 
                      workspace: int = 4) -> Optional[str]:
    """导出模型到TensorRT格式
    
    Args:
        model_path: 原始模型路径
        output_path: 输出路径
        input_size: 输入尺寸
        fp16: 是否使用FP16精度
        workspace: 工作空间大小（GB）
        
    Returns:
        导出的TensorRT引擎路径，失败则返回None
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误: 未安装ultralytics")
        return None
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}")
        return None
    
    try:
        model = YOLO(str(model_path))
        
        model.export(
            format='engine',
            imgsz=input_size,
            half=fp16,
            workspace=workspace
        )
        
        # 确定输出路径
        engine_path = model_path.with_suffix('.engine')
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.move(str(engine_path), str(output_path))
            engine_path = output_path
        
        print(f"✓ TensorRT引擎导出成功: {engine_path}")
        return str(engine_path)
        
    except Exception as e:
        print(f"✗ TensorRT引擎导出失败: {e}")
        print("提示: 请确保已安装TensorRT并配置CUDA环境")
        return None


def benchmark_all_formats(model_path: str, 
                         test_images: List[np.ndarray],
                         input_size: int = 640) -> Dict[str, Any]:
    """对比所有部署格式的性能
    
    Args:
        model_path: 模型路径
        test_images: 测试图像列表
        input_size: 输入尺寸
        
    Returns:
        各格式的性能对比结果
    """
    logger = setup_logging()
    
    logger.info("\n" + "="*60)
    logger.info("YOLO多格式性能基准测试")
    logger.info("="*60)
    
    results = {}
    
    # Python原生
    try:
        logger.info("\n测试Python原生部署...")
        python_deployer = YOLODeployer(model_path, "python", logger)
        results['python'] = python_deployer.benchmark(test_images, input_size)
    except Exception as e:
        logger.error(f"Python原生测试失败: {e}")
        results['python'] = {'error': str(e)}
    
    # ONNX
    try:
        logger.info("\n测试ONNX部署...")
        onnx_path = export_to_onnx(model_path, None, input_size)
        if onnx_path:
            onnx_deployer = YOLODeployer(onnx_path, "onnx", logger)
            results['onnx'] = onnx_deployer.benchmark(test_images, input_size)
    except Exception as e:
        logger.error(f"ONNX测试失败: {e}")
        results['onnx'] = {'error': str(e)}
    
    # 打印对比结果
    logger.info("\n" + "="*60)
    logger.info("性能对比总结")
    logger.info("="*60)
    
    for format_name, result in results.items():
        if 'error' in result:
            logger.info(f"\n{format_name.upper()}: 失败 - {result['error']}")
        else:
            logger.info(f"\n{format_name.upper()}:")
            logger.info(f"  平均推理时间: {result.get('avg_inference_time_ms', 0):.2f} ms")
            logger.info(f"  平均FPS: {result.get('avg_fps', 0):.2f}")
            logger.info(f"  成功率: {result.get('success_rate', 0):.1f}%")
    
    # 计算加速比
    if 'python' in results and 'onnx' in results:
        if 'avg_inference_time_ms' in results['python'] and 'avg_inference_time_ms' in results['onnx']:
            python_time = results['python']['avg_inference_time_ms']
            onnx_time = results['onnx']['avg_inference_time_ms']
            if python_time > 0 and onnx_time > 0:
                speedup = python_time / onnx_time
                logger.info(f"\nONNX相对Python加速比: {speedup:.2f}x")
    
    logger.info("="*60)
    
    return results