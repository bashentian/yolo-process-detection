import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

from utils import setup_logging, log_execution_time, handle_exceptions, validate_image_path


@dataclass
class SubPixelDetection:
    """亚像素级检测结果数据类
    
    Attributes:
        class_id: 类别ID
        class_name: 类别名称
        confidence: 置信度
        bbox: 边界框 (x, y, w, h) 亚像素级坐标
        center: 中心点 (x, y) 亚像素级坐标
        area: 面积
    """
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    area: float


class SubPixelDetector:
    """亚像素级检测器
    
    实现基于YOLO的亚像素级目标检测，支持显微图像增强和微缺陷模拟。
    
    Attributes:
        model: YOLO模型实例
        confidence_threshold: 置信度阈值
        iou_threshold: IoU阈值
        class_names: 类别名称映射
        logger: 日志记录器
    """
    
    def __init__(self, 
                 model_path: str, 
                 confidence_threshold: float = 0.5, 
                 iou_threshold: float = 0.45,
                 logger: Optional[logging.Logger] = None):
        """初始化亚像素级检测器
        
        Args:
            model_path: YOLO模型路径
            confidence_threshold: 置信度阈值，默认0.5
            iou_threshold: IoU阈值，默认0.45
            logger: 日志记录器，如果为None则创建默认logger
            
        Raises:
            FileNotFoundError: 模型文件不存在
            ImportError: ultralytics未安装
        """
        self.logger = logger or setup_logging()
        
        try:
            from ultralytics import YOLO
        except ImportError:
            self.logger.error("未安装ultralytics，请运行: pip install ultralytics")
            raise
        
        model_path = Path(model_path)
        if not model_path.exists():
            self.logger.warning(f"模型文件不存在: {model_path}，将尝试下载预训练模型")
        
        try:
            self.model = YOLO(str(model_path))
            self.logger.info(f"模型加载成功: {model_path}")
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise
        
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.class_names = self.model.names
        
        self.logger.info(f"检测器初始化完成: confidence={confidence_threshold}, iou={iou_threshold}")
        
    @log_execution_time()
    def detect_frame(self, frame: np.ndarray) -> Tuple[List[SubPixelDetection], np.ndarray]:
        """检测单帧图像
        
        Args:
            frame: 输入图像 (BGR格式)
            
        Returns:
            detections: 检测结果列表
            result_frame: 可视化结果图像
            
        Raises:
            ValueError: 输入图像格式错误
        """
        if frame is None or frame.size == 0:
            raise ValueError("输入图像为空或无效")
        
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(f"输入图像格式错误，期望3通道BGR图像，实际: {frame.shape}")
        
        try:
            results = self.model(
                frame, 
                conf=self.confidence_threshold, 
                iou=self.iou_threshold, 
                verbose=False
            )
        except Exception as e:
            self.logger.error(f"模型推理失败: {e}")
            raise
        
        detections = []
        result_frame = frame.copy()
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.cpu().numpy()
                
                for box in boxes:
                    try:
                        x1, y1, x2, y2 = box.xyxy[0]
                        conf = box.conf[0]
                        class_id = int(box.cls[0])
                        class_name = self.class_names.get(class_id, f"class_{class_id}")
                        
                        center_x, center_y = self._get_subpixel_center(
                            frame, int(x1), int(y1), int(x2), int(y2)
                        )
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        
                        detection = SubPixelDetection(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=float(conf),
                            bbox=(
                                float(center_x - width/2), 
                                float(center_y - height/2), 
                                float(width), 
                                float(height)
                            ),
                            center=(float(center_x), float(center_y)),
                            area=float(area)
                        )
                        detections.append(detection)
                        
                        # 绘制结果
                        cv2.rectangle(
                            result_frame, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (0, 255, 0), 
                            2
                        )
                        cv2.circle(
                            result_frame, 
                            (int(center_x), int(center_y)), 
                            3, 
                            (0, 0, 255), 
                            -1
                        )
                        cv2.putText(
                            result_frame, 
                            f"{class_name}: {conf:.2f}", 
                            (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (0, 255, 0), 
                            2
                        )
                    except Exception as e:
                        self.logger.warning(f"处理单个检测结果时出错: {e}")
                        continue
        
        self.logger.debug(f"检测到 {len(detections)} 个目标")
        return detections, result_frame
    
    def _get_subpixel_center(self, 
                            image: np.ndarray, 
                            x1: int, 
                            y1: int, 
                            x2: int, 
                            y2: int) -> Tuple[float, float]:
        """计算亚像素级中心点
        
        使用Shi-Tomasi角点检测和cornerSubPix实现亚像素级精度。
        
        Args:
            image: 输入图像
            x1, y1: 边界框左上角坐标
            x2, y2: 边界框右下角坐标
            
        Returns:
            (center_x, center_y): 亚像素级中心点坐标
        """
        # 确保坐标在有效范围内
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            self.logger.warning("ROI为空，使用像素级中心")
            return ((x1 + x2) / 2, (y1 + y2) / 2)
        
        try:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            self.logger.warning("ROI转换灰度图失败，使用像素级中心")
            return ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Shi-Tomasi角点检测
        try:
            corners = cv2.goodFeaturesToTrack(
                gray_roi,
                maxCorners=10,
                qualityLevel=0.01,
                minDistance=10,
                blockSize=3
            )
        except cv2.error:
            corners = None
        
        if corners is not None and len(corners) > 0:
            try:
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                    30, 
                    0.001
                )
                corners = cv2.cornerSubPix(
                    gray_roi, 
                    corners, 
                    (5, 5), 
                    (-1, -1), 
                    criteria
                )
                
                avg_x = np.mean(corners[:, 0, 0])
                avg_y = np.mean(corners[:, 0, 1])
                return (x1 + avg_x, y1 + avg_y)
            except cv2.error:
                self.logger.debug("cornerSubPix失败，使用角点均值")
                avg_x = np.mean(corners[:, 0, 0])
                avg_y = np.mean(corners[:, 0, 1])
                return (x1 + avg_x, y1 + avg_y)
        
        # 备用方案：使用图像矩
        try:
            gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
            M = cv2.moments(gray_roi)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                return (x1 + cx, y1 + cy)
        except Exception as e:
            self.logger.debug(f"图像矩计算失败: {e}")
        
        # 最终备用：像素级中心
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @handle_exceptions(default_return=None)
    def enhance_microscopic_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """增强显微图像
        
        使用CLAHE和锐化增强缺陷特征。
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            增强后的图像，如果失败则返回None
        """
        if image is None or image.size == 0:
            self.logger.error("输入图像为空")
            return None
        
        try:
            # 转换到LAB颜色空间
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE增强
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # 合并通道
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # 锐化
            sharpening_kernel = np.array([
                [-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]
            ])
            enhanced = cv2.filter2D(enhanced, -1, sharpening_kernel)
            
            self.logger.debug("显微图像增强完成")
            return enhanced
            
        except cv2.error as e:
            self.logger.error(f"图像增强失败: {e}")
            return None
    
    def simulate_micro_defects(self, 
                              image: np.ndarray, 
                              num_defects: int = 5, 
                              defect_size_range: Tuple[int, int] = (5, 20),
                              seed: Optional[int] = None) -> np.ndarray:
        """模拟微米级缺陷
        
        在图像上添加模拟的划痕、斑点、裂纹等缺陷。
        
        Args:
            image: 输入图像
            num_defects: 缺陷数量
            defect_size_range: 缺陷尺寸范围 (min, max)
            seed: 随机种子，用于可重复性
            
        Returns:
            添加了缺陷的图像
        """
        if image is None or image.size == 0:
            self.logger.error("输入图像为空")
            return image
        
        if seed is not None:
            np.random.seed(seed)
        
        height, width = image.shape[:2]
        result = image.copy()
        
        defect_count = 0
        max_attempts = num_defects * 3  # 防止无限循环
        attempts = 0
        
        while defect_count < num_defects and attempts < max_attempts:
            attempts += 1
            
            try:
                # 随机位置
                x = np.random.randint(0, max(1, width - defect_size_range[1]))
                y = np.random.randint(0, max(1, height - defect_size_range[1]))
                size = np.random.randint(defect_size_range[0], defect_size_range[1])
                
                # 随机缺陷类型
                defect_type = np.random.choice(['scratch', 'spot', 'crack'])
                
                if defect_type == 'scratch':
                    length = np.random.randint(size, size * 3)
                    angle = np.random.uniform(0, np.pi)
                    x2 = x + int(length * np.cos(angle))
                    y2 = y + int(length * np.sin(angle))
                    
                    # 确保终点在图像内
                    x2 = max(0, min(x2, width - 1))
                    y2 = max(0, min(y2, height - 1))
                    
                    cv2.line(result, (x, y), (x2, y2), (50, 50, 50), 1)
                    defect_count += 1
                    
                elif defect_type == 'spot':
                    radius = np.random.randint(1, max(2, size // 2))
                    cv2.circle(result, (x, y), radius, (30, 30, 30), -1)
                    defect_count += 1
                    
                elif defect_type == 'crack':
                    for i in range(3):
                        crack_x = x + np.random.randint(-5, 5)
                        crack_y = y + np.random.randint(-5, 5)
                        crack_length = np.random.randint(size, size * 2)
                        crack_angle = np.random.uniform(0, np.pi)
                        crack_x2 = crack_x + int(crack_length * np.cos(crack_angle))
                        crack_y2 = crack_y + int(crack_length * np.sin(crack_angle))
                        
                        # 确保在图像内
                        crack_x2 = max(0, min(crack_x2, width - 1))
                        crack_y2 = max(0, min(crack_y2, height - 1))
                        
                        cv2.line(result, (crack_x, crack_y), (crack_x2, crack_y2), (40, 40, 40), 1)
                    defect_count += 1
                    
            except Exception as e:
                self.logger.warning(f"生成缺陷时出错: {e}")
                continue
        
        self.logger.info(f"成功生成 {defect_count}/{num_defects} 个缺陷")
        return result
    
    @log_execution_time()
    def process_image(self, 
                     image_path: str, 
                     enhance: bool = True,
                     output_path: Optional[str] = None) -> Tuple[List[SubPixelDetection], np.ndarray]:
        """处理单张图像
        
        完整的图像处理流水线：加载 -> 增强 -> 检测 -> 保存结果。
        
        Args:
            image_path: 输入图像路径
            enhance: 是否进行图像增强
            output_path: 输出图像路径，如果为None则不保存
            
        Returns:
            detections: 检测结果列表
            result_image: 结果图像
            
        Raises:
            FileNotFoundError: 图像文件不存在
            ValueError: 图像格式无效
        """
        image_path = Path(image_path)
        
        if not validate_image_path(image_path):
            raise FileNotFoundError(f"图像文件不存在或格式无效: {image_path}")
        
        # 加载图像
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        self.logger.info(f"处理图像: {image_path}, 尺寸: {image.shape}")
        
        # 图像增强
        if enhance:
            enhanced = self.enhance_microscopic_image(image)
            if enhanced is not None:
                image = enhanced
                self.logger.info("已应用图像增强")
        
        # 检测
        detections, result_image = self.detect_frame(image)
        
        # 保存结果
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), result_image)
            self.logger.info(f"结果已保存: {output_path}")
        
        return detections, result_image