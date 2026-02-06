import cv2
import numpy as np
import time
from pathlib import Path
import argparse
import sys

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("TensorRT not available, using ONNX/PyTorch")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX Runtime not available, using PyTorch")

from config import ProcessDetectionConfig
from detector import ProcessDetector
from video_processor import VideoProcessor


class JetsonInferenceApp:
    def __init__(self, model_path: str, config: ProcessDetectionConfig):
        self.model_path = Path(model_path)
        self.config = config
        self.model_type = self._detect_model_type()
        
        print(f"加载模型: {self.model_path}")
        print(f"模型类型: {self.model_type}")
        
        self.engine = self._load_model()
        
        self.fps_history = []
        self.frame_count = 0
        self.start_time = time.time()
    
    def _detect_model_type(self) -> str:
        suffix = self.model_path.suffix.lower()
        
        if suffix == ".engine":
            return "TensorRT"
        elif suffix == ".onnx":
            return "ONNX"
        elif suffix == ".pt":
            return "PyTorch"
        else:
            return "Unknown"
    
    def _load_model(self):
        if self.model_type == "TensorRT" and TRT_AVAILABLE:
            return self._load_tensorrt_engine()
        elif self.model_type == "ONNX" and ONNX_AVAILABLE:
            return self._load_onnx_model()
        else:
            return self._load_pytorch_model()
    
    def _load_tensorrt_engine(self):
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            print("PyCUDA not available, trying to load TensorRT engine...")
        
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        
        print("加载TensorRT引擎...")
        with open(self.model_path, 'rb') as f:
            engine_data = f.read()
        
        engine = runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()
        
        print(f"TensorRT引擎加载成功")
        print(f"输入: {engine.get_binding_name(0)}")
        print(f"输出: {engine.get_binding_name(1)}")
        
        return {
            'type': 'tensorrt',
            'engine': engine,
            'context': context,
            'input_name': engine.get_binding_name(0),
            'output_name': engine.get_binding_name(1)
        }
    
    def _load_onnx_model(self):
        print("加载ONNX模型...")
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(str(self.model_path), providers=providers)
        
        print(f"ONNX模型加载成功")
        print(f"提供者: {session.get_providers()}")
        print(f"输入: {session.get_inputs()[0].name}")
        print(f"输出: {session.get_outputs()[0].name}")
        
        return {
            'type': 'onnx',
            'session': session,
            'input_name': session.get_inputs()[0].name,
            'input_shape': session.get_inputs()[0].shape
        }
    
    def _load_pytorch_model(self):
        from ultralytics import YOLO
        
        print("加载PyTorch模型...")
        model = YOLO(str(self.model_path))
        
        print("PyTorch模型加载成功")
        
        return {
            'type': 'pytorch',
            'model': model
        }
    
    def infer(self, frame: np.ndarray) -> dict:
        if self.engine['type'] == 'pytorch':
            return self._infer_pytorch(frame)
        elif self.engine['type'] == 'onnx':
            return self._infer_onnx(frame)
        elif self.engine['type'] == 'tensorrt':
            return self._infer_tensorrt(frame)
    
    def _infer_pytorch(self, frame: np.ndarray) -> dict:
        start_time = time.time()
        
        results = self.engine['model'].predict(
            frame,
            conf=self.config.CONFIDENCE_THRESHOLD,
            iou=self.config.IOU_THRESHOLD,
            device=self.config.DEVICE,
            verbose=False
        )
        
        inference_time = time.time() - start_time
        
        detections = []
        if len(results) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.config.CLASS_NAMES.get(class_id, f"class_{class_id}")
                
                detections.append({
                    'bbox': (float(x1), float(y1), float(x2), float(y2)),
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        return {
            'detections': detections,
            'inference_time': inference_time
        }
    
    def _infer_onnx(self, frame: np.ndarray) -> dict:
        start_time = time.time()
        
        input_shape = self.engine['input_shape']
        input_name = self.engine['input_name']
        
        frame_resized = cv2.resize(frame, (input_shape[2], input_shape[1]))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        frame_batched = frame_normalized.transpose(2, 0, 1)[np.newaxis, :]
        
        outputs = self.engine['session'].run(
            None,
            {input_name: frame_batched}
        )
        
        inference_time = time.time() - start_time
        
        detections = self._parse_outputs(outputs[0])
        
        return {
            'detections': detections,
            'inference_time': inference_time
        }
    
    def _infer_tensorrt(self, frame: np.ndarray) -> dict:
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        start_time = time.time()
        
        context = self.engine['context']
        input_name = self.engine['input_name']
        output_name = self.engine['output_name']
        
        input_shape = context.get_binding_shape(0)
        
        frame_resized = cv2.resize(frame, (input_shape[2], input_shape[1]))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        frame_batched = frame_normalized.transpose(2, 0, 1)[np.newaxis, :]
        
        input_size = frame_batched.nbytes
        output_size = trt.volume(context.get_binding_shape(1))
        
        device_input = cuda.mem_alloc(input_size)
        device_output = cuda.mem_alloc(output_size)
        
        bindings = [int(device_input), int(device_output)]
        
        stream = cuda.Stream()
        
        cuda.memcpy_htod_async(device_input, frame_batched, stream)
        context.execute_v2(bindings=bindings)
        cuda.memcpy_dtoh_async(self.host_output, device_output, stream)
        stream.synchronize()
        
        inference_time = time.time() - start_time
        
        device_input.free()
        device_output.free()
        
        detections = self._parse_outputs(self.host_output)
        
        return {
            'detections': detections,
            'inference_time': inference_time
        }
    
    def _parse_outputs(self, outputs: np.ndarray) -> list:
        detections = []
        
        if len(outputs.shape) == 4 and outputs.shape[1] > 0:
            for i in range(outputs.shape[1]):
                if len(outputs[0][i]) < 6:
                    continue
                
                x1, y1, x2, y2 = outputs[0][i][:4]
                confidence = outputs[0][i][4]
                class_id = int(outputs[0][i][5])
                
                if confidence < self.config.CONFIDENCE_THRESHOLD:
                    continue
                
                class_name = self.config.CLASS_NAMES.get(class_id, f"class_{class_id}")
                
                detections.append({
                    'bbox': (float(x1), float(y1), float(x2), float(y2)),
                    'confidence': float(confidence),
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        return detections
    
    def analyze_process_stage(self, detections: list) -> str:
        if not detections:
            return "idle"
        
        class_counts = {}
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        if class_counts.get("worker", 0) > 0 and class_counts.get("machine", 0) > 0:
            if class_counts.get("product", 0) > 0:
                return "processing"
            elif class_counts.get("material", 0) > 0:
                return "preparation"
        
        if class_counts.get("product", 0) > 0 and class_counts.get("tool", 0) > 0:
            return "assembly"
        
        if class_counts.get("product", 0) > 0 and "worker" in class_counts:
            return "quality_check"
        
        if class_counts.get("product", 0) > 0 and "machine" not in class_counts:
            return "packaging"
        
        return "idle"
    
    def draw_detections(self, frame: np.ndarray, detections: list, 
                     stage: str, fps: float) -> np.ndarray:
        annotated = frame.copy()
        
        colors = {
            'worker': (0, 255, 0),
            'machine': (255, 0, 0),
            'product': (0, 0, 255),
            'tool': (255, 255, 0),
            'material': (255, 0, 255)
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            color = colors.get(class_name, (255, 255, 255))
            
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(annotated, 
                        (int(x1), int(y1) - label_size[1] - 10),
                        (int(x1) + label_size[0], int(y1)),
                        color, -1)
            
            cv2.putText(annotated, label,
                       (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        stage_text = f"Stage: {stage}"
        cv2.putText(annotated, stage_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        fps_text = f"FPS: {fps:.1f} | Model: {self.model_type}"
        cv2.putText(annotated, fps_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        info_text = f"Detections: {len(detections)}"
        cv2.putText(annotated, info_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return annotated
    
    def run(self, camera_index: int = 0, display_size: tuple = (1280, 720)):
        print(f"\n启动实时检测...")
        print(f"摄像头: {camera_index}")
        print(f"显示尺寸: {display_size}")
        print("按 'q' 退出\n")
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"无法打开摄像头 {camera_index}")
            return
        
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"摄像头参数: {width}x{height} @ {actual_fps:.2f} FPS")
        
        self.start_time = time.time()
        self.frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("无法读取摄像头帧")
                    break
                
                result = self.infer(frame)
                detections = result['detections']
                inference_time = result['inference_time']
                stage = self.analyze_process_stage(detections)
                
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                current_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                
                annotated_frame = self.draw_detections(
                    frame, detections, stage, current_fps
                )
                
                display_frame = cv2.resize(annotated_frame, display_size)
                cv2.imshow('Jetson Detection', display_frame)
                
                print(f"\rFrame: {self.frame_count} | Stage: {stage} | "
                      f"FPS: {current_fps:.1f} | Inf: {inference_time*1000:.1f}ms", end="")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n用户退出")
                    break
        
        except KeyboardInterrupt:
            print("\n程序被中断")
        except Exception as e:
            print(f"\n发生错误: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            total_time = time.time() - self.start_time
            avg_fps = self.frame_count / total_time if total_time > 0 else 0
            
            print("\n" + "="*60)
            print("统计信息")
            print("="*60)
            print(f"总帧数: {self.frame_count}")
            print(f"总时间: {total_time:.2f}s")
            print(f"平均FPS: {avg_fps:.2f}")
            print(f"模型类型: {self.model_type}")
            print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Jetson Nano 实时检测')
    parser.add_argument('--model', default='jetson_optimized/model.engine',
                       help='模型路径 (.engine/.onnx/.pt)')
    parser.add_argument('--camera', type=int, default=0,
                       help='摄像头索引')
    parser.add_argument('--width', type=int, default=1280,
                       help='显示宽度')
    parser.add_argument('--height', type=int, default=720,
                       help='显示高度')
    
    args = parser.parse_args()
    
    config = ProcessDetectionConfig()
    config.DEVICE = "cuda"
    
    if not Path(args.model).exists():
        print(f"错误: 模型文件不存在: {args.model}")
        print("请先运行模型优化: python jetson_deployment.py --model best.pt --mode optimize")
        sys.exit(1)
    
    app = JetsonInferenceApp(args.model, config)
    app.run(
        camera_index=args.camera,
        display_size=(args.width, args.height)
    )


if __name__ == "__main__":
    main()
