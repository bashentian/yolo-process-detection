import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import json

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("TensorRT not available. Install with: pip install tensorrt")

try:
    from onnx import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX not available. Install with: pip install onnx")

from config import ProcessDetectionConfig
from detector import ProcessDetector, Detection


class JetsonModelOptimizer:
    def __init__(self, model_path: str, config: ProcessDetectionConfig):
        self.model_path = Path(model_path)
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"TensorRT Available: {TRT_AVAILABLE}")
    
    def convert_to_onnx(self, output_path: Optional[str] = None,
                       dynamic_batch: bool = False) -> str:
        from ultralytics import YOLO
        
        model = YOLO(str(self.model_path))
        model.model.eval()
        
        if output_path is None:
            output_path = self.model_path.parent / f"{self.model_path.stem}.onnx"
        
        dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
        
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                'images': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        print(f"Converting model to ONNX format...")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output path: {output_path}")
        
        torch.onnx.export(
            model.model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        print(f"ONNX model saved to: {output_path}")
        
        return str(output_path)
    
    def quantize_onnx(self, onnx_path: str, output_path: Optional[str] = None,
                      calibration_data_path: Optional[str] = None,
                      quantization_mode: str = "int8") -> str:
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not available. Install with: pip install onnx")
        
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        onnx_path = Path(onnx_path)
        if output_path is None:
            output_path = onnx_path.parent / f"{onnx_path.stem}_quantized.onnx"
        
        print(f"Quantizing ONNX model...")
        print(f"Mode: {quantization_mode}")
        print(f"Output: {output_path}")
        
        if quantization_mode == "int8":
            quant_type = QuantType.QInt8
        elif quantization_mode == "int16":
            quant_type = QuantType.QInt16
        else:
            quant_type = QuantType.QUInt8
        
        quantized_model = quantize_dynamic(
            str(onnx_path),
            str(output_path),
            weight_type=quant_type
        )
        
        model_size_before = onnx_path.stat().st_size / 1024 / 1024
        model_size_after = Path(output_path).stat().st_size / 1024 / 1024
        
        print(f"Original size: {model_size_before:.2f} MB")
        print(f"Quantized size: {model_size_after:.2f} MB")
        print(f"Compression ratio: {model_size_before / model_size_after:.2f}x")
        
        return str(output_path)
    
    def convert_to_tensorrt(self, onnx_path: str, output_path: Optional[str] = None,
                          fp16: bool = True, int8: bool = False,
                          calibration_data: Optional[List[np.ndarray]] = None,
                          max_batch_size: int = 1,
                          max_workspace_size: int = 1 << 30) -> str:
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT not available on this system")
        
        onnx_path = Path(onnx_path)
        if output_path is None:
            output_path = onnx_path.parent / f"{onnx_path.stem}.engine"
        
        print(f"Converting ONNX to TensorRT engine...")
        print(f"FP16: {fp16}, INT8: {int8}")
        print(f"Max batch size: {max_batch_size}")
        print(f"Max workspace: {max_workspace_size / (1024**3):.2f} GB")
        
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(f"Parser error: {parser.get_error(error)}")
                raise RuntimeError("Failed to parse ONNX model")
        
        config = builder.create_builder_config()
        config.max_workspace_size = max_workspace_size
        
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 mode enabled")
        
        if int8:
            if not builder.platform_has_fast_int8:
                print("Warning: INT8 not supported on this platform, falling back to FP16")
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                config.set_flag(trt.BuilderFlag.INT8)
                print("INT8 mode enabled")
                
                if calibration_data is not None:
                    print("Using calibration data for INT8 quantization")
                    calibrator = self._create_calibrator(calibration_data, config)
                    config.int8_calibrator = calibrator
                else:
                    print("Warning: No calibration data provided, INT8 may not work properly")
        
        config.max_batch_size = max_batch_size
        
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        with open(output_path, 'wb') as f:
            f.write(serialized_engine)
        
        print(f"TensorRT engine saved to: {output_path}")
        
        return str(output_path)
    
    def _create_calibrator(self, calibration_data: List[np.ndarray],
                          config) -> object:
        class JetsonINT8Calibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self, calibration_data, config):
                super().__init__()
                self.calibration_data = calibration_data
                self.config = config
                self.current_index = 0
                
            def get_batch_size(self):
                return len(self.calibration_data[0])
            
            def get_batch(self, names):
                if self.current_index >= len(self.calibration_data):
                    return None
                
                batch = self.calibration_data[self.current_index].astype(np.float32)
                self.current_index += 1
                
                return [batch]
            
            def read_calibration_cache(self):
                return None
            
            def write_calibration_cache(self, cache):
                pass
        
        return JetsonINT8Calibrator(calibration_data, config)
    
    def optimize_for_jetson(self, model_path: str, output_dir: Optional[str] = None,
                          fp16: bool = True, int8: bool = False,
                          quantize_onnx: bool = True) -> Dict[str, str]:
        if output_dir is None:
            output_dir = self.model_path.parent / "jetson_optimized"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Optimizing model for Jetson Nano...")
        print(f"Model: {model_path}")
        print(f"Output directory: {output_dir}")
        
        results = {}
        
        print("\n" + "="*60)
        print("Step 1: Convert to ONNX")
        print("="*60)
        onnx_path = self.convert_to_onnx(model_path, output_dir / "model.onnx")
        results['onnx'] = onnx_path
        
        if quantize_onnx:
            print("\n" + "="*60)
            print("Step 2: Quantize ONNX model")
            print("="*60)
            quantized_path = self.quantize_onnx(onnx_path, output_dir / "model_quantized.onnx")
            results['onnx_quantized'] = quantized_path
            
            onnx_to_convert = quantized_path
        else:
            onnx_to_convert = onnx_path
        
        if TRT_AVAILABLE:
            print("\n" + "="*60)
            print("Step 3: Convert to TensorRT")
            print("="*60)
            engine_path = self.convert_to_tensorrt(
                onnx_to_convert,
                output_dir / "model.engine",
                fp16=fp16,
                int8=int8
            )
            results['tensorrt'] = engine_path
        
        print("\n" + "="*60)
        print("Optimization complete!")
        print("="*60)
        
        self._print_optimization_summary(results)
        
        return results
    
    def _print_optimization_summary(self, results: Dict[str, str]):
        original_size = self.model_path.stat().st_size / 1024 / 1024
        
        print("\nModel Size Comparison:")
        print("-" * 60)
        print(f"Original (PyTorch):  {original_size:.2f} MB")
        
        if 'onnx' in results:
            onnx_size = Path(results['onnx']).stat().st_size / 1024 / 1024
            print(f"ONNX:                {onnx_size:.2f} MB  ({onnx_size/original_size:.2f}x)")
        
        if 'onnx_quantized' in results:
            quantized_size = Path(results['onnx_quantized']).stat().st_size / 1024 / 1024
            print(f"ONNX Quantized:       {quantized_size:.2f} MB  ({quantized_size/original_size:.2f}x)")
        
        if 'tensorrt' in results:
            engine_size = Path(results['tensorrt']).stat().st_size / 1024 / 1024
            print(f"TensorRT Engine:       {engine_size:.2f} MB  ({engine_size/original_size:.2f}x)")
        
        print("-" * 60)


class JetsonInferenceEngine:
    def __init__(self, model_path: str, config: ProcessDetectionConfig,
                 use_tensorrt: bool = True):
        self.config = config
        self.model_path = Path(model_path)
        self.use_tensorrt = use_tensorrt
        
        self.engine = None
        self.context = None
        self.bindings = []
        self.stream = None
        
        if use_tensorrt and TRT_AVAILABLE:
            self._load_tensorrt_engine()
        else:
            self._load_onnx_model()
    
    def _load_tensorrt_engine(self):
        if not self.model_path.suffix == ".engine":
            print(f"Loading TensorRT engine from {self.model_path}")
        else:
            print(f"Converting {self.model_path} to TensorRT...")
            optimizer = JetsonModelOptimizer(self.model_path, self.config)
            engine_path = optimizer.convert_to_tensorrt(
                optimizer.convert_to_onnx(self.model_path)
            )
            self.model_path = Path(engine_path)
        
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        
        with open(self.model_path, 'rb') as f:
            engine_data = f.read()
        
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        print(f"TensorRT engine loaded successfully")
        print(f"Input binding: {self.engine.get_binding_name(0)}")
        print(f"Output binding: {self.engine.get_binding_name(1)}")
    
    def _load_onnx_model(self):
        try:
            import onnxruntime as ort
            print(f"Loading ONNX model from {self.model_path}")
            
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.onnx_session = ort.InferenceSession(
                str(self.model_path),
                providers=providers
            )
            
            print(f"ONNX model loaded with providers: {self.onnx_session.get_providers()}")
        except ImportError:
            print("ONNX Runtime not available, falling back to PyTorch")
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
    
    def infer(self, image: np.ndarray) -> List[Detection]:
        if self.use_tensorrt and self.engine is not None:
            return self._infer_tensorrt(image)
        elif hasattr(self, 'onnx_session'):
            return self._infer_onnx(image)
        else:
            return self._infer_pytorch(image)
    
    def _infer_tensorrt(self, image: np.ndarray) -> List[Detection]:
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        input_shape = self.engine.get_binding_shape(0)
        input_name = self.engine.get_binding_name(0)
        output_name = self.engine.get_binding_name(1)
        
        image_resized = cv2.resize(image, (input_shape[2], input_shape[1]))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_normalized = image_rgb.astype(np.float32) / 255.0
        image_batched = image_normalized.transpose(2, 0, 1)[np.newaxis, :]
        
        input_size = image_batched.nbytes
        output_size = trt.volume(self.engine.get_binding_shape(1))
        
        device_input = cuda.mem_alloc(input_size)
        device_output = cuda.mem_alloc(output_size)
        
        bindings = [int(device_input), int(device_output)]
        
        stream = cuda.Stream()
        
        cuda.memcpy_htod_async(device_input, image_batched, stream)
        self.context.execute_v2(bindings=bindings)
        cuda.memcpy_dtoh_async(self.host_output, device_output, stream)
        stream.synchronize()
        
        device_input.free()
        device_output.free()
        
        detections = self._parse_output(self.host_output, image.shape)
        
        return detections
    
    def _infer_onnx(self, image: np.ndarray) -> List[Detection]:
        input_shape = self.onnx_session.get_inputs()[0].shape
        input_name = self.onnx_session.get_inputs()[0].name
        
        image_resized = cv2.resize(image, (input_shape[2], input_shape[1]))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_normalized = image_rgb.astype(np.float32) / 255.0
        image_batched = image_normalized.transpose(2, 0, 1)[np.newaxis, :]
        
        outputs = self.onnx_session.run(
            None,
            {input_name: image_batched}
        )
        
        detections = self._parse_output(outputs[0], image.shape)
        
        return detections
    
    def _infer_pytorch(self, image: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            image,
            conf=self.config.CONFIDENCE_THRESHOLD,
            iou=self.config.IOU_THRESHOLD,
            device=self.config.DEVICE,
            verbose=False
        )
        
        detections = []
        if len(results) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.config.CLASS_NAMES.get(class_id, f"class_{class_id}")
                
                detection = Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name
                )
                detections.append(detection)
        
        return detections
    
    def _parse_output(self, output: np.ndarray, original_shape: Tuple[int, int, int]) -> List[Detection]:
        detections = []
        
        if len(output.shape) == 3:
            output = output[0]
        
        num_detections = output.shape[0] if len(output.shape) > 0 else 0
        
        for i in range(num_detections):
            if len(output[i]) < 6:
                continue
            
            x1, y1, x2, y2 = output[i][:4]
            confidence = output[i][4]
            class_id = int(output[i][5])
            
            if confidence < self.config.CONFIDENCE_THRESHOLD:
                continue
            
            class_name = self.config.CLASS_NAMES.get(class_id, f"class_{class_id}")
            
            detection = Detection(
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(confidence),
                class_id=class_id,
                class_name=class_name
            )
            detections.append(detection)
        
        return detections
    
    def benchmark(self, num_iterations: int = 100) -> Dict[str, float]:
        import time
        
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        warmup_iterations = 10
        
        print(f"Running benchmark with {num_iterations} iterations...")
        
        for _ in range(warmup_iterations):
            self.infer(test_image)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        start_time = time.time()
        
        for _ in range(num_iterations):
            self.infer(test_image)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        fps = 1.0 / avg_time
        
        results = {
            'avg_inference_time_ms': avg_time * 1000,
            'fps': fps,
            'total_time_s': total_time,
            'iterations': num_iterations
        }
        
        print("\nBenchmark Results:")
        print("-" * 40)
        print(f"Avg inference time: {results['avg_inference_time_ms']:.2f} ms")
        print(f"FPS:                 {results['fps']:.2f}")
        print(f"Total time:          {results['total_time_s']:.2f} s")
        print("-" * 40)
        
        return results


class JetsonDeploymentManager:
    def __init__(self, config: ProcessDetectionConfig):
        self.config = config
        self.optimization_dir = Path("jetson_optimized")
        self.optimization_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_jetson_deployment(self, model_path: str,
                                 calibration_images_dir: Optional[str] = None,
                                 fp16: bool = True,
                                 int8: bool = False) -> Dict:
        print("="*60)
        print("Jetson Nano Deployment Preparation")
        print("="*60)
        
        optimizer = JetsonModelOptimizer(model_path, self.config)
        
        calibration_data = None
        if calibration_images_dir and int8:
            calibration_data = self._load_calibration_data(calibration_images_dir)
        
        results = optimizer.optimize_for_jetson(
            model_path,
            fp16=fp16,
            int8=int8
        )
        
        self._create_deployment_package(results)
        
        return results
    
    def _load_calibration_data(self, images_dir: str, 
                             max_images: int = 100) -> List[np.ndarray]:
        print(f"Loading calibration data from {images_dir}...")
        
        images = list(Path(images_dir).glob("*.jpg")) + \
                 list(Path(images_dir).glob("*.png"))
        
        images = images[:max_images]
        
        calibration_data = []
        for img_path in images:
            img = cv2.imread(str(img_path))
            img_resized = cv2.resize(img, (640, 640))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb.astype(np.float32) / 255.0
            calibration_data.append(img_normalized)
        
        print(f"Loaded {len(calibration_data)} calibration images")
        
        return calibration_data
    
    def _create_deployment_package(self, results: Dict[str, str]):
        print("\nCreating deployment package...")
        
        package_dir = self.optimization_dir / "deployment_package"
        package_dir.mkdir(parents=True, exist_ok=True)
        
        for key, path in results.items():
            if Path(path).exists():
                shutil_copy = __import__('shutil').copy
                shutil_copy(path, package_dir / Path(path).name)
        
        deployment_info = {
            'model_type': 'optimized',
            'target_device': 'NVIDIA Jetson Nano',
            'fp16': True,
            'int8': False,
            'input_size': [640, 640],
            'confidence_threshold': self.config.CONFIDENCE_THRESHOLD,
            'iou_threshold': self.config.IOU_THRESHOLD,
            'class_names': self.config.CLASS_NAMES
        }
        
        with open(package_dir / 'deployment_info.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        print(f"Deployment package created at: {package_dir}")
        print(f"Contents: {list(package_dir.glob('*'))}")


def benchmark_on_jetson(model_path: str, config: ProcessDetectionConfig):
    print("="*60)
    print("Jetson Nano Benchmark")
    print("="*60)
    
    optimizer = JetsonModelOptimizer(model_path, config)
    
    print("\n1. Original Model (PyTorch)")
    print("-" * 40)
    engine_pytorch = JetsonInferenceEngine(model_path, config, use_tensorrt=False)
    results_pytorch = engine_pytorch.benchmark(num_iterations=50)
    
    if TRT_AVAILABLE:
        onnx_path = optimizer.convert_to_onnx(model_path)
        
        print("\n2. ONNX Model")
        print("-" * 40)
        engine_onnx = JetsonInferenceEngine(onnx_path, config, use_tensorrt=False)
        results_onnx = engine_onnx.benchmark(num_iterations=50)
        
        print("\n3. TensorRT Engine (FP16)")
        print("-" * 40)
        trt_path = optimizer.convert_to_tensorrt(onnx_path, fp16=True)
        engine_trt_fp16 = JetsonInferenceEngine(trt_path, config, use_tensorrt=True)
        results_trt_fp16 = engine_trt_fp16.benchmark(num_iterations=50)
        
        print("\n" + "="*60)
        print("Performance Comparison")
        print("="*60)
        print(f"{'Method':<25} {'Time (ms)':<15} {'FPS':<10} {'Speedup':<10}")
        print("-" * 60)
        print(f"{'PyTorch':<25} {results_pytorch['avg_inference_time_ms']:<15.2f} {results_pytorch['fps']:<10.2f} {1.0:<10.2f}")
        print(f"{'ONNX':<25} {results_onnx['avg_inference_time_ms']:<15.2f} {results_onnx['fps']:<10.2f} {results_pytorch['avg_inference_time_ms']/results_onnx['avg_inference_time_ms']:<10.2f}")
        print(f"{'TensorRT FP16':<25} {results_trt_fp16['avg_inference_time_ms']:<15.2f} {results_trt_fp16['fps']:<10.2f} {results_pytorch['avg_inference_time_ms']/results_trt_fp16['avg_inference_time_ms']:<10.2f}")
        print("-" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Jetson Nano Model Optimization')
    parser.add_argument('--model', required=True, help='Path to PyTorch model')
    parser.add_argument('--mode', choices=['optimize', 'benchmark'], default='optimize',
                       help='Operation mode')
    parser.add_argument('--fp16', action='store_true', default=True,
                       help='Enable FP16 optimization')
    parser.add_argument('--int8', action='store_true',
                       help='Enable INT8 quantization')
    parser.add_argument('--calibration', help='Path to calibration images for INT8')
    parser.add_argument('--quantize-onnx', action='store_true',
                       help='Apply ONNX quantization')
    
    args = parser.parse_args()
    
    config = ProcessDetectionConfig()
    
    if args.mode == 'optimize':
        manager = JetsonDeploymentManager(config)
        manager.prepare_jetson_deployment(
            args.model,
            calibration_images_dir=args.calibration,
            fp16=args.fp16,
            int8=args.int8
        )
    elif args.mode == 'benchmark':
        benchmark_on_jetson(args.model, config)
