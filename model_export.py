import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
import cv2
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class YOLOModelExporter:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.export_history = []
    
    def export_to_onnx(self, output_path: str = None, 
                       input_size: int = 640, 
                       batch_size: int = 1,
                       simplify: bool = True,
                       opset: int = 12) -> str:
        if output_path is None:
            output_path = self.model_path.replace('.pt', '.onnx')
        
        try:
            self.model.export(
                format='onnx',
                imgsz=input_size,
                batch=batch_size,
                simplify=simplify,
                opset=opset,
                dynamic=False
            )
            
            print(f"ONNX model exported to: {output_path}")
            
            self._validate_onnx_model(output_path)
            
            self.export_history.append({
                'format': 'onnx',
                'path': output_path,
                'input_size': input_size,
                'batch_size': batch_size,
                'timestamp': datetime.now().isoformat()
            })
            
            return output_path
            
        except Exception as e:
            print(f"Error exporting to ONNX: {e}")
            raise
    
    def _validate_onnx_model(self, onnx_path: str) -> bool:
        try:
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            
            input_name = model.graph.input[0].name
            output_names = [output.name for output in model.graph.output]
            
            print(f"ONNX model validation successful!")
            print(f"Input: {input_name}")
            print(f"Outputs: {output_names}")
            
            return True
            
        except Exception as e:
            print(f"ONNX model validation failed: {e}")
            return False
    
    def export_to_torchscript(self, output_path: str = None) -> str:
        if output_path is None:
            output_path = self.model_path.replace('.pt', '.torchscript')
        
        try:
            self.model.export(format='torchscript')
            
            print(f"TorchScript model exported to: {output_path}")
            
            self.export_history.append({
                'format': 'torchscript',
                'path': output_path,
                'timestamp': datetime.now().isoformat()
            })
            
            return output_path
            
        except Exception as e:
            print(f"Error exporting to TorchScript: {e}")
            raise
    
    def export_to_coreml(self, output_path: str = None) -> str:
        if output_path is None:
            output_path = self.model_path.replace('.pt', '.mlmodel')
        
        try:
            self.model.export(format='coreml')
            
            print(f"CoreML model exported to: {output_path}")
            
            self.export_history.append({
                'format': 'coreml',
                'path': output_path,
                'timestamp': datetime.now().isoformat()
            })
            
            return output_path
            
        except Exception as e:
            print(f"Error exporting to CoreML: {e}")
            raise


class TensorRTOptimizer:
    def __init__(self, onnx_model_path: str):
        self.onnx_model_path = onnx_model_path
        self.engine_path = None
        self.session = None
    
    def build_engine(self, output_path: str = None,
                    precision: str = "fp16",
                    workspace_size: int = 4,
                    batch_size: int = 1) -> str:
        try:
            import tensorrt as trt
        except ImportError:
            print("TensorRT not installed. Please install it first.")
            print("Install command: pip install tensorrt")
            raise
        
        if output_path is None:
            output_path = self.onnx_model_path.replace('.onnx', '.trt')
        
        self.engine_path = output_path
        
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        
        with trt.Builder(TRT_LOGGER) as builder, \
             builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
             trt.OnnxParser(network, TRT_LOGGER) as parser:
            
            builder.max_batch_size = batch_size
            
            with open(self.onnx_model_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    raise RuntimeError("Failed to parse ONNX model")
            
            config = builder.create_builder_config()
            
            if precision == "fp16":
                config.set_flag(trt.BuilderFlag.FP16)
                print("Using FP16 precision")
            elif precision == "int8":
                config.set_flag(trt.BuilderFlag.INT8)
                print("Using INT8 precision")
            
            config.max_workspace_size = workspace_size * 1 << 30
            
            print("Building TensorRT engine...")
            engine = builder.build_engine(network, config)
            
            if engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
            
            print(f"TensorRT engine saved to: {output_path}")
            
            return output_path
    
    def load_engine(self, engine_path: str = None):
        if engine_path is None:
            engine_path = self.engine_path
        
        if not Path(engine_path).exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        
        self.session = ort.InferenceSession(
            engine_path,
            providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        print(f"TensorRT engine loaded from: {engine_path}")
        
        return self.session
    
    def inference(self, image: np.ndarray) -> np.ndarray:
        if self.session is None:
            raise RuntimeError("TensorRT engine not loaded. Call load_engine() first.")
        
        input_name = self.session.get_inputs()[0].name
        
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        input_tensor = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        outputs = self.session.run(None, {input_name: input_tensor})
        
        return outputs
    
    def benchmark_inference(self, image: np.ndarray, 
                           num_iterations: int = 100) -> Dict:
        import time
        
        if self.session is None:
            self.load_engine()
        
        warmup_iterations = 10
        inference_times = []
        
        for i in range(warmup_iterations):
            self.inference(image)
        
        for i in range(num_iterations):
            start_time = time.time()
            self.inference(image)
            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000)
        
        results = {
            'mean_time_ms': np.mean(inference_times),
            'std_time_ms': np.std(inference_times),
            'min_time_ms': np.min(inference_times),
            'max_time_ms': np.max(inference_times),
            'fps': 1000.0 / np.mean(inference_times)
        }
        
        print(f"\nInference Benchmark Results:")
        print(f"  Mean time: {results['mean_time_ms']:.2f} ms")
        print(f"  Std time: {results['std_time_ms']:.2f} ms")
        print(f"  Min time: {results['min_time_ms']:.2f} ms")
        print(f"  Max time: {results['max_time_ms']:.2f} ms")
        print(f"  FPS: {results['fps']:.2f}")
        
        return results


class ModelComparison:
    def __init__(self, models: Dict[str, str]):
        self.models = {}
        for name, path in models.items():
            self.models[name] = self._load_model(name, path)
    
    def _load_model(self, name: str, path: str):
        if path.endswith('.pt'):
            return YOLO(path)
        elif path.endswith('.onnx'):
            return ort.InferenceSession(
                path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
        elif path.endswith('.trt'):
            return ort.InferenceSession(
                path,
                providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider']
            )
        else:
            raise ValueError(f"Unsupported model format: {path}")
    
    def compare_models(self, test_images: List[str]) -> Dict:
        results = {
            'models': {},
            'comparison': {}
        }
        
        for name, model in self.models.items():
            print(f"\nTesting {name}...")
            
            model_results = []
            inference_times = []
            
            for image_path in test_images:
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                start_time = cv2.getTickCount()
                
                if isinstance(model, YOLO):
                    result = model.predict(image, verbose=False)
                    detections = len(result[0].boxes) if result else 0
                else:
                    result = model.run(None, {
                        model.get_inputs()[0].name: self._preprocess_image(image)
                    })
                    detections = len(result[0][0])
                
                end_time = cv2.getTickCount()
                inference_time = (end_time - start_time) / cv2.getTickFrequency() * 1000
                
                model_results.append(detections)
                inference_times.append(inference_time)
            
            results['models'][name] = {
                'avg_detections': np.mean(model_results),
                'avg_inference_time_ms': np.mean(inference_times),
                'fps': 1000.0 / np.mean(inference_times),
                'std_inference_time_ms': np.std(inference_times)
            }
        
        baseline_name = list(self.models.keys())[0]
        baseline_fps = results['models'][baseline_name]['fps']
        
        for name, metrics in results['models'].items():
            if name != baseline_name:
                speedup = metrics['fps'] / baseline_fps
                results['comparison'][f"{name}_vs_{baseline_name}"] = {
                    'speedup': speedup,
                    'time_reduction_pct': (1 - 1/speedup) * 100
                }
        
        self._print_comparison(results)
        
        return results
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        input_tensor = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def _print_comparison(self, results: Dict):
        print("\n" + "="*60)
        print("Model Comparison Results")
        print("="*60)
        
        for name, metrics in results['models'].items():
            print(f"\n{name}:")
            print(f"  Avg Detections: {metrics['avg_detections']:.2f}")
            print(f"  Avg Inference Time: {metrics['avg_inference_time_ms']:.2f} ms")
            print(f"  FPS: {metrics['fps']:.2f}")
            print(f"  Std Inference Time: {metrics['std_inference_time_ms']:.2f} ms")
        
        print("\n" + "-"*60)
        print("Speedup Comparison")
        print("-"*60)
        
        for comparison, metrics in results['comparison'].items():
            print(f"\n{comparison}:")
            print(f"  Speedup: {metrics['speedup']:.2f}x")
            print(f"  Time Reduction: {metrics['time_reduction_pct']:.1f}%")
    
    def save_comparison_report(self, results: Dict, output_path: str = None):
        if output_path is None:
            output_path = "models/model_comparison_report.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nComparison report saved to: {output_path}")


class IntegratedModelExporter:
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path
        self.exporter = YOLOModelExporter(base_model_path)
        self.tensorrt_optimizer = None
        self.export_results = {}
    
    def full_export_pipeline(self, input_size: int = 640, 
                            tensorrt_precision: str = "fp16") -> Dict:
        print("="*60)
        print("Starting Full Model Export Pipeline")
        print("="*60)
        
        print("\n1. Exporting to ONNX format...")
        onnx_path = self.exporter.export_to_onnx(
            input_size=input_size,
            simplify=True
        )
        self.export_results['onnx'] = {
            'path': onnx_path,
            'status': 'success'
        }
        
        print("\n2. Exporting to TorchScript format...")
        try:
            torchscript_path = self.exporter.export_to_torchscript()
            self.export_results['torchscript'] = {
                'path': torchscript_path,
                'status': 'success'
            }
        except Exception as e:
            print(f"TorchScript export failed: {e}")
            self.export_results['torchscript'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        print("\n3. Building TensorRT engine...")
        try:
            self.tensorrt_optimizer = TensorRTOptimizer(onnx_path)
            trt_path = self.tensorrt_optimizer.build_engine(
                precision=tensorrt_precision
            )
            self.export_results['tensorrt'] = {
                'path': trt_path,
                'precision': tensorrt_precision,
                'status': 'success'
            }
        except Exception as e:
            print(f"TensorRT build failed: {e}")
            self.export_results['tensorrt'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        print("\n" + "="*60)
        print("Export Pipeline Complete")
        print("="*60)
        
        self._save_export_report()
        
        return self.export_results
    
    def _save_export_report(self):
        report = {
            'base_model': self.base_model_path,
            'timestamp': datetime.now().isoformat(),
            'exports': self.export_results,
            'export_history': self.exporter.export_history
        }
        
        output_path = "models/export_report.json"
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nExport report saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Export and Optimization')
    parser.add_argument('--model', type=str, required=True, help='Model path (.pt)')
    parser.add_argument('--format', type=str, 
                       choices=['onnx', 'torchscript', 'coreml', 'tensorrt', 'all'],
                       default='onnx', help='Export format')
    parser.add_argument('--input-size', type=int, default=640, help='Input size')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--precision', type=str, default='fp16',
                       choices=['fp16', 'int8'], help='TensorRT precision')
    parser.add_argument('--compare', action='store_true',
                       help='Compare different model formats')
    parser.add_argument('--test-images', type=str, nargs='+',
                       help='Test images for comparison')
    
    args = parser.parse_args()
    
    if args.format == 'all':
        exporter = IntegratedModelExporter(args.model)
        results = exporter.full_export_pipeline(
            input_size=args.input_size,
            tensorrt_precision=args.precision
        )
        
        print("\nExport Results:")
        for format_name, result in results.items():
            if result['status'] == 'success':
                print(f"  {format_name}: {result.get('path', 'N/A')}")
            else:
                print(f"  {format_name}: Failed - {result.get('error', 'Unknown error')}")
    
    elif args.format == 'tensorrt':
        exporter = YOLOModelExporter(args.model)
        onnx_path = exporter.export_to_onnx(input_size=args.input_size)
        
        optimizer = TensorRTOptimizer(onnx_path)
        trt_path = optimizer.build_engine(precision=args.precision)
        optimizer.load_engine(trt_path)
    
    elif args.compare and args.test_images:
        models = {
            'pytorch': args.model,
            'onnx': args.model.replace('.pt', '.onnx'),
            'tensorrt': args.model.replace('.pt', '.trt')
        }
        
        comparison = ModelComparison(models)
        results = comparison.compare_models(args.test_images)
        comparison.save_comparison_report(results)
    
    else:
        exporter = YOLOModelExporter(args.model)
        if args.format == 'onnx':
            exporter.export_to_onnx(input_size=args.input_size, batch_size=args.batch_size)
        elif args.format == 'torchscript':
            exporter.export_to_torchscript()
        elif args.format == 'coreml':
            exporter.export_to_coreml()