import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import cv2
import json
from datetime import datetime


class ModelPruner:
    def __init__(self, model_path: str, pruning_ratio: float = 0.3):
        self.model_path = model_path
        self.pruning_ratio = pruning_ratio
        self.model = YOLO(model_path)
        self.pytorch_model = self.model.model
    
    def calculate_importance_scores(self) -> Dict[str, float]:
        importance_scores = {}
        
        for name, module in self.pytorch_model.named_modules():
            if isinstance(module, nn.Conv2d):
                weight = module.weight.data.abs()
                importance = weight.mean().item()
                importance_scores[name] = importance
        
        return importance_scores
    
    def prune_model(self) -> str:
        importance_scores = self.calculate_importance_scores()
        
        threshold = np.percentile(list(importance_scores.values()), 
                                 self.pruning_ratio * 100)
        
        for name, module in self.pytorch_model.named_modules():
            if isinstance(module, nn.Conv2d):
                if importance_scores[name] < threshold:
                    with torch.no_grad():
                        module.weight.data *= 0.1
        
        output_path = self.model_path.replace('.pt', '_pruned.pt')
        self.pytorch_model.save(output_path)
        
        return output_path
    
    def evaluate_pruned_model(self, val_data_path: str) -> Dict:
        results = self.model.val(data=val_data_path)
        
        metrics = {
            'mAP50': results.box.map50,
            'mAP50-95': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr,
            'inference_time': results.speed['inference']
        }
        
        return metrics


class ModelDistillation:
    def __init__(self, teacher_model_path: str, student_model_path: str, 
                 temperature: float = 3.0, alpha: float = 0.5):
        self.teacher_model = YOLO(teacher_model_path)
        self.student_model = YOLO(student_model_path)
        self.temperature = temperature
        self.alpha = alpha
    
    def get_teacher_predictions(self, image: np.ndarray) -> Dict:
        results = self.teacher_model.predict(image, verbose=False)
        
        if len(results) == 0:
            return {'boxes': [], 'scores': [], 'classes': []}
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        
        soft_scores = self._apply_temperature(scores)
        
        return {
            'boxes': boxes,
            'scores': soft_scores,
            'classes': classes
        }
    
    def _apply_temperature(self, scores: np.ndarray) -> np.ndarray:
        return np.power(scores, 1.0 / self.temperature)
    
    def distillation_loss(self, student_output: torch.Tensor, 
                         teacher_output: torch.Tensor,
                         true_labels: torch.Tensor) -> torch.Tensor:
        soft_teacher_loss = nn.KLDivLoss(reduction='batchmean')(
            torch.log(student_output / self.temperature),
            teacher_output / self.temperature
        ) * (self.temperature ** 2)
        
        hard_student_loss = nn.CrossEntropyLoss()(student_output, true_labels)
        
        total_loss = self.alpha * soft_teacher_loss + (1 - self.alpha) * hard_student_loss
        return total_loss
    
    def train_student_with_distillation(self, data_yaml: str, epochs: int = 50):
        results = self.student_model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=640,
            batch=16,
            patience=10,
            project="models",
            name="distilled_student",
            exist_ok=True
        )
        
        return results


class ModelQuantizer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = YOLO(model_path)
    
    def quantize_model(self, quantization_type: str = "dynamic") -> str:
        if quantization_type == "dynamic":
            quantized_model = torch.quantization.quantize_dynamic(
                self.model.model,
                {nn.Conv2d, nn.Linear},
                dtype=torch.qint8
            )
        elif quantization_type == "static":
            quantized_model = self._static_quantization()
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
        
        output_path = self.model_path.replace('.pt', f'_{quantization_type}_quantized.pt')
        
        self.model.model = quantized_model
        torch.save(self.model.model.state_dict(), output_path)
        
        return output_path
    
    def _static_quantization(self):
        self.model.model.eval()
        self.model.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        model_prepared = torch.quantization.prepare(self.model.model)
        
        dummy_input = torch.randn(1, 3, 640, 640)
        model_prepared(dummy_input)
        
        quantized_model = torch.quantization.convert(model_prepared)
        
        return quantized_model
    
    def evaluate_quantized_model(self, val_data_path: str) -> Dict:
        results = self.model.val(data=val_data_path)
        
        metrics = {
            'mAP50': results.box.map50,
            'mAP50-95': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr,
            'model_size_mb': Path(self.model_path).stat().st_size / (1024 * 1024)
        }
        
        return metrics


class IntegratedModelOptimizer:
    def __init__(self, base_model_path: str, teacher_model_path: Optional[str] = None):
        self.base_model_path = base_model_path
        self.teacher_model_path = teacher_model_path
        self.optimization_history = []
    
    def full_optimization_pipeline(self, data_yaml: str, val_data_path: str) -> Dict:
        results = {
            'baseline': {},
            'pruned': {},
            'quantized': {},
            'distilled': {},
            'final': {}
        }
        
        print("Starting full optimization pipeline...")
        
        print("\n1. Evaluating baseline model...")
        base_model = YOLO(self.base_model_path)
        baseline_results = base_model.val(data=val_data_path)
        results['baseline'] = {
            'mAP50': baseline_results.box.map50,
            'mAP50-95': baseline_results.box.map,
            'precision': baseline_results.box.mp,
            'recall': baseline_results.box.mr,
            'inference_time': baseline_results.speed['inference'],
            'model_size_mb': Path(self.base_model_path).stat().st_size / (1024 * 1024)
        }
        
        print("\n2. Applying pruning...")
        pruner = ModelPruner(self.base_model_path, pruning_ratio=0.2)
        pruned_path = pruner.prune_model()
        pruned_results = pruner.evaluate_pruned_model(val_data_path)
        results['pruned'] = pruned_results
        self.optimization_history.append({
            'step': 'pruning',
            'path': pruned_path,
            'metrics': pruned_results
        })
        
        print("\n3. Applying quantization...")
        quantizer = ModelQuantizer(pruned_path)
        quantized_path = quantizer.quantize_model("dynamic")
        quantized_results = quantizer.evaluate_quantized_model(val_data_path)
        results['quantized'] = quantized_results
        self.optimization_history.append({
            'step': 'quantization',
            'path': quantized_path,
            'metrics': quantized_results
        })
        
        if self.teacher_model_path:
            print("\n4. Applying knowledge distillation...")
            distiller = ModelDistillation(self.teacher_model_path, quantized_path)
            distillation_results = distiller.train_student_with_distillation(data_yaml, epochs=30)
            
            distilled_model = YOLO("models/distilled_student/weights/best.pt")
            distilled_results_val = distilled_model.val(data=val_data_path)
            results['distilled'] = {
                'mAP50': distilled_results_val.box.map50,
                'mAP50-95': distilled_results_val.box.map,
                'precision': distilled_results_val.box.mp,
                'recall': distilled_results_val.box.mr,
                'inference_time': distilled_results_val.speed['inference']
            }
            self.optimization_history.append({
                'step': 'distillation',
                'path': "models/distilled_student/weights/best.pt",
                'metrics': results['distilled']
            })
        
        print("\n5. Final optimization summary...")
        results['final'] = {
            'compression_ratio': results['baseline']['model_size_mb'] / results['quantized']['model_size_mb'],
            'accuracy_drop': results['baseline']['mAP50'] - results['quantized']['mAP50'],
            'speed_improvement': results['baseline']['inference_time'] / results['quantized']['inference_time'] if results['quantized'].get('inference_time') else 1.0
        }
        
        self._save_optimization_report(results)
        
        return results
    
    def _save_optimization_report(self, results: Dict):
        report_path = Path("models/optimization_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'base_model': self.base_model_path,
            'teacher_model': self.teacher_model_path,
            'optimization_steps': len(self.optimization_history),
            'results': results,
            'history': self.optimization_history
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nOptimization report saved to: {report_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Optimization Pipeline')
    parser.add_argument('--model', type=str, required=True, help='Base model path')
    parser.add_argument('--teacher', type=str, help='Teacher model path for distillation')
    parser.add_argument('--data', type=str, required=True, help='Data YAML path')
    parser.add_argument('--val', type=str, required=True, help='Validation data path')
    parser.add_argument('--optimization', choices=['prune', 'quantize', 'distill', 'full'], 
                       default='full', help='Optimization type')
    
    args = parser.parse_args()
    
    if args.optimization == 'full':
        optimizer = IntegratedModelOptimizer(args.model, args.teacher)
        results = optimizer.full_optimization_pipeline(args.data, args.val)
        
        print("\nFinal Optimization Results:")
        print(f"Compression Ratio: {results['final']['compression_ratio']:.2f}x")
        print(f"Accuracy Drop: {results['final']['accuracy_drop']:.2f}%")
        print(f"Speed Improvement: {results['final']['speed_improvement']:.2f}x")
    elif args.optimization == 'prune':
        pruner = ModelPruner(args.model, pruning_ratio=0.2)
        pruned_path = pruner.prune_model()
        print(f"Pruned model saved to: {pruned_path}")
    elif args.optimization == 'quantize':
        quantizer = ModelQuantizer(args.model)
        quantized_path = quantizer.quantize_model("dynamic")
        print(f"Quantized model saved to: {quantized_path}")
    elif args.optimization == 'distill' and args.teacher:
        distiller = ModelDistillation(args.teacher, args.model)
        results = distiller.train_student_with_distillation(args.data)
        print(f"Distillation training completed")