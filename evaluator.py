import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json

from detector import ProcessDetector, Detection
from config import ProcessDetectionConfig


class ModelEvaluator:
    def __init__(self, config: ProcessDetectionConfig):
        self.config = config
        self.detector = ProcessDetector(config)
        
    def evaluate_on_dataset(self, images_dir: str, labels_dir: str) -> Dict:
        image_files = list(Path(images_dir).glob("*.jpg")) + \
                     list(Path(images_dir).glob("*.png"))
        
        results = {
            'total_images': len(image_files),
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'class_metrics': defaultdict(lambda: {
                'tp': 0, 'fp': 0, 'fn': 0
            }),
            'confidences': [],
            'inference_times': []
        }
        
        for image_file in image_files:
            label_file = Path(labels_dir) / f"{image_file.stem}.txt"
            
            image = cv2.imread(str(image_file))
            height, width = image.shape[:2]
            
            import time
            start_time = time.time()
            detections = self.detector.detect(image)
            inference_time = time.time() - start_time
            
            results['inference_times'].append(inference_time)
            
            ground_truth = self._load_label(label_file, width, height)
            
            tp, fp, fn = self._calculate_metrics(detections, ground_truth, iou_threshold=0.5)
            
            results['true_positives'] += tp
            results['false_positives'] += fp
            results['false_negatives'] += fn
            
            for det in detections:
                results['confidences'].append(det.confidence)
        
        results['precision'] = self._calculate_precision(results)
        results['recall'] = self._calculate_recall(results)
        results['f1_score'] = self._calculate_f1(results)
        results['mAP'] = self._calculate_map(results)
        results['avg_inference_time'] = np.mean(results['inference_times'])
        results['fps'] = 1.0 / results['avg_inference_time']
        
        return dict(results)
    
    def _load_label(self, label_file: Path, width: int, height: int) -> List[Dict]:
        if not label_file.exists():
            return []
        
        labels = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, bbox_width, bbox_height = map(float, parts[1:5])
                    
                    x1 = (x_center - bbox_width / 2) * width
                    y1 = (y_center - bbox_height / 2) * height
                    x2 = (x1 + bbox_width) * width
                    y2 = (y1 + bbox_height) * height
                    
                    labels.append({
                        'class_id': class_id,
                        'bbox': (x1, y1, x2, y2)
                    })
        
        return labels
    
    def _calculate_metrics(self, detections: List[Detection], 
                          ground_truth: List[Dict],
                          iou_threshold: float = 0.5) -> Tuple[int, int, int]:
        if len(ground_truth) == 0:
            return 0, len(detections), 0
        
        if len(detections) == 0:
            return 0, 0, len(ground_truth)
        
        matched_gt = set()
        tp = 0
        fp = 0
        
        for det in detections:
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt in enumerate(ground_truth):
                if i in matched_gt:
                    continue
                
                if det.class_id == gt['class_id']:
                    iou = self._calculate_iou(det.bbox, gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
            
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        fn = len(ground_truth) - len(matched_gt)
        
        return tp, fp, fn
    
    def _calculate_iou(self, bbox1: Tuple[float, float, float, float],
                       bbox2: Tuple[float, float, float, float]) -> float:
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_x1 = max(x1_min, x2_min)
        inter_y1 = max(y1_min, y2_min)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _calculate_precision(self, results: Dict) -> float:
        total_predictions = results['true_positives'] + results['false_positives']
        return results['true_positives'] / total_predictions if total_predictions > 0 else 0.0
    
    def _calculate_recall(self, results: Dict) -> float:
        total_gt = results['true_positives'] + results['false_negatives']
        return results['true_positives'] / total_gt if total_gt > 0 else 0.0
    
    def _calculate_f1(self, results: Dict) -> float:
        precision = self._calculate_precision(results)
        recall = self._calculate_recall(results)
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def _calculate_map(self, results: Dict, iou_thresholds: List[float] = None) -> float:
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
        mAP = 0.0
        
        for iou_threshold in iou_thresholds:
            precision = self._calculate_precision(results)
            recall = self._calculate_recall(results)
            ap = (precision + recall) / 2
            mAP += ap
        
        return mAP / len(iou_thresholds)
    
    def print_evaluation_report(self, results: Dict):
        print("\n" + "="*60)
        print("模型评估报告")
        print("="*60)
        print(f"总图像数: {results['total_images']}")
        print(f"真实正例: {results['true_positives']}")
        print(f"虚假正例: {results['false_positives']}")
        print(f"虚假反例: {results['false_negatives']}")
        print()
        print(f"精确率: {results['precision']:.4f}")
        print(f"召回率: {results['recall']:.4f}")
        print(f"F1分数: {results['f1_score']:.4f}")
        print(f"mAP@0.5:0.95: {results['mAP']:.4f}")
        print()
        print(f"平均推理时间: {results['avg_inference_time']*1000:.2f}ms")
        print(f"推理速度: {results['fps']:.2f} FPS")
        print("="*60)
    
    def save_evaluation_report(self, results: Dict, output_path: str):
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"评估报告已保存到: {output_path}")


class ResultVisualizer:
    def __init__(self, output_dir: str = "outputs/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int],
                             class_names: List[str]):
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import confusion_matrix
        except ImportError:
            print("matplotlib or sklearn not installed. Install with: pip install matplotlib scikit-learn")
            return
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        output_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"混淆矩阵已保存到: {output_path}")
    
    def plot_metrics_over_time(self, metrics: Dict):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].bar(['Precision', 'Recall', 'F1', 'mAP'],
                      [metrics['precision'], metrics['recall'], 
                       metrics['f1_score'], metrics['mAP']])
        axes[0, 0].set_title('Performance Metrics')
        axes[0, 0].set_ylim(0, 1)
        
        axes[0, 1].hist(metrics['confidences'], bins=20, alpha=0.7)
        axes[0, 1].set_title('Confidence Distribution')
        axes[0, 1].set_xlabel('Confidence')
        axes[0, 1].set_ylabel('Frequency')
        
        axes[1, 0].plot(metrics['inference_times'])
        axes[1, 0].set_title('Inference Time per Image')
        axes[1, 0].set_xlabel('Image Index')
        axes[1, 0].set_ylabel('Time (s)')
        
        axes[1, 1].bar(['TP', 'FP', 'FN'],
                      [metrics['true_positives'], 
                       metrics['false_positives'],
                       metrics['false_negatives']])
        axes[1, 1].set_title('Detection Statistics')
        
        plt.tight_layout()
        
        output_path = self.output_dir / "metrics_visualization.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"指标可视化已保存到: {output_path}")
    
    def plot_class_distribution(self, class_counts: Dict[str, int],
                               class_names: List[str]):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        plt.figure(figsize=(10, 6))
        plt.bar(classes, counts)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_path = self.output_dir / "class_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"类别分布图已保存到: {output_path}")
