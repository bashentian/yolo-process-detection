import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import json
from datetime import datetime
import logging
import pickle
from dataclasses import dataclass, asdict
from scipy import stats
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DriftDetectionResult:
    drift_detected: bool
    drift_score: float
    drift_type: str
    confidence: float
    affected_features: List[str]
    timestamp: str
    details: Dict


@dataclass
class PerformanceMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_time: float
    fps: float
    timestamp: str


class DataDriftDetector:
    def __init__(self, 
                 reference_data: np.ndarray,
                 window_size: int = 100,
                 drift_threshold: float = 0.05,
                 significance_level: float = 0.05):
        self.reference_data = reference_data
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.significance_level = significance_level
        
        self.pca = PCA(n_components=min(10, reference_data.shape[1]))
        self.reference_pca = self.pca.fit_transform(reference_data)
        
        self.data_window = deque(maxlen=window_size)
        self.drift_history = []
        
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.isolation_forest.fit(self.reference_pca)
    
    def update_reference_data(self, new_reference_data: np.ndarray):
        self.reference_data = new_reference_data
        self.reference_pca = self.pca.fit_transform(new_reference_data)
        self.isolation_forest.fit(self.reference_pca)
        logger.info("Reference data updated")
    
    def detect_drift_psi(self, current_data: np.ndarray) -> DriftDetectionResult:
        reference_hist, _ = np.histogram(self.reference_data, bins=50, density=True)
        current_hist, _ = np.histogram(current_data, bins=50, density=True)
        
        reference_hist = reference_hist + 1e-10
        current_hist = current_hist + 1e-10
        
        psi = np.sum((reference_hist - current_hist) * np.log(reference_hist / current_hist))
        
        drift_detected = psi > self.drift_threshold
        
        result = DriftDetectionResult(
            drift_detected=drift_detected,
            drift_score=float(psi),
            drift_type="population_stability_index",
            confidence=min(psi / self.drift_threshold, 1.0) if drift_detected else 0.0,
            affected_features=["all"],
            timestamp=datetime.now().isoformat(),
            details={"psi_value": float(psi), "threshold": self.drift_threshold}
        )
        
        self.drift_history.append(asdict(result))
        
        return result
    
    def detect_drift_ks(self, current_data: np.ndarray) -> DriftDetectionResult:
        ks_statistic, p_value = stats.ks_2samp(
            self.reference_data.flatten(),
            current_data.flatten()
        )
        
        drift_detected = p_value < self.significance_level
        
        result = DriftDetectionResult(
            drift_detected=drift_detected,
            drift_score=float(ks_statistic),
            drift_type="kolmogorov_smirnov",
            confidence=float(1 - p_value),
            affected_features=["distribution"],
            timestamp=datetime.now().isoformat(),
            details={
                "ks_statistic": float(ks_statistic),
                "p_value": float(p_value),
                "significance_level": self.significance_level
            }
        )
        
        self.drift_history.append(asdict(result))
        
        return result
    
    def detect_drift_isolation_forest(self, current_data: np.ndarray) -> DriftDetectionResult:
        current_pca = self.pca.transform(current_data)
        
        anomaly_scores = self.isolation_forest.decision_function(current_pca)
        
        avg_anomaly_score = np.mean(anomaly_scores)
        
        reference_avg_score = np.mean(self.isolation_forest.decision_function(self.reference_pca))
        
        drift_detected = avg_anomaly_score < reference_avg_score * 0.5
        
        result = DriftDetectionResult(
            drift_detected=drift_detected,
            drift_score=float(avg_anomaly_score),
            drift_type="isolation_forest",
            confidence=min(abs(avg_anomaly_score - reference_avg_score) / abs(reference_avg_score), 1.0),
            affected_features=["feature_space"],
            timestamp=datetime.now().isoformat(),
            details={
                "current_avg_score": float(avg_anomaly_score),
                "reference_avg_score": float(reference_avg_score)
            }
        )
        
        self.drift_history.append(asdict(result))
        
        return result
    
    def detect_drift_comprehensive(self, current_data: np.ndarray) -> Dict[str, DriftDetectionResult]:
        results = {}
        
        results['psi'] = self.detect_drift_psi(current_data)
        results['ks'] = self.detect_drift_ks(current_data)
        results['isolation_forest'] = self.detect_drift_isolation_forest(current_data)
        
        return results
    
    def process_new_data(self, new_data: np.ndarray) -> Dict[str, DriftDetectionResult]:
        self.data_window.extend(new_data)
        
        if len(self.data_window) < self.window_size:
            logger.warning("Insufficient data in window for drift detection")
            return {}
        
        current_window_data = np.array(self.data_window)
        
        return self.detect_drift_comprehensive(current_window_data)
    
    def get_drift_summary(self) -> Dict:
        if not self.drift_history:
            return {"message": "No drift detection history available"}
        
        total_detections = len(self.drift_history)
        drift_count = sum(1 for d in self.drift_history if d['drift_detected'])
        
        drift_by_type = {}
        for drift in self.drift_history:
            drift_type = drift['drift_type']
            if drift_type not in drift_by_type:
                drift_by_type[drift_type] = {'total': 0, 'drift': 0}
            drift_by_type[drift_type]['total'] += 1
            if drift['drift_detected']:
                drift_by_type[drift_type]['drift'] += 1
        
        return {
            "total_detections": total_detections,
            "drift_detected_count": drift_count,
            "drift_rate": drift_count / total_detections if total_detections > 0 else 0,
            "drift_by_type": drift_by_type,
            "latest_drift": self.drift_history[-1] if self.drift_history else None
        }


class PerformanceMonitor:
    def __init__(self, 
                 baseline_accuracy: float = 0.95,
                 performance_threshold: float = 0.05,
                 window_size: int = 100):
        self.baseline_accuracy = baseline_accuracy
        self.performance_threshold = performance_threshold
        self.window_size = window_size
        
        self.metrics_window = deque(maxlen=window_size)
        self.performance_history = []
        self.alerts = []
    
    def update_baseline(self, new_baseline: float):
        self.baseline_accuracy = new_baseline
        logger.info(f"Baseline accuracy updated to: {new_baseline:.4f}")
    
    def record_metrics(self, metrics: PerformanceMetrics):
        self.metrics_window.append(asdict(metrics))
        
        if len(self.metrics_window) == self.window_size:
            self._check_performance_degradation()
        
        self.performance_history.append(asdict(metrics))
    
    def _check_performance_degradation(self):
        if len(self.metrics_window) < self.window_size:
            return
        
        recent_accuracy = np.mean([m['accuracy'] for m in self.metrics_window])
        
        accuracy_drop = self.baseline_accuracy - recent_accuracy
        
        if accuracy_drop > self.performance_threshold:
            alert = {
                "type": "performance_degradation",
                "severity": "high" if accuracy_drop > self.performance_threshold * 2 else "medium",
                "baseline_accuracy": self.baseline_accuracy,
                "current_accuracy": recent_accuracy,
                "accuracy_drop": accuracy_drop,
                "threshold": self.performance_threshold,
                "timestamp": datetime.now().isoformat()
            }
            self.alerts.append(alert)
            logger.warning(f"Performance degradation detected: {alert}")
    
    def get_performance_summary(self) -> Dict:
        if not self.metrics_window:
            return {"message": "No performance data available"}
        
        metrics_df = pd.DataFrame(self.metrics_window)
        
        summary = {
            "window_size": len(self.metrics_window),
            "baseline_accuracy": self.baseline_accuracy,
            "current_metrics": {
                "accuracy": {
                    "mean": float(metrics_df['accuracy'].mean()),
                    "std": float(metrics_df['accuracy'].std()),
                    "min": float(metrics_df['accuracy'].min()),
                    "max": float(metrics_df['accuracy'].max())
                },
                "precision": {
                    "mean": float(metrics_df['precision'].mean()),
                    "std": float(metrics_df['precision'].std())
                },
                "recall": {
                    "mean": float(metrics_df['recall'].mean()),
                    "std": float(metrics_df['recall'].std())
                },
                "f1_score": {
                    "mean": float(metrics_df['f1_score'].mean()),
                    "std": float(metrics_df['f1_score'].std())
                },
                "inference_time": {
                    "mean_ms": float(metrics_df['inference_time'].mean()),
                    "std_ms": float(metrics_df['inference_time'].std())
                },
                "fps": {
                    "mean": float(metrics_df['fps'].mean()),
                    "std": float(metrics_df['fps'].std())
                }
            },
            "performance_trend": self._calculate_trend(metrics_df['accuracy']),
            "alerts": self.alerts[-10:] if self.alerts else []
        }
        
        return summary
    
    def _calculate_trend(self, series: pd.Series) -> str:
        if len(series) < 10:
            return "insufficient_data"
        
        recent_half = series[-len(series)//2:]
        early_half = series[:len(series)//2]
        
        recent_mean = recent_half.mean()
        early_mean = early_half.mean()
        
        if recent_mean > early_mean * 1.02:
            return "improving"
        elif recent_mean < early_mean * 0.98:
            return "degrading"
        else:
            return "stable"


class ModelHealthMonitor:
    def __init__(self, 
                 reference_data_path: Optional[str] = None,
                 baseline_accuracy: float = 0.95):
        self.drift_detector = None
        self.performance_monitor = PerformanceMonitor(
            baseline_accuracy=baseline_accuracy
        )
        self.model_name = "yolo_process_detection"
        self.monitoring_active = False
        
        if reference_data_path:
            self.initialize_drift_detector(reference_data_path)
    
    def initialize_drift_detector(self, reference_data_path: str):
        try:
            reference_data = np.load(reference_data_path)
            self.drift_detector = DataDriftDetector(reference_data)
            self.monitoring_active = True
            logger.info(f"Drift detector initialized with data from {reference_data_path}")
        except Exception as e:
            logger.error(f"Failed to initialize drift detector: {e}")
    
    def monitor_prediction(self, 
                         predictions: np.ndarray,
                         ground_truth: Optional[np.ndarray] = None) -> Dict:
        results = {}
        
        if self.drift_detector and self.monitoring_active:
            drift_results = self.drift_detector.process_new_data(predictions)
            results['drift_detection'] = {
                drift_type: asdict(drift_result) 
                for drift_type, drift_result in drift_results.items()
            }
        
        if ground_truth is not None and self.monitoring_active:
            accuracy = np.mean(predictions == ground_truth)
            precision = np.sum((predictions == ground_truth) & (predictions == 1)) / max(np.sum(predictions == 1), 1)
            recall = np.sum((predictions == ground_truth) & (ground_truth == 1)) / max(np.sum(ground_truth == 1), 1)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-10)
            
            metrics = PerformanceMetrics(
                accuracy=float(accuracy),
                precision=float(precision),
                recall=float(recall),
                f1_score=float(f1),
                inference_time=0.0,
                fps=0.0,
                timestamp=datetime.now().isoformat()
            )
            
            self.performance_monitor.record_metrics(metrics)
            results['performance_metrics'] = asdict(metrics)
        
        return results
    
    def get_health_report(self) -> Dict:
        report = {
            "model_name": self.model_name,
            "monitoring_active": self.monitoring_active,
            "timestamp": datetime.now().isoformat(),
            "drift_summary": {},
            "performance_summary": {},
            "overall_health": "unknown"
        }
        
        if self.drift_detector:
            report['drift_summary'] = self.drift_detector.get_drift_summary()
        
        if self.monitoring_active:
            report['performance_summary'] = self.performance_monitor.get_performance_summary()
        
        report['overall_health'] = self._calculate_overall_health(report)
        
        return report
    
    def _calculate_overall_health(self, report: Dict) -> str:
        if not self.monitoring_active:
            return "monitoring_inactive"
        
        health_score = 100
        
        if 'drift_summary' in report and report['drift_summary'].get('drift_rate', 0) > 0.1:
            health_score -= 20
        
        if 'performance_summary' in report:
            perf_summary = report['performance_summary']
            if 'current_metrics' in perf_summary:
                current_accuracy = perf_summary['current_metrics']['accuracy']['mean']
                if current_accuracy < self.performance_monitor.baseline_accuracy * 0.9:
                    health_score -= 30
                elif current_accuracy < self.performance_monitor.baseline_accuracy * 0.95:
                    health_score -= 15
        
        if 'performance_summary' in report:
            alerts = report['performance_summary'].get('alerts', [])
            recent_high_severity = sum(1 for a in alerts if a.get('severity') == 'high')
            if recent_high_severity > 0:
                health_score -= 20
        
        if health_score >= 80:
            return "healthy"
        elif health_score >= 60:
            return "degraded"
        elif health_score >= 40:
            return "warning"
        else:
            return "critical"
    
    def save_monitoring_state(self, save_path: str = "logs/monitoring_state.pkl"):
        state = {
            'model_name': self.model_name,
            'monitoring_active': self.monitoring_active,
            'performance_history': list(self.performance_monitor.performance_history),
            'alerts': self.performance_monitor.alerts,
            'drift_history': list(self.drift_detector.drift_history) if self.drift_detector else []
        }
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Monitoring state saved to {save_path}")
    
    def load_monitoring_state(self, load_path: str = "logs/monitoring_state.pkl"):
        if not Path(load_path).exists():
            logger.warning(f"Monitoring state file not found: {load_path}")
            return False
        
        try:
            with open(load_path, 'rb') as f:
                state = pickle.load(f)
            
            self.model_name = state.get('model_name', self.model_name)
            self.monitoring_active = state.get('monitoring_active', False)
            self.performance_monitor.performance_history = deque(
                state.get('performance_history', []),
                maxlen=self.performance_monitor.window_size
            )
            self.performance_monitor.alerts = state.get('alerts', [])
            
            if self.drift_detector and 'drift_history' in state:
                self.drift_detector.drift_history = state.get('drift_history', [])
            
            logger.info(f"Monitoring state loaded from {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load monitoring state: {e}")
            return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Drift and Performance Monitoring')
    parser.add_argument('--reference-data', type=str, help='Path to reference data')
    parser.add_argument('--current-data', type=str, help='Path to current data')
    parser.add_argument('--baseline-accuracy', type=float, default=0.95, 
                       help='Baseline accuracy for performance monitoring')
    parser.add_argument('--save-report', type=str, help='Path to save health report')
    
    args = parser.parse_args()
    
    monitor = ModelHealthMonitor(
        reference_data_path=args.reference_data,
        baseline_accuracy=args.baseline_accuracy
    )
    
    if args.current_data and args.reference_data:
        current_data = np.load(args.current_data)
        
        results = monitor.monitor_prediction(current_data)
        print("Drift Detection Results:")
        print(json.dumps(results.get('drift_detection', {}), indent=2))
    
    health_report = monitor.get_health_report()
    print("\nHealth Report:")
    print(json.dumps(health_report, indent=2))
    
    if args.save_report:
        with open(args.save_report, 'w') as f:
            json.dump(health_report, f, indent=2)
        print(f"\nHealth report saved to: {args.save_report}")