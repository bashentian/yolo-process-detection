import optuna
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, List, Optional, Callable
import json
import numpy as np
from datetime import datetime
import logging


class YOLOHyperparameterOptimizer:
    def __init__(self, data_yaml: str, model_size: str = "n", 
                 n_trials: int = 50, timeout: int = 7200):
        self.data_yaml = data_yaml
        self.model_size = model_size
        self.n_trials = n_trials
        self.timeout = timeout
        self.study = None
        self.best_params = None
        self.best_score = None
        self.trials_history = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_study(self, study_name: str = "yolo_hyperparameter_optimization",
                    direction: str = "maximize", storage: Optional[str] = None):
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=storage,
            load_if_exists=True
        )
        self.logger.info(f"Study created: {study_name}")
    
    def define_search_space(self, trial: optuna.Trial) -> Dict:
        search_space = {
            'lr0': trial.suggest_float('lr0', 1e-5, 1e-2, log=True),
            'lrf': trial.suggest_float('lrf', 0.01, 0.5),
            'momentum': trial.suggest_float('momentum', 0.8, 0.98),
            'weight_decay': trial.suggest_float('weight_decay', 0.0, 0.001),
            'warmup_epochs': trial.suggest_int('warmup_epochs', 0, 5),
            'warmup_momentum': trial.suggest_float('warmup_momentum', 0.8, 0.95),
            'warmup_bias_lr': trial.suggest_float('warmup_bias_lr', 0.0, 0.2),
            'box': trial.suggest_float('box', 0.02, 0.2),
            'cls': trial.suggest_float('cls', 0.2, 1.0),
            'dfl': trial.suggest_float('dfl', 0.5, 2.0),
            'hsv_h': trial.suggest_float('hsv_h', 0.0, 0.1),
            'hsv_s': trial.suggest_float('hsv_s', 0.0, 0.9),
            'hsv_v': trial.suggest_float('hsv_v', 0.0, 0.9),
            'degrees': trial.suggest_float('degrees', 0.0, 45.0),
            'translate': trial.suggest_float('translate', 0.0, 0.2),
            'scale': trial.suggest_float('scale', 0.0, 0.9),
            'shear': trial.suggest_float('shear', 0.0, 10.0),
            'perspective': trial.suggest_float('perspective', 0.0, 0.001),
            'flipud': trial.suggest_float('flipud', 0.0, 1.0),
            'fliplr': trial.suggest_float('fliplr', 0.0, 1.0),
            'mosaic': trial.suggest_float('mosaic', 0.0, 1.0),
            'mixup': trial.suggest_float('mixup', 0.0, 0.5),
        }
        
        return search_space
    
    def objective(self, trial: optuna.Trial) -> float:
        params = self.define_search_space(trial)
        
        model_name = f"yolov8{self.model_size}.pt"
        model = YOLO(model_name)
        
        epochs = trial.suggest_int('epochs', 10, 50)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        
        try:
            results = model.train(
                data=self.data_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=640,
                **params,
                project="models/hyperparameter_search",
                name=f"trial_{trial.number}",
                exist_ok=True,
                verbose=False
            )
            
            mAP50 = results.results_dict.get('metrics/mAP50(B)', 0)
            
            self.trials_history.append({
                'trial_number': trial.number,
                'params': params,
                'epochs': epochs,
                'batch_size': batch_size,
                'mAP50': mAP50,
                'timestamp': datetime.now().isoformat()
            })
            
            self.logger.info(f"Trial {trial.number}: mAP50 = {mAP50:.4f}")
            
            return mAP50
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {str(e)}")
            return 0.0
    
    def optimize(self, study_name: str = "yolo_hyperparameter_optimization"):
        self.create_study(study_name)
        
        self.logger.info(f"Starting optimization with {self.n_trials} trials...")
        self.logger.info(f"Timeout: {self.timeout} seconds")
        
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        self.logger.info(f"Best mAP50: {self.best_score:.4f}")
        self.logger.info(f"Best parameters: {self.best_params}")
        
        self.save_results()
        
        return self.best_params, self.best_score
    
    def save_results(self):
        results = {
            'study_name': self.study.study_name,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(self.trials_history),
            'trials_history': self.trials_history,
            'timestamp': datetime.now().isoformat()
        }
        
        output_dir = Path("models/hyperparameter_search")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / "optimization_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to: {results_path}")
    
    def visualize_results(self):
        if self.study is None:
            self.logger.warning("No study found. Run optimize() first.")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig = optuna.visualization.matplotlib.plot_optimization_history(self.study)
            plt.title("Optimization History")
            plt.savefig("models/hyperparameter_search/optimization_history.png")
            plt.close()
            
            fig = optuna.visualization.matplotlib.plot_param_importances(self.study)
            plt.title("Parameter Importances")
            plt.savefig("models/hyperparameter_search/param_importances.png")
            plt.close()
            
            fig = optuna.visualization.matplotlib.plot_parallel_coordinate(self.study)
            plt.title("Parallel Coordinate Plot")
            plt.savefig("models/hyperparameter_search/parallel_coordinate.png")
            plt.close()
            
            self.logger.info("Visualization plots saved to models/hyperparameter_search/")
            
        except ImportError:
            self.logger.warning("Matplotlib not available. Skipping visualization.")
    
    def train_with_best_params(self, data_yaml: str, epochs: int = 100,
                               output_name: str = "best_model"):
        if self.best_params is None:
            raise ValueError("No best parameters found. Run optimize() first.")
        
        model_name = f"yolov8{self.model_size}.pt"
        model = YOLO(model_name)
        
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=16,
            imgsz=640,
            **self.best_params,
            project="models",
            name=output_name,
            exist_ok=True
        )
        
        self.logger.info(f"Training with best parameters completed!")
        self.logger.info(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
        
        return results


class MultiObjectiveOptimizer:
    def __init__(self, data_yaml: str, model_size: str = "n", 
                 n_trials: int = 50):
        self.data_yaml = data_yaml
        self.model_size = model_size
        self.n_trials = n_trials
        self.study = None
    
    def multi_objective(self, trial: optuna.Trial) -> tuple:
        params = {
            'lr0': trial.suggest_float('lr0', 1e-5, 1e-2, log=True),
            'lrf': trial.suggest_float('lrf', 0.01, 0.5),
            'momentum': trial.suggest_float('momentum', 0.8, 0.98),
            'weight_decay': trial.suggest_float('weight_decay', 0.0, 0.001),
            'box': trial.suggest_float('box', 0.02, 0.2),
            'cls': trial.suggest_float('cls', 0.2, 1.0),
            'dfl': trial.suggest_float('dfl', 0.5, 2.0),
        }
        
        epochs = trial.suggest_int('epochs', 20, 100)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        
        try:
            model_name = f"yolov8{self.model_size}.pt"
            model = YOLO(model_name)
            
            results = model.train(
                data=self.data_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=640,
                **params,
                project="models/multi_objective_search",
                name=f"trial_{trial.number}",
                exist_ok=True,
                verbose=False
            )
            
            mAP50 = results.results_dict.get('metrics/mAP50(B)', 0)
            inference_time = results.speed.get('inference', 1000) / 1000
            
            return mAP50, -inference_time
            
        except Exception as e:
            return 0.0, -1000.0
    
    def optimize(self):
        self.study = optuna.create_study(
            study_name="yolo_multi_objective",
            directions=["maximize", "maximize"]
        )
        
        self.study.optimize(
            self.multi_objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        best_trials = self.study.best_trials
        
        print("Best Pareto optimal trials:")
        for trial in best_trials:
            print(f"  Trial {trial.number}: mAP50={trial.values[0]:.4f}, "
                  f"inference_time={-trial.values[1]:.2f}ms")
        
        return best_trials


class GridSearchOptimizer:
    def __init__(self, data_yaml: str, model_size: str = "n"):
        self.data_yaml = data_yaml
        self.model_size = model_size
        self.results = []
    
    def define_grid(self) -> Dict[str, List]:
        grid = {
            'lr0': [1e-4, 1e-3, 1e-2],
            'batch_size': [8, 16, 32],
            'epochs': [50, 100],
            'momentum': [0.9, 0.937]
        }
        return grid
    
    def grid_search(self):
        grid = self.define_grid()
        
        total_combinations = 1
        for key, values in grid.items():
            total_combinations *= len(values)
        
        print(f"Total combinations to test: {total_combinations}")
        
        for lr0 in grid['lr0']:
            for batch_size in grid['batch_size']:
                for epochs in grid['epochs']:
                    for momentum in grid['momentum']:
                        try:
                            model_name = f"yolov8{self.model_size}.pt"
                            model = YOLO(model_name)
                            
                            results = model.train(
                                data=self.data_yaml,
                                epochs=epochs,
                                batch=batch_size,
                                lr0=lr0,
                                momentum=momentum,
                                project="models/grid_search",
                                exist_ok=True,
                                verbose=False
                            )
                            
                            mAP50 = results.results_dict.get('metrics/mAP50(B)', 0)
                            
                            self.results.append({
                                'lr0': lr0,
                                'batch_size': batch_size,
                                'epochs': epochs,
                                'momentum': momentum,
                                'mAP50': mAP50
                            })
                            
                            print(f"lr0={lr0:.0e}, batch={batch_size}, "
                                  f"epochs={epochs}, momentum={momentum}: mAP50={mAP50:.4f}")
                            
                        except Exception as e:
                            print(f"Error with parameters lr0={lr0}, batch={batch_size}: {e}")
        
        self.save_results()
        
        best_result = max(self.results, key=lambda x: x['mAP50'])
        print(f"\nBest result: {best_result}")
        
        return best_result
    
    def save_results(self):
        output_dir = Path("models/grid_search")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / "grid_search_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Grid search results saved to: {results_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization for YOLO')
    parser.add_argument('--data', type=str, required=True, help='Data YAML path')
    parser.add_argument('--model-size', type=str, default='n', 
                       choices=['n', 's', 'm', 'l', 'x'], help='Model size')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of trials')
    parser.add_argument('--timeout', type=int, default=7200, help='Timeout in seconds')
    parser.add_argument('--method', type=str, default='bayesian', 
                       choices=['bayesian', 'multi_objective', 'grid'],
                       help='Optimization method')
    parser.add_argument('--train-best', action='store_true',
                       help='Train model with best parameters after optimization')
    
    args = parser.parse_args()
    
    if args.method == 'bayesian':
        optimizer = YOLOHyperparameterOptimizer(
            args.data, args.model_size, args.n_trials, args.timeout
        )
        best_params, best_score = optimizer.optimize()
        optimizer.visualize_results()
        
        if args.train_best:
            optimizer.train_with_best_params(args.data)
    
    elif args.method == 'multi_objective':
        optimizer = MultiObjectiveOptimizer(args.data, args.model_size, args.n_trials)
        best_trials = optimizer.optimize()
    
    elif args.method == 'grid':
        optimizer = GridSearchOptimizer(args.data, args.model_size)
        best_result = optimizer.grid_search()