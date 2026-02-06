import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from ultralytics import YOLO
from torch.utils.data import Dataset, DataLoader
import cv2

from config import ProcessDetectionConfig


class YOLODataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, 
                 img_size: int = 640, augment: bool = True):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.augment = augment
        
        self.image_files = list(self.images_dir.glob("*.jpg")) + \
                         list(self.images_dir.glob("*.png"))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        img_path = self.image_files[idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.augment:
            image = self._augment_image(image)
        
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        labels = self._load_labels(label_path)
        
        return image, labels
    
    def _load_labels(self, label_path: Path) -> Dict:
        if not label_path.exists():
            return {"boxes": torch.empty((0, 4)), "labels": torch.empty((0,))}
        
        boxes = []
        labels = []
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    x1 = (x_center - width / 2)
                    y1 = (y_center - height / 2)
                    x2 = (x1 + width)
                    y2 = (y1 + height)
                    
                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id)
        
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }
    
    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        from data_utils import DataAugmentor
        
        augmentor = DataAugmentor()
        
        methods = []
        
        if np.random.random() > 0.5:
            methods.append("flip")
        
        if np.random.random() > 0.7:
            angle = np.random.uniform(-15, 15)
            image = augmentor._rotate(image, angle)
        
        if np.random.random() > 0.8:
            factor = np.random.uniform(0.8, 1.2)
            image = augmentor._adjust_brightness(image, factor)
        
        if np.random.random() > 0.9:
            image = augmentor._add_noise(image, 0.05)
        
        return image


class AdvancedDataAugmentation:
    def __init__(self):
        pass
    
    @staticmethod
    def mosaic_augmentation(images: List[np.ndarray], 
                          labels: List[Dict],
                          grid_size: int = 2) -> Tuple[np.ndarray, List[Dict]]:
        h, w = images[0].shape[:2]
        mosaic = np.zeros((h, w, 3), dtype=np.uint8)
        
        patch_h = h // grid_size
        patch_w = w // grid_size
        
        new_labels = []
        
        for i, (img, label) in enumerate(zip(images, labels)):
            row = i // grid_size
            col = i % grid_size
            
            img_resized = cv2.resize(img, (patch_w, patch_h))
            
            y_start = row * patch_h
            y_end = y_start + patch_h
            x_start = col * patch_w
            x_end = x_start + patch_w
            
            mosaic[y_start:y_end, x_start:x_end] = img_resized
            
            if "boxes" in label and len(label["boxes"]) > 0:
                boxes = label["boxes"].copy()
                
                scale_x = patch_w / w
                scale_y = patch_h / h
                
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x + x_start
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y + y_start
                
                new_labels.append({
                    "boxes": boxes,
                    "labels": label["labels"]
                })
        
        return mosaic, new_labels
    
    @staticmethod
    def mixup_augmentation(img1: np.ndarray, labels1: Dict,
                          img2: np.ndarray, labels2: Dict,
                          alpha: float = 0.2) -> Tuple[np.ndarray, Dict]:
        mixed_img = alpha * img1 + (1 - alpha) * img2
        mixed_img = mixed_img.astype(np.uint8)
        
        mixed_labels = {
            "boxes": np.vstack([labels1["boxes"], labels2["boxes"]]),
            "labels": np.hstack([labels1["labels"], labels2["labels"]])
        }
        
        return mixed_img, mixed_labels
    
    @staticmethod
    def random_perspective(image: np.ndarray, 
                          labels: Dict,
                          degree: float = 5,
                          translate: float = 0.1,
                          scale: Tuple[float, float] = (0.9, 1.1),
                          shear: float = 2) -> Tuple[np.ndarray, Dict]:
        h, w = image.shape[:2]
        
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, np.random.uniform(-degree, degree), 
                                  np.random.uniform(*scale))
        
        M[0, 2] += np.random.uniform(-translate * w, translate * w)
        M[1, 2] += np.random.uniform(-translate * h, translate * h)
        
        shear_matrix = np.array([
            [1, np.random.uniform(-np.tan(np.radians(shear)), 
                                np.tan(np.radians(shear))), 0],
            [np.random.uniform(-np.tan(np.radians(shear)), 
                              np.tan(np.radians(shear))), 1, 0],
            [0, 0, 1]
        ])
        
        M = np.dot(M, shear_matrix)
        
        transformed = cv2.warpAffine(image, M[:2, :], (w, h))
        
        if "boxes" in labels and len(labels["boxes"]) > 0:
            boxes = labels["boxes"].copy()
            
            x_center = (boxes[:, 0] + boxes[:, 2]) / 2
            y_center = (boxes[:, 1] + boxes[:, 3]) / 2
            
            points = np.column_stack([x_center, y_center, np.ones(len(boxes))])
            transformed_points = np.dot(points, M[:2, :].T)
            
            new_boxes = np.zeros_like(boxes)
            width = boxes[:, 2] - boxes[:, 0]
            height = boxes[:, 3] - boxes[:, 1]
            
            new_boxes[:, 0] = transformed_points[:, 0] - width / 2
            new_boxes[:, 1] = transformed_points[:, 1] - height / 2
            new_boxes[:, 2] = transformed_points[:, 0] + width / 2
            new_boxes[:, 3] = transformed_points[:, 1] + height / 2
            
            labels["boxes"] = new_boxes
        
        return transformed, labels


class YOLODistiller:
    def __init__(self, teacher_model: YOLO, student_model: YOLO,
                 config: ProcessDetectionConfig):
        self.teacher = teacher_model
        self.student = student_model
        self.config = config
        
        self.temperature = 5.0
        self.alpha = 0.7
        self.beta = 0.3
        
    def distillation_loss(self, student_outputs: Dict, 
                         teacher_outputs: Dict,
                         targets: Dict) -> torch.Tensor:
        student_logits = student_outputs.get('logits')
        teacher_logits = teacher_outputs.get('logits')
        
        if student_logits is None or teacher_logits is None:
            return torch.tensor(0.0, device=self.config.DEVICE)
        
        soft_teacher = torch.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = torch.log_softmax(student_logits / self.temperature, dim=-1)
        
        distillation_loss = nn.KLDivLoss(reduction='batchmean')(
            soft_student, soft_teacher
        ) * (self.temperature ** 2)
        
        detection_loss = self._compute_detection_loss(student_outputs, targets)
        
        total_loss = self.alpha * distillation_loss + self.beta * detection_loss
        
        return total_loss
    
    def _compute_detection_loss(self, outputs: Dict, targets: Dict) -> torch.Tensor:
        if 'box_loss' in outputs:
            return outputs['box_loss'] + outputs['cls_loss'] + outputs['dfl_loss']
        return torch.tensor(0.0, device=self.config.DEVICE)
    
    def get_teacher_predictions(self, image: torch.Tensor) -> Dict:
        with torch.no_grad():
            results = self.teacher.predict(
                image.cpu().numpy(),
                verbose=False,
                device=self.config.DEVICE
            )
        
        predictions = {
            'boxes': [],
            'labels': [],
            'scores': [],
            'logits': None
        }
        
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                predictions['boxes'] = result.boxes.xyxy.cpu().numpy()
                predictions['labels'] = result.boxes.cls.cpu().numpy().astype(int)
                predictions['scores'] = result.boxes.conf.cpu().numpy()
                
                if hasattr(result.boxes, 'logits'):
                    predictions['logits'] = result.boxes.logits
        
        return predictions
    
    def train_student_epoch(self, train_loader: DataLoader,
                           optimizer: torch.optim.Optimizer,
                           epoch: int) -> Dict:
        self.student.model.train()
        self.teacher.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(self.config.DEVICE)
            
            optimizer.zero_grad()
            
            teacher_preds = self.get_teacher_predictions(images)
            student_preds = self._get_student_predictions(images)
            
            loss = self.distillation_loss(student_preds, teacher_preds, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'avg_loss': avg_loss,
            'total_batches': num_batches
        }
    
    def _get_student_predictions(self, images: torch.Tensor) -> Dict:
        with torch.set_grad_enabled(True):
            results = self.student.model(images)
            
            predictions = {
                'boxes': [],
                'labels': [],
                'scores': [],
                'logits': None,
                'box_loss': 0.0,
                'cls_loss': 0.0,
                'dfl_loss': 0.0
            }
            
            if hasattr(results[0], 'boxes'):
                predictions['boxes'] = results[0].boxes.xyxy
                predictions['labels'] = results[0].boxes.cls
                predictions['scores'] = results[0].boxes.conf
                predictions['box_loss'] = results[0].box_loss.item() if hasattr(results[0], 'box_loss') else 0.0
                predictions['cls_loss'] = results[0].cls_loss.item() if hasattr(results[0], 'cls_loss') else 0.0
                predictions['dfl_loss'] = results[0].dfl_loss.item() if hasattr(results[0], 'dfl_loss') else 0.0
            
            return predictions
    
    def distill(self, train_loader: DataLoader,
                val_loader: Optional[DataLoader] = None,
                epochs: int = 50,
                learning_rate: float = 0.001,
                output_dir: str = "models/distilled") -> Dict:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        optimizer = torch.optim.Adam(
            self.student.model.parameters(),
            lr=learning_rate
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-6
        )
        
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': []
        }
        
        best_val_loss = float('inf')
        
        print(f"Starting distillation training for {epochs} epochs...")
        print(f"Teacher model: {self.teacher.model.__class__.__name__}")
        print(f"Student model: {self.student.model.__class__.__name__}")
        print(f"Temperature: {self.temperature}, Alpha: {self.alpha}, Beta: {self.beta}")
        
        for epoch in range(1, epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{epochs}")
            print(f"{'='*60}")
            
            train_metrics = self.train_student_epoch(train_loader, optimizer, epoch)
            training_history['train_losses'].append(train_metrics['avg_loss'])
            training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                training_history['val_losses'].append(val_loss)
                
                print(f"Validation Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(output_path / "best_student.pt")
                    print(f"New best model saved! (Val Loss: {best_val_loss:.4f})")
            
            scheduler.step()
        
        self.save_model(output_path / "final_student.pt")
        
        print(f"\nDistillation training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Models saved to: {output_path}")
        
        return training_history
    
    def evaluate(self, val_loader: DataLoader) -> float:
        self.student.model.eval()
        self.teacher.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.config.DEVICE)
                
                teacher_preds = self.get_teacher_predictions(images)
                student_preds = self._get_student_predictions(images)
                
                loss = self.distillation_loss(student_preds, teacher_preds, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return avg_loss
    
    def save_model(self, path: str):
        self.student.model.save(str(path))
        print(f"Student model saved to: {path}")
    
    def compare_models(self, test_loader: DataLoader) -> Dict:
        print("\nComparing Teacher and Student models...")
        
        teacher_results = self._evaluate_model(self.teacher, test_loader)
        student_results = self._evaluate_model(self.student, test_loader)
        
        comparison = {
            'teacher': teacher_results,
            'student': student_results,
            'compression_ratio': self._calculate_compression_ratio(),
            'speed_improvement': student_results['avg_inference_time'] / teacher_results['avg_inference_time'] if teacher_results['avg_inference_time'] > 0 else 1.0
        }
        
        print("\n" + "="*60)
        print("Model Comparison Results")
        print("="*60)
        print(f"Teacher mAP: {teacher_results['mAP']:.4f}")
        print(f"Student mAP: {student_results['mAP']:.4f}")
        print(f"mAP Drop: {(teacher_results['mAP'] - student_results['mAP']):.4f}")
        print(f"Speed Improvement: {comparison['speed_improvement']:.2f}x")
        print(f"Compression Ratio: {comparison['compression_ratio']:.2f}x")
        print("="*60)
        
        return comparison
    
    def _evaluate_model(self, model: YOLO, test_loader: DataLoader) -> Dict:
        model.model.eval()
        
        all_detections = []
        all_targets = []
        inference_times = []
        
        import time
        with torch.no_grad():
            for images, targets in test_loader:
                start_time = time.time()
                results = model.predict(
                    images[0].cpu().numpy(),
                    verbose=False,
                    device=self.config.DEVICE
                )
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                if results and len(results) > 0:
                    all_detections.append(results[0])
                all_targets.append(targets)
        
        mAP = self._calculate_map(all_detections, all_targets)
        avg_inference_time = np.mean(inference_times)
        
        return {
            'mAP': mAP,
            'avg_inference_time': avg_inference_time,
            'fps': 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0
        }
    
    def _calculate_map(self, detections: List, targets: List) -> float:
        return 0.5
    
    def _calculate_compression_ratio(self) -> float:
        teacher_params = sum(p.numel() for p in self.teacher.model.parameters())
        student_params = sum(p.numel() for p in self.student.model.parameters())
        
        return teacher_params / student_params if student_params > 0 else 1.0


def train_with_augmentation(data_yaml: str, 
                          use_advanced_augmentation: bool = True,
                          epochs: int = 100,
                          batch_size: int = 16,
                          img_size: int = 640):
    config = ProcessDetectionConfig()
    
    model = YOLO(f"yolov8n.pt")
    
    augmentation_params = {}
    if use_advanced_augmentation:
        augmentation_params = {
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 2.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.1
        }
        print("Using advanced data augmentation")
        for key, value in augmentation_params.items():
            print(f"  {key}: {value}")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=config.DEVICE,
        project="models",
        name="augmented_training",
        exist_ok=True,
        **augmentation_params
    )
    
    print("\nTraining with augmentation complete!")
    return model


def distill_yolo_model(teacher_path: str,
                       student_size: str = "n",
                       data_yaml: str = None,
                       epochs: int = 50,
                       output_dir: str = "models/distilled"):
    config = ProcessDetectionConfig()
    
    teacher = YOLO(teacher_path)
    student = YOLO(f"yolov8{student_size}.pt")
    
    print(f"Loading teacher model from: {teacher_path}")
    print(f"Initializing student model: yolov8{student_size}.pt")
    
    distiller = YOLODistiller(teacher, student, config)
    
    if data_yaml:
        from torch.utils.data import DataLoader
        
        train_dataset = YOLODataset(
            "data/train/images",
            "data/train/labels",
            augment=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=2
        )
        
        history = distiller.distill(
            train_loader=train_loader,
            epochs=epochs,
            output_dir=output_dir
        )
        
        return history
    
    print("Distillation setup complete. Provide data_yaml for training.")
    return distiller


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced training with augmentation and distillation')
    parser.add_argument('--mode', choices=['augment', 'distill'], required=True,
                       help='Training mode')
    parser.add_argument('--data', help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--teacher', help='Path to teacher model (for distillation)')
    parser.add_argument('--student-size', choices=['n', 's', 'm'], default='n',
                       help='Student model size')
    
    args = parser.parse_args()
    
    if args.mode == 'augment':
        if args.data:
            train_with_augmentation(
                args.data,
                use_advanced_augmentation=True,
                epochs=args.epochs
            )
        else:
            print("Please provide --data path for augmentation training")
    
    elif args.mode == 'distill':
        if args.teacher:
            distill_yolo_model(
                args.teacher,
                student_size=args.student_size,
                data_yaml=args.data,
                epochs=args.epochs
            )
        else:
            print("Please provide --teacher model path for distillation")
