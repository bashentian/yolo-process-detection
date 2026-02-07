import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import random


class YOLODataAugmentation:
    def __init__(self, augment_ratio: float = 0.5):
        self.augment_ratio = augment_ratio
        self.transform = self._create_transform_pipeline()
    
    def _create_transform_pipeline(self) -> A.Compose:
        return A.Compose([
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.3),
            
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
            
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=0.2),
            
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            
            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=1.0),
                A.RandomScale(scale_limit=0.2, p=1.0),
            ], p=0.3),
            
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                A.GridDistortion(p=1.0),
            ], p=0.2),
        ], bbox_params=A.BboxParams(
            format='yolo',
            min_visibility=0.1,
            label_fields=['class_labels']
        ))
    
    def augment_single(self, image: np.ndarray, bboxes: List[List[float]], 
                      class_labels: List[int]) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        if random.random() > self.augment_ratio:
            return image, bboxes, class_labels
        
        if not bboxes:
            return image, bboxes, class_labels
        
        augmented = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
        
        valid_bboxes = []
        valid_labels = []
        
        for bbox, label in zip(augmented['bboxes'], augmented['class_labels']):
            x, y, w, h = bbox
            
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                continue
            
            if x - w/2 < 0 or x + w/2 > 1 or y - h/2 < 0 or y + h/2 > 1:
                continue
            
            valid_bboxes.append(bbox)
            valid_labels.append(label)
        
        return augmented['image'], valid_bboxes, valid_labels
    
    def augment_dataset(self, dataset_dir: str, output_dir: str, 
                        augmentation_factor: int = 3):
        dataset_path = Path(dataset_dir)
        output_path = Path(output_dir)
        
        for split in ['train', 'val']:
            images_dir = dataset_path / split / 'images'
            labels_dir = dataset_path / split / 'labels'
            
            if not images_dir.exists() or not labels_dir.exists():
                continue
            
            output_images_dir = output_path / split / 'images'
            output_labels_dir = output_path / split / 'labels'
            output_images_dir.mkdir(parents=True, exist_ok=True)
            output_labels_dir.mkdir(parents=True, exist_ok=True)
            
            for image_file in images_dir.glob('*'):
                if image_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                
                label_file = labels_dir / (image_file.stem + '.txt')
                
                if not label_file.exists():
                    continue
                
                image = cv2.imread(str(image_file))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                bboxes = []
                class_labels = []
                
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            bbox = list(map(float, parts[1:5]))
                            bboxes.append(bbox)
                            class_labels.append(class_id)
                
                cv2.imwrite(str(output_images_dir / image_file.name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                
                with open(output_labels_dir / label_file.name, 'w') as f:
                    for bbox, label in zip(bboxes, class_labels):
                        line = f"{label} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"
                        f.write(line)
                
                for i in range(augmentation_factor):
                    aug_image, aug_bboxes, aug_labels = self.augment_single(
                        image.copy(), bboxes.copy(), class_labels.copy()
                    )
                    
                    aug_filename = f"{image_file.stem}_aug_{i}{image_file.suffix}"
                    aug_labelname = f"{label_file.stem}_aug_{i}.txt"
                    
                    cv2.imwrite(str(output_images_dir / aug_filename), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                    
                    with open(output_labels_dir / aug_labelname, 'w') as f:
                        for bbox, label in zip(aug_bboxes, aug_labels):
                            line = f"{label} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"
                            f.write(line)
                
                print(f"Processed: {image_file.name} -> Created {augmentation_factor + 1} images")


class DistillationAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            A.MixUp(p=0.5),
            A.CutMix(p=0.5),
            A.Mosaic(p=0.5),
        ])
    
    def augment_for_distillation(self, images: List[np.ndarray], 
                                 bboxes_list: List[List[List[float]]],
                                 labels_list: List[List[int]]) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        if len(images) < 2:
            return images[0], bboxes_list[0], labels_list[0]
        
        if random.random() < 0.5:
            return images[0], bboxes_list[0], labels_list[0]
        
        idx1, idx2 = random.sample(range(len(images)), 2)
        image1, bboxes1, labels1 = images[idx1], bboxes_list[idx1], labels_list[idx1]
        image2, bboxes2, labels2 = images[idx2], bboxes_list[idx2], labels_list[idx2]
        
        mixed_image = (image1.astype(np.float32) + image2.astype(np.float32)) / 2
        mixed_image = mixed_image.astype(np.uint8)
        
        mixed_bboxes = bboxes1 + bboxes2
        mixed_labels = labels1 + labels2
        
        return mixed_image, mixed_bboxes, mixed_labels


class AugmentationValidator:
    @staticmethod
    def validate_yolo_format(bboxes: List[List[float]], image_shape: Tuple[int, int]) -> bool:
        height, width = image_shape[:2]
        
        for bbox in bboxes:
            x, y, w, h = bbox
            
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                return False
            
            if x - w/2 < 0 or x + w/2 > 1 or y - h/2 < 0 or y + h/2 > 1:
                return False
        
        return True
    
    @staticmethod
    def check_augmentation_quality(original_dir: str, augmented_dir: str) -> Dict:
        original_path = Path(original_dir)
        augmented_path = Path(augmented_dir)
        
        stats = {
            'original_count': 0,
            'augmented_count': 0,
            'valid_augmentation_ratio': 0.0,
            'avg_bbox_count_original': 0.0,
            'avg_bbox_count_augmented': 0.0
        }
        
        for split in ['train', 'val']:
            orig_images = list((original_path / split / 'images').glob('*'))
            aug_images = list((augmented_path / split / 'images').glob('*'))
            
            stats['original_count'] += len(orig_images)
            stats['augmented_count'] += len(aug_images)
        
        if stats['original_count'] > 0:
            stats['augmentation_ratio'] = stats['augmented_count'] / stats['original_count']
        
        return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Augmentation for YOLO')
    parser.add_argument('--input', type=str, required=True, help='Input dataset directory')
    parser.add_argument('--output', type=str, required=True, help='Output dataset directory')
    parser.add_argument('--factor', type=int, default=3, help='Augmentation factor')
    parser.add_argument('--augment-ratio', type=float, default=0.8, help='Augmentation ratio per image')
    
    args = parser.parse_args()
    
    augmentor = YOLODataAugmentation(augment_ratio=args.augment_ratio)
    augmentor.augment_dataset(args.input, args.output, args.factor)
    
    validator = AugmentationValidator()
    stats = validator.check_augmentation_quality(args.input, args.output)
    
    print("\nAugmentation Statistics:")
    print(f"Original images: {stats['original_count']}")
    print(f"Augmented images: {stats['augmented_count']}")
    print(f"Augmentation ratio: {stats['augmentation_ratio']:.2f}x")