import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json
import shutil


class DatasetPreparer:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / "images"
        self.labels_dir = self.base_dir / "labels"
        self.train_dir = self.base_dir / "train"
        self.val_dir = self.base_dir / "val"
        
    def create_directories(self):
        for dir_path in [self.images_dir, self.labels_dir, 
                         self.train_dir, self.val_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        for split in [self.train_dir, self.val_dir]:
            (split / "images").mkdir(exist_ok=True)
            (split / "labels").mkdir(exist_ok=True)
    
    def extract_frames_from_video(self, video_path: str, output_dir: str,
                                  interval: int = 30):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % interval == 0:
                output_file = output_path / f"frame_{saved_count:06d}.jpg"
                cv2.imwrite(str(output_file), frame)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {saved_count} frames from video")
        return saved_count
    
    def convert_yolo_to_coco(self, yolo_labels_dir: str, 
                            image_width: int, image_height: int):
        coco_format = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        label_files = list(Path(yolo_labels_dir).glob("*.txt"))
        annotation_id = 1
        
        for label_file in label_files:
            image_id = int(label_file.stem)
            
            coco_format["images"].append({
                "id": image_id,
                "file_name": f"{label_file.stem}.jpg",
                "width": image_width,
                "height": image_height
            })
            
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        x_min = (x_center - width / 2) * image_width
                        y_min = (y_center - height / 2) * image_height
                        bbox_width = width * image_width
                        bbox_height = height * image_height
                        
                        coco_format["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id,
                            "bbox": [x_min, y_min, bbox_width, bbox_height],
                            "area": bbox_width * bbox_height,
                            "iscrowd": 0
                        })
                        annotation_id += 1
        
        return coco_format
    
    def split_dataset(self, ratio: float = 0.8):
        images = list(self.images_dir.glob("*.jpg")) + \
                 list(self.images_dir.glob("*.png"))
        
        np.random.shuffle(images)
        split_idx = int(len(images) * ratio)
        
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        for img in train_images:
            label_path = self.labels_dir / f"{img.stem}.txt"
            shutil.copy(img, self.train_dir / "images" / img.name)
            if label_path.exists():
                shutil.copy(label_path, self.train_dir / "labels" / label_path.name)
        
        for img in val_images:
            label_path = self.labels_dir / f"{img.stem}.txt"
            shutil.copy(img, self.val_dir / "images" / img.name)
            if label_path.exists():
                shutil.copy(label_path, self.val_dir / "labels" / label_path.name)
        
        print(f"Dataset split: {len(train_images)} train, {len(val_images)} val")


class AnnotationTool:
    def __init__(self, image_dir: str, class_names: List[str]):
        self.image_dir = Path(image_dir)
        self.class_names = class_names
        self.output_dir = self.image_dir.parent / "labels"
        self.output_dir.mkdir(exist_ok=True)
        
        self.current_annotations = []
        self.current_class = 0
        self.current_image = None
        self.current_image_name = None
        
    def annotate_images(self):
        images = list(self.image_dir.glob("*.jpg")) + \
                 list(self.image_dir.glob("*.png"))
        
        for img_path in images:
            self.current_image = cv2.imread(str(img_path))
            self.current_image_name = img_path.stem
            self.current_annotations = []
            
            print(f"\nAnnotating: {img_path.name}")
            print(f"Current class: {self.class_names[self.current_class]}")
            print("Controls:")
            print("  Left click: Add bounding box")
            print("  Right click: Remove last box")
            print("  Space: Switch class")
            print("  S: Save annotations")
            print("  N: Next image")
            print("  Q: Quit")
            
            cv2.namedWindow('Annotation')
            cv2.setMouseCallback('Annotation', self._mouse_callback)
            
            self._display_image()
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    return
                elif key == ord('n'):
                    break
                elif key == ord('s'):
                    self._save_annotations()
                elif key == ord(' '):
                    self.current_class = (self.current_class + 1) % len(self.class_names)
                    print(f"Current class: {self.class_names[self.current_class]}")
            
            self._save_annotations()
        
        cv2.destroyAllWindows()
    
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_annotations.append([x, y])
            self._display_image()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.current_annotations:
                self.current_annotations.pop()
                self._display_image()
    
    def _display_image(self):
        display = self.current_image.copy()
        
        for i in range(0, len(self.current_annotations), 4):
            if i + 3 < len(self.current_annotations):
                pts = self.current_annotations[i:i+4]
                cv2.rectangle(display, (pts[0], pts[1]), (pts[2], pts[3]),
                             (0, 255, 0), 2)
        
        status_text = f"Class: {self.class_names[self.current_class]} | Boxes: {len(self.current_annotations)//4}"
        cv2.putText(display, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Annotation', display)
    
    def _save_annotations(self):
        if not self.current_annotations:
            return
        
        height, width = self.current_image.shape[:2]
        
        label_file = self.output_dir / f"{self.current_image_name}.txt"
        
        with open(label_file, 'w') as f:
            for i in range(0, len(self.current_annotations), 4):
                if i + 3 < len(self.current_annotations):
                    x1, y1, x2, y2 = self.current_annotations[i:i+4]
                    
                    x_center = (x1 + x2) / 2 / width
                    y_center = (y1 + y2) / 2 / height
                    box_width = (x2 - x1) / width
                    box_height = (y2 - y1) / height
                    
                    f.write(f"{self.current_class} {x_center} {y_center} {box_width} {box_height}\n")
        
        print(f"Saved annotations to {label_file}")


class DataAugmentor:
    def __init__(self):
        self.augmentations = {
            "rotate": self._rotate,
            "flip": self._flip,
            "brightness": self._adjust_brightness,
            "noise": self._add_noise
        }
    
    def augment_image(self, image: np.ndarray, 
                     methods: List[str] = None) -> np.ndarray:
        if methods is None:
            methods = list(self.augmentations.keys())
        
        augmented = image.copy()
        
        for method in methods:
            if method in self.augmentations:
                augmented = self.augmentations[method](augmented)
        
        return augmented
    
    def _rotate(self, image: np.ndarray, angle: float = 15) -> np.ndarray:
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (width, height))
    
    def _flip(self, image: np.ndarray, direction: int = 1) -> np.ndarray:
        return cv2.flip(image, direction)
    
    def _adjust_brightness(self, image: np.ndarray, 
                           factor: float = 1.2) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def _add_noise(self, image: np.ndarray, 
                  noise_factor: float = 0.1) -> np.ndarray:
        noise = np.random.normal(0, noise_factor * 255, image.shape)
        noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy
    
    def augment_dataset(self, input_dir: str, output_dir: str,
                       augmentation_factor: int = 3):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        images = list(input_path.glob("*.jpg")) + \
                 list(input_path.glob("*.png"))
        
        for img_path in images:
            image = cv2.imread(str(img_path))
            
            for i in range(augmentation_factor):
                augmented = self.augment_image(image)
                output_file = output_path / f"{img_path.stem}_aug_{i}.jpg"
                cv2.imwrite(str(output_file), augmented)
        
        print(f"Augmented {len(images)} images to {len(images) * augmentation_factor}")
