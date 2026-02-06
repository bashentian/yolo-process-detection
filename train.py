from ultralytics import YOLO
from pathlib import Path
from config import ProcessDetectionConfig


def train_custom_model(data_yaml: str, epochs: int = 100, 
                      model_size: str = "n"):
    config = ProcessDetectionConfig()
    
    model_name = f"yolov8{model_size}.pt"
    model = YOLO(model_name)
    
    print(f"Training model: {model_name}")
    print(f"Data config: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Device: {config.DEVICE}")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16,
        device=config.DEVICE,
        project="models",
        name="custom_process_detection",
        exist_ok=True
    )
    
    print("\nTraining complete!")
    print(f"Best model saved at: {results.save_dir}")
    
    return model


def export_model_to_onnx(model_path: str, output_path: str = None):
    model = YOLO(model_path)
    
    if output_path is None:
        output_path = model_path.replace('.pt', '.onnx')
    
    model.export(format="onnx")
    print(f"Model exported to ONNX format: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train custom YOLO model')
    parser.add_argument('data', help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--size', choices=['n', 's', 'm', 'l', 'x'],
                       default='n', help='Model size')
    parser.add_argument('--export', action='store_true',
                       help='Export model to ONNX after training')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    
    args = parser.parse_args()
    
    use_augmentation = not args.no_augmentation
    model = train_custom_model(args.data, args.epochs, args.size, use_augmentation)
    
    if args.export:
        best_model = Path("models/custom_process_detection/weights/best.pt")
        if best_model.exists():
            export_model_to_onnx(str(best_model))
