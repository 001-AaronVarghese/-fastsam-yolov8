import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')  # Pretrained nano

    results = model.train(
        data='datasets/merged_coco_idd/data.yaml',  # Your YAML path
        epochs=25,  # Reduced for faster training
        imgsz=512,  # Reduced image size for speed
        batch=10,    # Batch size for GPU memory
        amp=True,   # Mixed precision for acceleration
        patience=10,  # Early stopping
        freeze=10,   # Freeze early layers
        cache=True,  # Cache images for faster loading
        workers=4,   # Data loader workers
        device=0    # Use GPU
    )

    model.export(format='onnx')