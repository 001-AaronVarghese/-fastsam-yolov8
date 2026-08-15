from ultralytics import YOLO

if __name__ == '__main__':
    model_path = r'D:/new stuff/phase1_yolo8_fastsam/runs/detect/train8/weights/best.pt'
    model = YOLO(model_path)  # Load the best trained model
    model.export(format='onnx')  # Export to ONNX format