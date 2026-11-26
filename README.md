# Phase 1: YOLOv8n + FastSAM Ensemble System

Real-time obstacle detection using YOLOv8n (detection) + FastSAM (segmentation) ensemble.

## 📋 Features

- Fast detection with YOLOv8n (YOLO nano model)
- Instance segmentation with FastSAM
- IoU-based ensemble fusion for robust predictions
- Real-time processing on GTX 1650 (10-25 FPS)
- CPU fallback support (2-4 FPS)
- Webcam and video file support
- Performance statistics and logging

## 🚀 Installation

### 1. Create project folder

```bash
mkdir phase1_yolo8_fastsam
cd phase1_yolo8_fastsam
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify setup

```bash
python test_phase1.py
```

## 🎮 Usage

### Webcam mode

```bash
# Use GPU (CUDA)
python main_phase1.py --mode webcam --device cuda

# Or CPU only
python main_phase1.py --mode webcam --device cpu
```

### Video file mode

```bash
python main_phase1.py --mode video --video path/to/video.mp4
```

### Command-line options

```
--mode {webcam,video}  Input source (default: webcam)
--video PATH           Path to video file (required for video mode)
--camera ID            Camera ID for webcam (default: 0)
--device {cuda,cpu}    Computation device (default: cuda)
```

## ⌨️ Keyboard controls

- **q**: Quit
- **s**: Save screenshot

## 📊 Performance

| Hardware | Mode | FPS |
|----------|------|-----|
| GTX 1650 | GPU (CUDA) | 10-25 |
| Intel/AMD | CPU | 2-4 |
| Raspberry Pi 4 | CPU | 0.5-1 |

## 🔧 Configuration

Edit `ensemble_config.py` to adjust:

- Model selection (`yolo_model`, `fastsam_model`)
- Confidence thresholds
- Fusion parameters (IoU threshold, weights)
- Visualization options

## 📁 Project Structure

```
phase1_yolo8_fastsam/
├── ensemble_config.py      # Configuration
├── ensemble_detector.py    # Core detection logic
├── main_phase1.py         # Main entry point
├── test_phase1.py         # Testing script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🧪 Testing

Run the test suite to verify installation:

```bash
python test_phase1.py
```

This will:
1. Initialize the ensemble system
2. Create a test image
3. Run detection
4. Save visualization to `test_result.jpg`
5. Print performance statistics

## 📈 Next Steps

- **Phase 2**: Add Zero-DCE++ low-light enhancement
- **Phase 3**: Add monocular distance estimation and LED alerts

## 🤝 Troubleshooting

### CUDA not available
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Camera not found
Try different camera IDs:
```bash
python main_phase1.py --camera 1
python main_phase1.py --camera 2
```

### Slow performance
- Use GPU with `--device cuda`
- Close background applications
- Reduce input resolution in config

## 📚 References

- YOLOv8: https://docs.ultralytics.com/models/yolov8/
- FastSAM: https://docs.ultralytics.com/models/fast-sam/
- Ultralytics Docs: https://docs.ultralytics.com/

----------------------

**Phase 1 Complete!** Ready for Phase 2 enhancement integration.🚀
