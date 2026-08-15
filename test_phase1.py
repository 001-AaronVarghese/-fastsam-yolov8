import cv2
import numpy as np
from ensemble_detector import YoloFastSAMEnsemble
from ensemble_config import CONFIG


def create_test_image():
    """Create a simple test image with colored objects"""
    image = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray background

    # Add some test objects
    cv2.rectangle(image, (100, 100), (250, 250), (0, 0, 255), -1)  # Red rectangle
    cv2.rectangle(image, (350, 150), (550, 350), (0, 255, 0), -1)  # Green rectangle
    cv2.circle(image, (150, 400), 40, (255, 0, 0), -1)  # Blue circle

    return image


def test_ensemble():
    """Test ensemble system"""
    print("\n" + "=" * 70)
    print("PHASE 1 ENSEMBLE SYSTEM TEST")
    print("=" * 70)

    # Initialize
    print("\n1️⃣ Initializing ensemble...")
    ensemble = YoloFastSAMEnsemble(CONFIG)
    print("✅ Ensemble initialized")

    # Create test image
    print("\n2️⃣ Creating test image...")
    test_img = create_test_image()
    print(f"✅ Test image created: {test_img.shape}")

    # Process test image
    print("\n3️⃣ Processing test image...")
    results = ensemble.process_frame(test_img)
    print(f"✅ Frame processed in {results['time']*1000:.1f}ms")

    # Print results
    print("\n4️⃣ Detection Results:")
    yolo_count = len(results["yolo"])
    fs_count = len(results["fastsam"])
    fused_count = len(results["fused"])

    print(f"   YOLOv8n detections: {yolo_count}")
    print(f"   FastSAM detections: {fs_count}")
    print(f"   Fused detections: {fused_count}")

    for i, det in enumerate(results["fused"], 1):
        print(f"     {i}. {det['cls_name']}: {det['conf']:.2f} (source: {det['source']})")

    # Draw detections
    print("\n5️⃣ Drawing detections...")
    vis = ensemble.draw_detections(test_img, results)
    cv2.imwrite("test_result.jpg", vis)
    print("✅ Visualization saved: test_result.jpg")

    # Print stats
    print("\n6️⃣ Performance Statistics:")
    stats = ensemble.get_stats()
    print(f"   Total frames: {stats['total_frames']}")
    print(f"   Average FPS: {stats['avg_fps']:.2f}")
    print(f"   Processing time: {stats['avg_time_ms']:.2f}ms")

    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_ensemble()
