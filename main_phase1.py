import cv2
import argparse
import datetime
from ensemble_detector import YoloFastSAMEnsemble
from ensemble_config import CONFIG

def run_webcam(ensemble, camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        return
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('final_demo_output.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = ensemble.process_frame(frame)
        vis = ensemble.draw_detections(frame, results)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
        cv2.putText(vis, timestamp, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        stats = ensemble.get_stats()
        metrics_text = f"FPS: {stats.get('avg_fps', 0):.1f} | Detections: {len(results['fused'])}"
        cv2.putText(vis, metrics_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Obstacle Detection System", vis)
        out.write(vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        frame_count += 1
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def run_video(ensemble, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('final_demo_output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = ensemble.process_frame(frame)
        vis = ensemble.draw_detections(frame, results)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
        cv2.putText(vis, timestamp, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        stats = ensemble.get_stats()
        metrics_text = f"FPS: {stats.get('avg_fps', 0):.1f} | Detections: {len(results['fused'])}"
        cv2.putText(vis, metrics_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Obstacle Detection System", vis)
        out.write(vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        frame_count += 1
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["webcam", "video"], default="webcam")
    parser.add_argument("--video", type=str)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    ensemble = YoloFastSAMEnsemble(CONFIG)
    if args.mode == "webcam":
        run_webcam(ensemble, args.camera)
    elif args.mode == "video":
        if not args.video:
            return
        run_video(ensemble, args.video)

if __name__ == "__main__":
    main()