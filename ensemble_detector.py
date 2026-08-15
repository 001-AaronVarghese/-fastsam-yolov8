import cv2
import numpy as np
import time
from typing import List, Dict, Tuple
from ultralytics import YOLO, FastSAM
from ensemble_config import CONFIG, CLASS_COLORS
from add_nms_and_thresholding import AdaptiveDetectionFilter
from distance_estimation_alert_system import DistanceEstimator, AlertInterface, process_frame_distance_alert
from zero_dce_module import AdaptiveLowLightEnhancer

class YoloFastSAMEnsemble:
    def __init__(self, cfg: Dict = CONFIG):
        self.cfg = cfg
        self.device = cfg["device"]
        self.frame_count = 0
        self.processing_times = []
        self.distance_estimator = DistanceEstimator()
        self.alert_interface = AlertInterface()
        self.enhancer = AdaptiveLowLightEnhancer(device=self.device, brightness_thresh=0.4)
        self.yolo = YOLO(cfg["yolo_model"])
        self.yolo.to(self.device)
        self.fastsam = FastSAM(cfg["fastsam_model"])
        self.fastsam.to(self.device)

    @staticmethod
    def _box_iou(b1: np.ndarray, b2: np.ndarray) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h
        if inter == 0:
            return 0.0
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def _run_yolo(self, frame: np.ndarray) -> List[Dict]:
        results = self.yolo.predict(
            source=frame,
            imgsz=self.cfg["imgsz"],
            conf=self.cfg["yolo_conf"],
            iou=self.cfg["yolo_iou"],
            device=self.device,
            verbose=False,
            half=True,
        )[0]
        dets = []
        if results.boxes is None or len(results.boxes) == 0:
            return dets
        for box in results.boxes:
            xyxy = box.xyxy.cpu().numpy()[0]
            conf = float(box.conf.cpu().numpy()[0])
            cls_id = int(box.cls.cpu().numpy()[0])
            cls_name = results.names.get(cls_id, f"class_{cls_id}")
            dets.append({
                "bbox": xyxy,
                "conf": conf,
                "cls_id": cls_id,
                "cls_name": cls_name,
                "source": "yolo",
                "mask": None,
            })
        return dets

    def _run_fastsam(self, frame: np.ndarray) -> List[Dict]:
        results = self.fastsam(
            frame,
            device=self.device,
            retina_masks=True,
            imgsz=self.cfg["imgsz"],
            conf=self.cfg["fastsam_conf"],
            iou=self.cfg["fastsam_iou"],
            verbose=False,
        )[0]
        dets = []
        if results.masks is None or results.boxes is None:
            return dets
        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        for i in range(len(boxes)):
            dets.append({
                "bbox": boxes[i],
                "conf": float(scores[i]),
                "cls_id": 0,
                "cls_name": "object",
                "source": "fastsam",
                "mask": masks[i],
            })
        return dets

    def _fuse_detections(self, yolo_dets: List[Dict], fs_dets: List[Dict]) -> List[Dict]:
        fused = []
        used_fastsam = set()
        for y_det in yolo_dets:
            y_box = y_det["bbox"]
            best_iou = 0
            best_j = -1
            for j, f_det in enumerate(fs_dets):
                if j in used_fastsam:
                    continue
                f_box = f_det["bbox"]
                iou = self._box_iou(y_box, f_box)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= self.cfg["fusion_iou_thresh"]:
                f_det = fs_dets[best_j]
                used_fastsam.add(best_j)
                w_y = self.cfg["fusion_yolo_weight"]
                w_f = self.cfg["fusion_fastsam_weight"]
                fused_box = w_y * y_box + w_f * f_det["bbox"]
                fused_conf = w_y * y_det["conf"] + w_f * f_det["conf"]
                fused.append({
                    "bbox": fused_box,
                    "conf": fused_conf,
                    "cls_id": y_det["cls_id"],
                    "cls_name": y_det["cls_name"],
                    "source": "ensemble",
                    "mask": f_det.get("mask", None),
                })
            else:
                fused.append(y_det)
        for j, f_det in enumerate(fs_dets):
            if j not in used_fastsam:
                fused.append(f_det)
        return fused

    def process_frame(self, frame: np.ndarray) -> Dict:
        start_time = time.time()
        frame, was_enhanced = self.enhancer.enhance(frame, apply_check=True)
        yolo_dets = self._run_yolo(frame)
        fs_dets = self._run_fastsam(frame)
        fused_dets = self._fuse_detections(yolo_dets, fs_dets)
        fused_dets = AdaptiveDetectionFilter.combine_nms_and_threshold(fused_dets, iou_threshold=0.5)
        proc_time = time.time() - start_time
        self.processing_times.append(proc_time)
        self.frame_count += 1
        out, system_state = process_frame_distance_alert(frame, fused_dets, self.distance_estimator, self.alert_interface)
        return {
            "yolo": yolo_dets,
            "fastsam": fs_dets,
            "fused": fused_dets,
            "time": proc_time,
            "fps": 1.0 / proc_time if proc_time > 0 else 0.0,
            "was_enhanced": was_enhanced,
            "annotated_frame": out,
            "system_state": system_state,
        }

    def draw_detections(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        out = results['annotated_frame']
        fused_dets = results["fused"]
        for det in fused_dets:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["conf"]
            cls_name = det["cls_name"]
            color = CLASS_COLORS.get(cls_name, (200, 200, 200))
            line_width = 3 if det["source"] == "ensemble" else 2
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, line_width)
            if self.cfg["draw_scores"]:
                label = f"{cls_name} {conf:.2f}"
            else:
                label = cls_name
            cv2.putText(out, label, (int(x1), max(int(y1) - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            if self.cfg["show_masks"] and det.get("mask") is not None:
                mask = det["mask"].astype(np.uint8) * 255
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                colored_mask = np.zeros_like(frame)
                color = CLASS_COLORS.get(det["cls_name"], (200, 200, 200))
                colored_mask[:, :, 0] = color[0] * mask // 255
                colored_mask[:, :, 1] = color[1] * mask // 255
                colored_mask[:, :, 2] = color[2] * mask // 255
                out = cv2.addWeighted(out, 1.0, colored_mask, 0.35, 0)
        if results.get("was_enhanced", False):
            cv2.putText(out, "LOW-LIGHT ENHANCEMENT ACTIVE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        return out

    def get_stats(self) -> Dict:
        if not self.processing_times:
            return {}
        times = np.array(self.processing_times)
        return {
            "total_frames": self.frame_count,
            "avg_time_ms": np.mean(times) * 1000,
            "avg_fps": np.mean(1.0 / times),
        }