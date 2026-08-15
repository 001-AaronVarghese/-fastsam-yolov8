import numpy as np
import torch
from typing import List, Dict

class AdaptiveDetectionFilter:
    CONFIDENCE_THRESHOLDS = {
        'car': 0.45,
        'person': 0.50,
        'motorcycle': 0.45,
        'bus': 0.45,
        'bicycle': 0.55,
        'auto_rickshaw': 0.45,
        'truck': 0.45,
        'animal': 0.60,
        'pothole': 0.55,
        'traffic_cone': 0.60,
        'object': 0.65,
    }

    @staticmethod
    def apply_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
        if len(boxes) == 0:
            return []
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        boxes = boxes.astype(np.float32)
        scores = scores.astype(np.float32)
        indices = np.argsort(-scores)
        keep = []
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            if len(indices) == 1:
                break
            current_box = boxes[current]
            remaining_boxes = boxes[indices[1:]]
            iou_scores = AdaptiveDetectionFilter._compute_iou(current_box, remaining_boxes)
            valid_mask = iou_scores < iou_threshold
            indices = indices[1:][valid_mask]
        return keep

    @staticmethod
    def _compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        x1_inter = np.maximum(box[0], boxes[:, 0])
        y1_inter = np.maximum(box[1], boxes[:, 1])
        x2_inter = np.minimum(box[2], boxes[:, 2])
        y2_inter = np.minimum(box[3], boxes[:, 3])
        inter_w = np.maximum(0.0, x2_inter - x1_inter)
        inter_h = np.maximum(0.0, y2_inter - y1_inter)
        inter = inter_w * inter_h
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_box + area_boxes - inter
        iou = inter / (union + 1e-8)
        return iou

    @staticmethod
    def filter_by_class_threshold(detections: List[Dict], thresholds: Dict = None) -> List[Dict]:
        if thresholds is None:
            thresholds = AdaptiveDetectionFilter.CONFIDENCE_THRESHOLDS
        filtered = []
        for det in detections:
            class_name = det.get('cls_name', 'object')
            min_conf = thresholds.get(class_name, 0.55)
            if det.get('conf', 0) >= min_conf:
                filtered.append(det)
        return filtered

    @staticmethod
    def combine_nms_and_threshold(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        if len(detections) == 0:
            return []
        boxes = []
        scores = []
        for det in detections:
            bbox = det.get('bbox', [0, 0, 0, 0])
            if isinstance(bbox, torch.Tensor):
                bbox = bbox.cpu().numpy()
            boxes.append(bbox)
            scores.append(det.get('conf', 0.5))
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        keep_indices = AdaptiveDetectionFilter.apply_nms(boxes, scores, iou_threshold)
        nms_detections = [detections[i] for i in keep_indices]
        filtered = AdaptiveDetectionFilter.filter_by_class_threshold(nms_detections)
        return filtered