import numpy as np
import cv2
from typing import Dict, List, Tuple

class DistanceEstimator:
    CAMERA_PARAMS = {
        'h': 1.35,
        'theta_deg': 10,
        'focal_length': 800,
    }

    RISK_THRESHOLDS = {
        'critical_front': 5,   # Inner red (front)
        'caution_mid': 10,     # Orange (mid/sides)
        'safe_outer': float('inf')
    }

    RISK_LEVELS = {
        'CRITICAL': {'color': (0, 0, 255), 'text': 'CRITICAL'},
        'CAUTION': {'color': (0, 165, 255), 'text': 'CAUTION'},
        'SAFE': {'color': (0, 255, 0), 'text': 'SAFE'}
    }

    def __init__(self, camera_params: Dict = None):
        if camera_params:
            self.CAMERA_PARAMS.update(camera_params)
        self.theta_rad = np.deg2rad(self.CAMERA_PARAMS['theta_deg'])
        self.h = self.CAMERA_PARAMS['h']
        self.f = self.CAMERA_PARAMS['focal_length']

    def estimate_distance(self, bbox: Tuple[int, int, int, int], image_height: int) -> float:
        x1, y1, x2, y2 = bbox
        c_y = image_height / 2
        y_bottom = y2
        if y_bottom <= c_y:
            return 999
        try:
            arctan_term = np.arctan((y_bottom - c_y) / self.f)
            total_angle = self.theta_rad + arctan_term
            if total_angle <= 0 or total_angle > np.pi/2:
                return 999
            distance = self.h / np.tan(total_angle)
            return max(0.5, min(100, distance))
        except:
            return 999

    def classify_risk_level(self, distance: float, x_norm: float) -> str:
        center_width = 0.2
        if abs(x_norm - 0.5) < center_width / 2:
            if distance < self.RISK_THRESHOLDS['critical_front']:
                return 'CRITICAL'
            elif distance < self.RISK_THRESHOLDS['caution_mid']:
                return 'CAUTION'
            else:
                return 'SAFE'
        elif abs(x_norm - 0.5) < 0.4:
            if distance < self.RISK_THRESHOLDS['caution_mid']:
                return 'CAUTION'
            else:
                return 'SAFE'
        else:
            return 'SAFE'

    def get_risk_info(self, distance: float) -> Dict:
        risk_level = self.classify_risk_level(distance, 0.5)
        return {
            'risk_level': risk_level,
            'distance': distance,
            **self.RISK_LEVELS[risk_level]
        }

class AlertInterface:
    def draw_led_style_indicator(self, frame: np.ndarray, risk_level: str) -> np.ndarray:
        out = frame.copy()
        led_radius = 40
        led_x = out.shape[1] - 70
        led_y = 60
        colors = {
            'CRITICAL': (0, 0, 255),
            'CAUTION': (0, 165, 255),
            'SAFE': (0, 255, 0)
        }
        color = colors.get(risk_level, (0, 255, 0))
        cv2.circle(out, (led_x, led_y), led_radius, color, -1)
        cv2.circle(out, (led_x, led_y), led_radius + 8, color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = risk_level
        text_size = cv2.getTextSize(text, font, 0.8, 2)[0]
        text_x = led_x - text_size[0] // 2
        text_y = led_y + 65
        cv2.putText(out, text, (text_x, text_y), font, 0.8, (255, 255, 255), 2)
        return out

    def draw_distance_label(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], distance: float, risk_info: Dict) -> np.ndarray:
        out = frame.copy()
        x1, y1, x2, y2 = bbox
        color = risk_info['color']
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        return out

def process_frame_distance_alert(frame: np.ndarray, detections: List[Dict], estimator: DistanceEstimator, alert_interface: AlertInterface) -> Tuple[np.ndarray, Dict]:
    out = frame.copy()
    image_height = frame.shape[0]
    image_width = frame.shape[1]
    max_risk_level = 'SAFE'
    critical_count = 0
    caution_count = 0
    obstacle_classes = ['car', 'person', 'motorcycle', 'bus', 'bicycle', 'auto_rickshaw', 'truck', 'animal', 'pothole', 'traffic_cone']
    min_aspect = 0.2
    max_aspect = 5.0
    filtered_dets = []
    for det in detections:
        if det['cls_name'] not in obstacle_classes:
            continue
        x1, y1, x2, y2 = det['bbox']
        width = x2 - x1
        height = y2 - y1
        aspect = width / height if height > 0 else 0
        if min_aspect < aspect < max_aspect:
            filtered_dets.append(det)
    for det in filtered_dets:
        bbox = det['bbox']
        distance = estimator.estimate_distance(bbox, image_height)
        x_center = (bbox[0] + bbox[2]) / 2
        x_norm = x_center / image_width
        risk_level = estimator.classify_risk_level(distance, x_norm)
        risk_info = {'risk_level': risk_level, 'distance': distance, **estimator.RISK_LEVELS[risk_level]}
        out = alert_interface.draw_distance_label(out, bbox, distance, risk_info)
        if risk_level == 'CRITICAL':
            max_risk_level = 'CRITICAL'
            critical_count += 1
        elif risk_level == 'CAUTION' and max_risk_level != 'CRITICAL':
            max_risk_level = 'CAUTION'
            caution_count += 1
    out = alert_interface.draw_led_style_indicator(out, max_risk_level)
    system_state = {
        'max_risk_level': max_risk_level,
        'critical_count': critical_count,
        'caution_count': caution_count,
        'detection_count': len(filtered_dets)
    }
    return out, system_state