import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import deque

from ml_pipeline.modules.road_sign import RoadSignDetector
from ml_pipeline.modules.pothole import PotholeDetector
from ml_pipeline.modules.lane_line import LaneLineDetector
from ml_pipeline.modules.anomaly import AnomalyDetector
from ml_pipeline.alert_pipeline import ReportAlertPipeline
from ml_pipeline.rule_config import RuleConfiguration

# Fine-Grained Per-Class Priority Taxonomy Map
CLASS_PRIORITY_MAP = {
    # Rank 1: CRITICAL Hazards (Color: Bright Red [0, 0, 255])
    "pothole": {"rank": 1, "level": "CRITICAL", "color": [0, 0, 255]},
    "accident": {"rank": 1, "level": "CRITICAL", "color": [0, 0, 255]},
    "car fire": {"rank": 1, "level": "CRITICAL", "color": [0, 0, 255]},
    "stop": {"rank": 1, "level": "CRITICAL", "color": [0, 0, 255]},
    "red light": {"rank": 1, "level": "CRITICAL", "color": [0, 0, 255]},

    # Rank 2: HIGH Hazards (Color: Orange [0, 165, 255])
    "alligator_crack": {"rank": 2, "level": "HIGH", "color": [0, 165, 255]},
    "fighting": {"rank": 2, "level": "HIGH", "color": [0, 165, 255]},
    "snatching": {"rank": 2, "level": "HIGH", "color": [0, 165, 255]},
    "pedestrian-crossing": {"rank": 2, "level": "HIGH", "color": [0, 165, 255]},
    "pedestrian-crossing-ahead": {"rank": 2, "level": "HIGH", "color": [0, 165, 255]},
    "children crossing": {"rank": 2, "level": "HIGH", "color": [0, 165, 255]},
    "children-present-or-crossing-ahead": {"rank": 2, "level": "HIGH", "color": [0, 165, 255]},
    "crossing": {"rank": 2, "level": "HIGH", "color": [255, 255, 0]},
    "slow": {"rank": 2, "level": "HIGH", "color": [255, 255, 0]},
    "bus lane": {"rank": 2, "level": "HIGH", "color": [255, 255, 0]},
    "bicycle": {"rank": 2, "level": "HIGH", "color": [255, 255, 0]},

    # Rank 3: MEDIUM Hazards (Color: Amber/Yellow [0, 215, 255])
    "longitudinal_crack": {"rank": 3, "level": "MEDIUM", "color": [0, 215, 255]},
    "transverse_crack": {"rank": 3, "level": "MEDIUM", "color": [0, 215, 255]},
    "yellow markings": {"rank": 3, "level": "MEDIUM", "color": [255, 255, 0]},
    "line 1": {"rank": 3, "level": "MEDIUM", "color": [255, 255, 0]},
    "line 2": {"rank": 3, "level": "MEDIUM", "color": [255, 255, 0]},
    "romb": {"rank": 3, "level": "MEDIUM", "color": [255, 255, 0]},
    "stop-ahead": {"rank": 3, "level": "MEDIUM", "color": [0, 215, 255]},
    "left-bend-ahead": {"rank": 3, "level": "MEDIUM", "color": [0, 215, 255]},
    "right-bend-ahead": {"rank": 3, "level": "MEDIUM", "color": [0, 215, 255]},
    "double-bend-to-left-ahead": {"rank": 3, "level": "MEDIUM", "color": [0, 215, 255]},
    "double-bend-to-right-ahead": {"rank": 3, "level": "MEDIUM", "color": [0, 215, 255]},
    "narrow-bridge-or-culvert-ahead": {"rank": 3, "level": "MEDIUM", "color": [0, 215, 255]},
    "t-junction-ahead": {"rank": 3, "level": "MEDIUM", "color": [0, 215, 255]},
    "level crossing with gates": {"rank": 3, "level": "MEDIUM", "color": [0, 215, 255]},

    # Rank 4: LOW Priority (Color: Cyan [255, 255, 0])
    "green light": {"rank": 4, "level": "LOW", "color": [255, 255, 0]},
    "bus-stop": {"rank": 4, "level": "LOW", "color": [255, 255, 0]},
    "hospital": {"rank": 4, "level": "LOW", "color": [255, 255, 0]},
    "no honking": {"rank": 4, "level": "LOW", "color": [255, 255, 0]},
    "left arrow": {"rank": 4, "level": "LOW", "color": [255, 255, 0]},
    "forward arrow": {"rank": 4, "level": "LOW", "color": [255, 255, 0]},
    "forward arrow -left": {"rank": 4, "level": "LOW", "color": [255, 255, 0]},
    "forward arrow -right": {"rank": 4, "level": "LOW", "color": [255, 255, 0]},
    "right arrow": {"rank": 4, "level": "LOW", "color": [255, 255, 0]},
}

MODEL_ALIAS_MAP = {
    "anomaly": "anomaly",
    "road_anomaly": "anomaly",
    "road-anomaly": "anomaly",
    "anomalies": "anomaly",

    "lane_line": "lane_line",
    "lane-line": "lane_line",
    "lane": "lane_line",
    "lanes": "lane_line",

    "pothole": "pothole",
    "potholes": "pothole",
    "pothole_hazard": "pothole",

    "road_sign": "road_sign",
    "road-sign": "road_sign",
    "sign": "road_sign",
    "signs": "road_sign",
}

def sanitize_for_json(obj: Any) -> Any:
    """Recursively converts NumPy numeric types and arrays into standard Python types."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [sanitize_for_json(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]
    return obj


class FusionManager:
    """
    Central Fusion Layer Manager.
    - Runs selected deep learning models strictly for classes defined in weights Readme.md
    - Triggers alert whenever any class is detected
    - Suppresses duplicate bounding boxes (IoU > 0.5)
    - Ensures clean JSON serialization
    """

    def __init__(self):
        self.detectors = {
            "road_sign": RoadSignDetector(),
            "pothole": PotholeDetector(),
            "lane_line": LaneLineDetector(),
            "anomaly": AnomalyDetector()
        }
        self.rule_config = RuleConfiguration()
        self.alert_pipeline = ReportAlertPipeline(self.rule_config)
        self._apply_model_rule_settings()
        self.last_primary_alert = None
        self.last_image_quality = None

    def _apply_model_rule_settings(self):
        road_sign = self.detectors.get("road_sign")
        if road_sign:
            road_sign.nms_iou = self.rule_config.data["road_sign_filter"][
                "nms_iou_threshold"
            ]
        pothole = self.detectors.get("pothole")
        if pothole:
            pothole.nms_iou = self.rule_config.data["pothole_filter"][
                "nms_iou_threshold"
            ]
        anomaly = self.detectors.get("anomaly")
        if anomaly:
            anomaly.nms_iou = self.rule_config.data["anomaly_filter"][
                "nms_iou_threshold"
            ]

    def replace_rules(self, data: Dict[str, Any]):
        self.rule_config.replace(data)
        self.alert_pipeline.update_rule_config(self.rule_config)
        self._apply_model_rule_settings()

    def reload_rules(self):
        self.rule_config.reload()
        self.alert_pipeline.update_rule_config(self.rule_config)
        self._apply_model_rule_settings()

    def reset_state(self):
        """Reset temporal queue states."""
        pothole_det = self.detectors.get("pothole")
        if pothole_det and hasattr(pothole_det, "reset_history"):
            pothole_det.reset_history()
        self.alert_pipeline.reset()
        self.last_primary_alert = None
        self.last_image_quality = None

    def calculate_iou(self, boxA: List[int], boxB: List[int]) -> float:
        """Computes Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def get_class_priority(self, class_name: str, category: str) -> Tuple[int, str, List[int]]:
        """Maps detected object class to priority rank, level, and color."""
        name_key = class_name.lower().strip()
        if name_key in CLASS_PRIORITY_MAP:
            item = CLASS_PRIORITY_MAP[name_key]
            return item["rank"], item["level"], item["color"]
        
        # Category defaults
        if category == "anomaly":
            return 1, "CRITICAL", [0, 0, 255]
        elif category == "pothole":
            return 2, "HIGH", [0, 165, 255]
        elif category == "lane_line":
            return 3, "MEDIUM", [255, 255, 0]
        else:
            return 4, "LOW", [0, 215, 255]

    def suppress_duplicates(self, detections: List[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Suppresses duplicate bounding boxes with IoU > threshold."""
        if not detections:
            return []

        sorted_dets = sorted(
            detections,
            key=lambda d: (d.get("priority_rank", 99), -d.get("confidence", 0.0))
        )

        keep = []
        for det in sorted_dets:
            box = det["bbox"]
            duplicate = False
            for kept_det in keep:
                iou = self.calculate_iou(box, kept_det["bbox"])
                if iou > iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                keep.append(det)

        return keep

    def render_annotations(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        primary_alert: Optional[Dict[str, Any]] = None,
        image_quality: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Render thin raw boxes, stronger confirmed hazards, and one alert banner."""
        annotated_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = det.get("color", [0, 255, 0])
            p_level = det.get("priority_level", "LOW")
            label = f"[{p_level}] {det['class_name']} ({int(det['confidence'] * 100)}%)"
            active = det.get("alert_state") == "active"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 4 if active else 1)

            # Label banner
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(
                annotated_frame,
                (x1, max(0, y1 - text_h - 10)),
                (x1 + text_w + 10, y1),
                color,
                -1
            )
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 5, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0) if sum(color) > 400 else (255, 255, 255),
                2,
                cv2.LINE_AA
            )
        if primary_alert:
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (annotated_frame.shape[1], 54), (20, 20, 150), -1)
            cv2.addWeighted(overlay, 0.75, annotated_frame, 0.25, 0, annotated_frame)
            cv2.putText(
                annotated_frame,
                primary_alert["message"],
                (18, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if image_quality and image_quality.get("status_trigger", False):
            cv2.putText(
                annotated_frame,
                "LOW CAMERA VISIBILITY - ALERTS LIMITED",
                (18, annotated_frame.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 215, 255),
                2,
                cv2.LINE_AA,
            )
        return annotated_frame

    def process_frame(
        self,
        frame: np.ndarray,
        active_models: List[str],
        is_video: bool = False,
        timestamp_ms: float = 0.0,
        turn_signal: str = "off",
        vehicle_speed_kmh: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], str, bool]:
        raw_detections = []
        
        parsed_active = set()
        for m in active_models:
            m_clean = m.lower().strip()
            if m_clean in MODEL_ALIAS_MAP:
                parsed_active.add(MODEL_ALIAS_MAP[m_clean])
            elif m_clean == "all":
                parsed_active = {"anomaly", "lane_line", "pothole", "road_sign"}

        for model_key in ["anomaly", "lane_line", "pothole", "road_sign"]:
            if model_key in parsed_active:
                detector = self.detectors.get(model_key)
                if detector:
                    try:
                        dets = detector.detect(frame, is_video=is_video)
                        for d in dets:
                            c_name = d.get("class_name", "")
                            cat = d.get("category", model_key)
                            rank, level, color = self.get_class_priority(c_name, cat)
                            d["priority_rank"] = rank
                            d["priority_level"] = level
                            d["color"] = color
                        raw_detections.extend(dets)
                    except Exception as e:
                        print(f"[FusionManager] Error in model '{model_key}': {e}")

        final_detections, primary_alert, quality = self.alert_pipeline.evaluate(
            frame,
            raw_detections,
            timestamp_ms=timestamp_ms,
            turn_signal=turn_signal,
            vehicle_speed_kmh=vehicle_speed_kmh,
            is_video=is_video,
        )
        self.last_primary_alert = primary_alert
        self.last_image_quality = quality
        highest_priority = primary_alert["category"] if primary_alert else "normal"
        audio_trigger = bool(primary_alert and primary_alert.get("audio_trigger"))
        annotated_frame = self.render_annotations(
            frame, final_detections, primary_alert, quality
        )

        serializable_detections = sanitize_for_json(final_detections)
        return annotated_frame, serializable_detections, highest_priority, audio_trigger
