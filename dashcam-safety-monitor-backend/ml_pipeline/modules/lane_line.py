import os
import cv2
import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLO

# Configurable via .env or hardcoded fallbacks (default: 0.15)
DEFAULT_MODEL_CONF = float(os.getenv("CONF_LANE_LINE", os.getenv("DEFAULT_MODEL_CONFIDENCE", "0.15")))
DEFAULT_DET_CONF = float(os.getenv("DET_CONF_LANE_LINE", os.getenv("DEFAULT_DETECTION_CONFIDENCE", "0.15")))

# Exact classes from ml_pipeline/weights/lane-line/Readme.md
LANE_LINE_CLASSES = {
    0: "BUS LANE",
    1: "Yellow Markings",
    2: "Line 1",
    3: "Line 2",
    4: "Crossing",
    5: "Romb",
    6: "SLOW",
    7: "Left Arrow",
    8: "Forward Arrow",
    9: "Forward Arrow -Left",
    10: "Forward Arrow -Right",
    11: "Right Arrow",
    12: "Bicycle"
}

class LaneLineDetector:
    """
    Lane Line & Road Surface Marking Detection Module.
    Weights file: ml_pipeline/weights/lane-line/best-7.pt
    Detects ONLY the 13 classes specified in weights/lane-line/Readme.md
    """

    def __init__(self, weights_path: str = None, model_conf: float = None, det_conf: float = None):
        self.model_conf = model_conf if model_conf is not None else DEFAULT_MODEL_CONF
        self.det_conf = det_conf if det_conf is not None else DEFAULT_DET_CONF

        if weights_path is None:
            weights_dir = os.path.join(
                os.path.dirname(__file__), "..", "weights", "lane-line", "best-7.pt"
            )
            weights_path = os.path.abspath(weights_dir)

        print(f"[LaneLineDetector] Loading custom weights from: {weights_path} (model_conf={self.model_conf}, det_conf={self.det_conf})")
        if os.path.exists(weights_path):
            try:
                self.model = YOLO(weights_path)
                self.initialized = True
            except Exception as e:
                print(f"[LaneLineDetector] Model load error: {e}")
                self.model = None
                self.initialized = False
        else:
            print(f"[LaneLineDetector] Weights file not found at: {weights_path}")
            self.model = None
            self.initialized = False

    def detect(self, frame: np.ndarray, is_video: bool = False, model_conf: float = None, det_conf: float = None) -> List[Dict[str, Any]]:
        detections = []
        h, w, _ = frame.shape
        model_conf_to_use = model_conf if model_conf is not None else self.model_conf
        det_conf_to_use = det_conf if det_conf is not None else self.det_conf

        if self.initialized and self.model is not None:
            try:
                # 1. YOLO Model Inference Confidence Threshold
                results = self.model(frame, conf=model_conf_to_use, verbose=False)[0]
                for box in results.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())

                    # 2. Post-Inference Detection Confidence Filter
                    if conf < det_conf_to_use:
                        continue

                    # Prioritize exact Readme.md class mapping over raw model.names (e.g. class__2)
                    if cls_id in LANE_LINE_CLASSES:
                        class_name = LANE_LINE_CLASSES[cls_id]
                    elif hasattr(self.model, "names") and cls_id in self.model.names:
                        class_name = self.model.names[cls_id]
                    else:
                        class_name = f"Lane Marking #{cls_id}"

                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(conf, 2),
                        "class_name": class_name,
                        "category": "lane_line",
                        "color": [255, 255, 0]  # Cyan / Yellow (BGR)
                    })
            except Exception as err:
                print(f"[LaneLineDetector] Inference error: {err}")

        return detections
