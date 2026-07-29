import os
import cv2
import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLO

# Configurable via .env or hardcoded fallbacks (default: 0.15)
DEFAULT_MODEL_CONF = float(os.getenv("CONF_ANOMALY", os.getenv("DEFAULT_MODEL_CONFIDENCE", "0.15")))
DEFAULT_DET_CONF = float(os.getenv("DET_CONF_ANOMALY", os.getenv("DEFAULT_DETECTION_CONFIDENCE", "0.15")))

# Exact classes from ml_pipeline/weights/road-anomaly/Readme.md
ANOMALY_CLASSES = {
    0: "Accident",
    1: "Car Fire",
    2: "Fighting",
    3: "Snatching"
}

class AnomalyDetector:
    """
    Road Anomaly & Critical Hazard Detection Module.
    Weights file: ml_pipeline/weights/road-anomaly/best.pt
    Detects ONLY the 4 classes specified in weights/road-anomaly/Readme.md
    """

    def __init__(self, weights_path: str = None, model_conf: float = None, det_conf: float = None):
        self.model_conf = model_conf if model_conf is not None else DEFAULT_MODEL_CONF
        self.det_conf = det_conf if det_conf is not None else DEFAULT_DET_CONF

        if weights_path is None:
            weights_dir = os.path.join(
                os.path.dirname(__file__), "..", "weights", "road-anomaly", "best.pt"
            )
            weights_path = os.path.abspath(weights_dir)

        print(f"[AnomalyDetector] Loading custom weights from: {weights_path} (model_conf={self.model_conf}, det_conf={self.det_conf})")
        if os.path.exists(weights_path):
            try:
                self.model = YOLO(weights_path)
                self.initialized = True
            except Exception as e:
                print(f"[AnomalyDetector] Model load error: {e}")
                self.model = None
                self.initialized = False
        else:
            print(f"[AnomalyDetector] Weights file not found at: {weights_path}")
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

                    if hasattr(self.model, "names") and cls_id in self.model.names:
                        class_name = self.model.names[cls_id]
                    else:
                        class_name = ANOMALY_CLASSES.get(cls_id, f"Anomaly #{cls_id}")

                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(conf, 2),
                        "class_name": class_name,
                        "category": "anomaly",
                        "color": [0, 0, 255]  # Bright Red (BGR)
                    })
            except Exception as err:
                print(f"[AnomalyDetector] Inference error: {err}")

        return detections
