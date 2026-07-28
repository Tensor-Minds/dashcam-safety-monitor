import os
import cv2
import numpy as np
from collections import deque
from typing import List, Dict, Any
from ultralytics import YOLO

# Exact classes from ml_pipeline/weights/pothole/Readme.md
POTHOLE_CLASSES = {
    0: "alligator_crack",
    1: "longitudinal_crack",
    2: "Pothole",
    3: "transverse_crack"
}

class PotholeDetector:
    """
    Pothole & Road Surface Damage Detection Module.
    Weights file: ml_pipeline/weights/pothole/yolov8s-350-epoch-coslrtrue.pt
    Detects ONLY the 4 classes specified in weights/pothole/Readme.md:
    - 0: alligator_crack
    - 1: longitudinal_crack
    - 2: Pothole
    - 3: transverse_crack
    """

    def __init__(self, weights_path: str = None):
        if weights_path is None:
            weights_dir = os.path.join(
                os.path.dirname(__file__), "..", "weights", "pothole", "yolov8s-350-epoch-coslrtrue.pt"
            )
            weights_path = os.path.abspath(weights_dir)

        print(f"[PotholeDetector] Loading custom weights from: {weights_path}")
        if os.path.exists(weights_path):
            try:
                self.model = YOLO(weights_path)
                self.initialized = True
            except Exception as e:
                print(f"[PotholeDetector] Model load error: {e}")
                self.model = None
                self.initialized = False
        else:
            print(f"[PotholeDetector] Weights file not found at: {weights_path}")
            self.model = None
            self.initialized = False

        self.history = deque(maxlen=6)

    def reset_history(self):
        """Clear temporal history queue."""
        self.history.clear()

    def detect(self, frame: np.ndarray, is_video: bool = False) -> List[Dict[str, Any]]:
        detections = []
        h, w, _ = frame.shape

        if self.initialized and self.model is not None:
            try:
                results = self.model(frame, conf=0.15, verbose=False)[0]
                for box in results.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())

                    # Match exact class name from Readme.md mapping or model.names
                    if hasattr(self.model, "names") and cls_id in self.model.names:
                        class_name = self.model.names[cls_id]
                    else:
                        class_name = POTHOLE_CLASSES.get(cls_id, f"Pothole Hazard #{cls_id}")

                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(conf, 2),
                        "class_name": class_name,
                        "category": "pothole",
                        "color": [0, 165, 255]  # Orange (BGR)
                    })
            except Exception as err:
                print(f"[PotholeDetector] Inference error: {err}")

        return detections
