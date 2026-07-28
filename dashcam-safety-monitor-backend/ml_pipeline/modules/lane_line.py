import os
import cv2
import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLO

class LaneLineDetector:
    """
    Lane Line Detection Module.
    Will be fully configured once custom lane line logic/weights are provided by the user.
    """

    def __init__(self, weights_path: str = None):
        if weights_path is None:
            weights_dir = os.path.join(os.path.dirname(__file__), "..", "weights", "lane-line")
            abs_dir = os.path.abspath(weights_dir)
            pt_files = [f for f in os.listdir(abs_dir) if f.endswith(".pt")] if os.path.exists(abs_dir) else []
            weights_path = os.path.join(abs_dir, pt_files[0]) if pt_files else None

        if weights_path and os.path.exists(weights_path):
            print(f"[LaneLineDetector] Loading custom weights from: {weights_path}")
            try:
                self.model = YOLO(weights_path)
                self.initialized = True
            except Exception as e:
                print(f"[LaneLineDetector] Model load error: {e}")
                self.model = None
                self.initialized = False
        else:
            print("[LaneLineDetector] Awaiting custom lane line logic from user.")
            self.model = None
            self.initialized = False

    def detect(self, frame: np.ndarray, is_video: bool = False) -> List[Dict[str, Any]]:
        detections = []
        h, w, _ = frame.shape

        if self.initialized and self.model is not None:
            try:
                results = self.model(frame, conf=0.15, verbose=False)[0]
                for box in results.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    class_name = self.model.names.get(cls_id, "Lane Marking")
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(conf, 2),
                        "class_name": class_name,
                        "category": "lane_line",
                        "color": [255, 255, 0]
                    })
            except Exception as err:
                print(f"[LaneLineDetector] Inference error: {err}")

        return detections
