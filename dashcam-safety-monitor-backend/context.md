# Backend Context & Technical Documentation (`dashcam-safety-monitor-backend`)

## 1. Overview
The **Dashcam Safety Monitor Backend** is a high-performance Python service built with **FastAPI**, **PyTorch**, **Ultralytics YOLOv8**, and **OpenCV**. It executes a real-time multi-model vision pipeline to detect road hazards, traffic signs, potholes, and lane markings from single images, uploaded MP4 videos, and live WebSocket video streams.

---

## 2. Directory Structure & File Map

```
dashcam-safety-monitor-backend/
├── main.py                    # FastAPI application, REST endpoints & WebSocket server
├── requirements.txt           # Python dependencies (FastAPI, PyTorch, Ultralytics, OpenCV, etc.)
├── .env                       # Active environment configuration for confidence thresholds
├── .env.example               # Template for environment configuration
├── Readme.md                  # Backend overview & environment table
├── static/
│   └── videos/                # Output directory for server-rendered annotated MP4 videos
└── ml_pipeline/
    ├── fusion_layer.py        # Central FusionManager (IoU suppression, priority taxonomy, JSON sanitizer)
    ├── modules/
    │   ├── road_sign.py       # Road Sign Detector (weights/road-sign/best.pt - 35 classes)
    │   ├── pothole.py         # Pothole & Surface Damage Detector (weights/pothole/yolov8s-350-epoch-coslrtrue.pt - 4 classes)
    │   ├── lane_line.py       # Lane Line & Road Marking Detector (weights/lane-line/best-7.pt - 13 classes)
    │   └── anomaly.py         # Road Anomaly Detector (weights/road-anomaly/best.pt - 4 classes)
    └── weights/
        ├── road-sign/         # best.pt + Readme.md (35 classes)
        ├── pothole/           # yolov8s-350-epoch-coslrtrue.pt + Readme.md (4 classes)
        ├── lane-line/         # best-7.pt + Readme.md (13 classes)
        └── road-anomaly/      # best.pt + Readme.md (4 classes)
```

---

## 3. ML Pipeline & Model Specifications

The backend operates four dedicated YOLOv8 deep learning models in parallel:

| Model ID | Weight File | Total Classes | Model Architecture | Key Classes |
| :--- | :--- | :--- | :--- | :--- |
| `road_sign` | `weights/road-sign/best.pt` | 35 | YOLOv8n | Speed Limit 10-120, Stop, Red Light, Green Light, Pedestrian Crossing |
| `pothole` | `weights/pothole/yolov8s-350-epoch-coslrtrue.pt` | 4 | YOLOv8s | `alligator_crack`, `longitudinal_crack`, `Pothole`, `transverse_crack` |
| `lane_line` | `weights/lane-line/best-7.pt` | 13 | YOLOv8s-seg | `BUS LANE`, `Yellow Markings`, `Line 1`, `Line 2`, `Crossing`, `SLOW`, `Bicycle`, Arrows |
| `anomaly` | `weights/road-anomaly/best.pt` | 4 | YOLOv8n | `Accident`, `Car Fire`, `Fighting`, `Snatching` |

### Class Name Resolution Rule
To prevent raw PyTorch model labels (e.g. `class__2`) from leaking into output annotations, each module checks its custom class dictionary (`ROAD_SIGN_CLASSES`, `POTHOLE_CLASSES`, `LANE_LINE_CLASSES`, `ANOMALY_CLASSES`) matching the model's `Readme.md` documentation before querying `model.names`.

---

## 4. Fusion Layer (`ml_pipeline/fusion_layer.py`)

The `FusionManager` coordinates multi-model execution, priority sorting, bounding box duplicate suppression, and JSON serialization.

### A. Priority Taxonomy (`CLASS_PRIORITY_MAP`)
- **Rank 1: CRITICAL** (Color: Red `[0, 0, 255]`) — Pothole, Accident, Car Fire, Stop, Red Light.
- **Rank 2: HIGH** (Color: Orange `[0, 165, 255]`) — Alligator Crack, Fighting, Snatching, Pedestrian Crossing, Bus Lane, Crossing, Slow, Bicycle.
- **Rank 3: MEDIUM** (Color: Amber `[0, 215, 255]`) — Longitudinal Crack, Transverse Crack, Yellow Markings, Line 1, Line 2, Romb, Warning signs.
- **Rank 4: LOW** (Color: Cyan `[255, 255, 0]`) — Green Light, Bus Stop, Hospital, Directional Arrows.

### B. Duplicate Bounding Box Suppression
Computes Intersection-over-Union (IoU) between bounding boxes. Boxes with **IoU > 0.5** are deduplicated, preserving the highest confidence and highest priority detection.

### C. NumPy JSON Serialization (`sanitize_for_json()`)
Recursively converts `numpy.int32`, `numpy.int64`, `numpy.float32`, and `numpy.ndarray` objects into Python native `int`, `float`, and `list` types to prevent WebSocket JSON serialization errors.

---

## 5. API Endpoints & Protocols

### A. Root Endpoint: `GET /`
Returns backend operational status and currently active confidence thresholds.

### B. Image Processing Endpoint: `POST /api/process-image`
- **Payload**: `file` (multipart image), `models` (comma-separated list, e.g. `anomaly,lane_line,pothole,road_sign`).
- **Response**: JSON with Base64 annotated image URL, detection count, and priority breakdown.

### C. Video Processing Endpoint: `POST /api/process-video`
- **Payload**: `file` (multipart MP4/WEBM video), `models` (comma-separated list).
- **Processing**: Renders OpenCV priority boxes across video frames and writes an annotated MP4 to `static/videos/annotated_<timestamp>.mp4`.
- **Response**: JSON containing `annotated_video_url`, total frame count, FPS, and sample timeline alert array.

### D. WebSocket Streaming Endpoint: `WS /ws/video`
- **Request Format**:
  ```json
  {
    "command": "process_frame",
    "active_models": ["anomaly", "lane_line", "pothole", "road_sign"],
    "frame_idx": 1.25,
    "timestamp": 1.25,
    "frame_b64": "data:image/jpeg;base64,..."
  }
  ```
- **Response Format**:
  ```json
  {
    "status": "success",
    "frame_idx": 1.25,
    "timestamp": 1.25,
    "active_models": ["anomaly", "lane_line", "pothole", "road_sign"],
    "highest_priority": "anomaly",
    "audio_trigger": true,
    "detections": [...],
    "annotated_frame": "data:image/jpeg;base64,..."
  }
  ```

---

## 6. Environment Configuration (`.env`)

```env
# YOLO Model Inference Confidence Thresholds (0.01 to 1.00)
CONF_ROAD_SIGN=0.15
CONF_POTHOLE=0.15
CONF_ANOMALY=0.15
CONF_LANE_LINE=0.15
DEFAULT_MODEL_CONFIDENCE=0.15

# Post-Inference Detection Filter Confidence Thresholds (0.01 to 1.00)
DET_CONF_ROAD_SIGN=0.15
DET_CONF_POTHOLE=0.15
DET_CONF_ANOMALY=0.15
DET_CONF_LANE_LINE=0.15
DEFAULT_DETECTION_CONFIDENCE=0.15
```

---

## 7. Command Execution & Startup

```bash
cd dashcam-safety-monitor-backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py
```
Backend runs on `http://0.0.0.0:8000`.
