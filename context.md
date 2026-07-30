# Master Context & System Architecture (`dashcam-safety-monitor`)

## 1. Project Overview
The **Dashcam Safety Monitor** is an end-to-end, multi-model AI vision application for real-time road hazard detection, traffic sign compliance, pothole monitoring, and lane departure warnings. 

It consists of:
1. **Backend Service (`dashcam-safety-monitor-backend`)**: Built with **FastAPI**, **PyTorch**, **Ultralytics YOLOv8**, and **OpenCV**.
2. **Frontend Dashboard (`dashcam-safety-monitor-frontend`)**: Built with **Next.js 16**, **React**, **TailwindCSS**, and **Lucide React**.

---

## 2. Architecture & Data Flow

```
+-----------------------------------------------------------------------------------+
|                            FRONTEND (Next.js Dashboard)                           |
|  - Drag & Drop Media Uploader (Images & MP4 Videos)                              |
|  - Active Model Selection Filter Pills & Grid                                    |
|  - Real-Time WebSocket Streaming Overlay Player (~15 FPS)                          |
|  - Non-blocking Emergency Audio Alerts (alert.mp3)                               |
+-----------------------------------------------------------------------------------+
                               |                        ^
                 HTTP REST /   |                        | WebSocket Stream /
                 FormData      v                        | JSON Frame Payloads
+-----------------------------------------------------------------------------------+
|                            BACKEND (FastAPI + PyTorch)                            |
|                                                                                   |
|  [FusionManager]                                                                 |
|     ├── RoadSignDetector     (weights/road-sign/best.pt - 35 classes)             |
|     ├── PotholeDetector      (weights/pothole/yolov8s-350-epoch-coslrtrue.pt)    |
|     ├── LaneLineDetector     (weights/lane-line/best.pt - 13 classes)          |
|     └── AnomalyDetector      (weights/road-anomaly/best.pt - 4 classes)           |
|                                                                                   |
|  [Post-Processing]                                                                |
|     ├── IoU > 0.5 Bounding Box Deduplication                                     |
|     ├── Fine-Grained Per-Class Priority Taxonomy (Ranks 1 to 4)                  |
|     └── sanitize_for_json() NumPy Serializer                                     |
+-----------------------------------------------------------------------------------+
```

---

## 3. Key Backend Specifications ([`dashcam-safety-monitor-backend/context.md`](file:///Users/paranietharan/Documents/Codes/Tensor-Minds/dashcam-safety-monitor/dashcam-safety-monitor-backend/context.md))
- **Parallel Models**: Runs `anomaly`, `pothole`, `lane_line`, and `road_sign` in parallel.
- **Class Mapping Rule**: Resolves exact class names from `Readme.md` before querying `model.names` to eliminate generic labels like `class__2`.
- **Environment Configuration**: Configured via `.env` with separate `CONF_*` (inference confidence) and `DET_CONF_*` (post-inference filter) settings.

---

## 4. Key Frontend Specifications ([`dashcam-safety-monitor-frontend/context.md`](file:///Users/paranietharan/Documents/Codes/Tensor-Minds/dashcam-safety-monitor/dashcam-safety-monitor-frontend/context.md))
- **Live WebSocket Stream**: Streams canvas-captured frames over `WS /ws/video` with real-time overlay image rendering.
- **Server Video Processing UI**: High-tech glassmorphism modal with animated AI scanner, elapsed timer (`formatTimer`), and step checklist.
- **State Synchronization**: `onPlay` and `onPause` handlers keep native video playback and WebSocket frame processing strictly synchronized.

---

## 5. Development Quickstart

### Backend:
```bash
cd dashcam-safety-monitor-backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

### Frontend:
```bash
cd dashcam-safety-monitor-frontend
npm install
npm run dev
```
