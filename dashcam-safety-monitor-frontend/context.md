# Frontend Context & Technical Documentation (`dashcam-safety-monitor-frontend`)

## 1. Overview
The **Dashcam Safety Monitor Frontend** is a modern Next.js (App Router) web application built with **React**, **TypeScript**, **TailwindCSS**, and **Lucide Icons**. It provides an interactive dashboard for multi-model road safety monitoring, real-time WebSocket video frame streaming, dynamic active model controls, and automated emergency audio alerts.

---

## 2. Directory Structure & File Map

```
dashcam-safety-monitor-frontend/
├── app/
│   ├── layout.tsx             # Global HTML layout & metadata
│   ├── page.tsx               # Primary application page (manages media state & host resolution)
│   └── globals.css            # Tailwind CSS directives & ambient glow utility styles
├── components/
│   ├── Dashboard.tsx          # Main Monitoring Console (WebSocket player, server video renderer, alert feed, sound)
│   ├── MediaUploader.tsx      # Drag & drop media intake component for images & videos
│   └── ModelSelector.tsx       # Model selection cards & toggle control pills
├── public/
│   └── alert.mp3              # High-priority emergency alert sound audio asset
├── package.json               # Next.js, React, Lucide-React dependencies
├── next.config.ts             # Next.js build configuration
├── tsconfig.json              # TypeScript path aliases (@/*)
└── context.md                 # Frontend technical documentation
```

---

## 3. Key Components & Application Flow

### A. Main Entry Page (`app/page.tsx`)
- **API Host Resolution (`getApiHost()`)**: Dynamically resolves the FastAPI backend host (`process.env.NEXT_PUBLIC_API_URL` -> `window.location.hostname:8000` -> `localhost:8000`).
- **State Management**:
  - `selectedModels`: Active model IDs (defaults to `["anomaly", "lane_line", "pothole", "road_sign"]`).
  - `mediaFile` & `mediaType`: Uploaded file instance and type (`"image"` | `"video"`).
  - `imageResult` & `videoResult`: REST response objects from backend.
- **Dynamic Image Re-processing**: Toggling models while viewing an image automatically re-submits the image payload to `/api/process-image`.

### B. Media Uploader (`components/MediaUploader.tsx`)
- Provides a drag-and-drop intake zone supporting images (`.jpg`, `.png`, `.webp`) and videos (`.mp4`, `.webm`, `.mov`).
- Validates file format and confirms active model selection before submitting.

### C. Model Selector (`components/ModelSelector.tsx`)
- Renders grid cards for all 4 YOLO models with priority badges and descriptions.
- Features **Select All** and **Clear All** utility actions.

### D. Safety Dashboard (`components/Dashboard.tsx`)
- **Server Video Processing Loader Modal**:
  - Displays a high-tech glassmorphism overlay when `isServerProcessingVideo === true`.
  - Features an animated central scanner radar, live elapsed timer formatted as `MM:SS` (`formatTimer`), and step-by-step checklist.
- **Live Video Streaming Player**:
  - Plays the uploaded video while an hidden canvas captures frames (~15 FPS) and sends Base64 payloads over `ws://localhost:8000/ws/video`.
  - Overlays live annotated frames (`liveAnnotatedFrame`) on top of the HTML5 `<video>` element.
  - Synchronized `onPlay` and `onPause` handlers keep native video controls and custom action buttons synchronized.
- **Audio Alerts (`playAlertSound`)**:
  - Plays `/alert.mp3` with non-blocking cooldown intervals (1.2s for CRITICAL anomalies, 1.8s for HIGH hazards) when `audio_trigger === true`.
- **Dynamic Model Control Bar**:
  - Allows live model toggling and includes a **"⚡ Run All Parallel"** action button.
- **Prioritized Hazard Feed Log**:
  - Displays real-time detection logs with priority filter dropdown (`ALL`, `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`).

---

## 4. REST & WebSocket Protocols

### A. Environment Configuration (`.env.local`)
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws/video
```

### B. WebSocket Frame Schema
- **Client Frame Request**:
  ```json
  {
    "command": "process_frame",
    "active_models": ["anomaly", "lane_line", "pothole", "road_sign"],
    "frame_idx": 4.5,
    "timestamp": 4.5,
    "frame_b64": "data:image/jpeg;base64,..."
  }
  ```
- **Server Frame Response**:
  ```json
  {
    "status": "success",
    "frame_idx": 4.5,
    "timestamp": 4.5,
    "highest_priority": "anomaly",
    "audio_trigger": true,
    "detections": [
      {
        "bbox": [100, 200, 300, 400],
        "confidence": 0.85,
        "class_name": "Accident",
        "category": "anomaly",
        "color": [0, 0, 255],
        "priority_rank": 1,
        "priority_level": "CRITICAL"
      }
    ],
    "annotated_frame": "data:image/jpeg;base64,..."
  }
  ```

---

## 5. Command Execution & Startup

```bash
cd dashcam-safety-monitor-frontend
npm install
npm run dev
```
Frontend runs on `http://localhost:3000`.

To build for production:
```bash
npm run build
```
