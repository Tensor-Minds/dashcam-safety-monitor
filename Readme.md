# Dashcam Safety Monitor

Real-Time Multi-Model Vision Pipeline for Road Hazard Detection, Priority Bounding Box Overlay, and Audio Emergency Alerts.

---

## Architecture
![Architecture Diagram](./docs/architecture.png)

---

## Environment Configurations (`.env`)

Backend environment variables are configured in [`dashcam-safety-monitor-backend/.env`](file:///Users/paranietharan/Documents/Codes/Tensor-Minds/dashcam-safety-monitor/dashcam-safety-monitor-backend/.env).

### Backend Model & Detection Confidence Settings

| Variable Name | Type | Default | Stage | Description |
| :--- | :--- | :--- | :--- | :--- |
| `CONF_ROAD_SIGN` | `float` | `0.15` | Inference | Model confidence threshold for Road Signs |
| `CONF_POTHOLE` | `float` | `0.15` | Inference | Model confidence threshold for Potholes |
| `CONF_ANOMALY` | `float` | `0.15` | Inference | Model confidence threshold for Road Anomalies |
| `CONF_LANE_LINE` | `float` | `0.15` | Inference | Model confidence threshold for Lane Lines |
| `DEFAULT_MODEL_CONFIDENCE` | `float` | `0.15` | Inference | Fallback model confidence threshold |
| `DET_CONF_ROAD_SIGN` | `float` | `0.15` | Post-Filter | Output detection filter threshold for Road Signs |
| `DET_CONF_POTHOLE` | `float` | `0.15` | Post-Filter | Output detection filter threshold for Potholes |
| `DET_CONF_ANOMALY` | `float` | `0.15` | Post-Filter | Output detection filter threshold for Road Anomalies |
| `DET_CONF_LANE_LINE` | `float` | `0.15` | Post-Filter | Output detection filter threshold for Lane Lines |
| `DEFAULT_DETECTION_CONFIDENCE` | `float` | `0.15` | Post-Filter | Fallback detection filter threshold |

### Frontend Settings (`dashcam-safety-monitor-frontend/.env.local`)

| Variable Name | Default | Description |
| :--- | :--- | :--- |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | FastAPI Backend REST API Host |
| `NEXT_PUBLIC_WS_URL` | `ws://localhost:8000/ws/video` | Real-Time Video Safety Stream WebSocket Endpoint |