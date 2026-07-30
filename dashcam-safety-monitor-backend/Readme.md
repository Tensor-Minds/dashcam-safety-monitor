# Dash Cam Safety Monitor Backend

FastAPI & PyTorch multi-model vision backend for real-time hazard detection, video streaming, and priority audio alerts.

---

## Environment Configurations (`.env`)

Configure the environment variables in `.env` to customize YOLO model inference confidence thresholds and post-inference detection filters.

### 1. YOLO Model Inference Confidence Thresholds (`CONF_*`)
Controls the minimum probability required for YOLO candidate bounding boxes during PyTorch model forward inference.

| Variable Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `CONF_ROAD_SIGN` | `float` | `0.15` | Inference confidence threshold for Traffic Sign Detection (`best.pt`) |
| `CONF_POTHOLE` | `float` | `0.15` | Inference confidence threshold for Pothole Detection (`yolov8s-350-epoch-coslrtrue.pt`) |
| `CONF_ANOMALY` | `float` | `0.15` | Inference confidence threshold for Road Anomaly Detection (`best.pt`) |
| `CONF_LANE_LINE` | `float` | `0.15` | Inference confidence threshold for Lane Line & Marking Detection (`best.pt`) |
| `DEFAULT_MODEL_CONFIDENCE` | `float` | `0.15` | Default fallback threshold if a model-specific `CONF_*` variable is omitted |

---

### 2. Post-Inference Detection Filter Confidence Thresholds (`DET_CONF_*`)
Filters out detected bounding boxes whose confidence score is lower than this threshold before returning results or rendering output video frames.

| Variable Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `DET_CONF_ROAD_SIGN` | `float` | `0.15` | Post-inference filter threshold for Road Signs |
| `DET_CONF_POTHOLE` | `float` | `0.15` | Post-inference filter threshold for Pothole hazards |
| `DET_CONF_ANOMALY` | `float` | `0.15` | Post-inference filter threshold for Road Anomalies |
| `DET_CONF_LANE_LINE` | `float` | `0.15` | Post-inference filter threshold for Lane Lines & Markings |
| `DEFAULT_DETECTION_CONFIDENCE` | `float` | `0.15` | Default fallback threshold if a model-specific `DET_CONF_*` variable is omitted |

---

## Quick Setup

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file
cp .env.example .env

# 4. Start FastAPI server
python3 main.py
```