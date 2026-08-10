# Dashcam Safety Monitor - Configuration Guide (`config.md`)

This document provides a detailed breakdown of the `rules.yml` configuration schema used by the Multi-Model YOLOv8 AI Pipeline and Rule Evaluation Engine.

---

## 🖥️ System Settings (`system`)

Global parameters for video sampling, object tracking, image quality checks, and alert presentation.

```yaml
system:
  processing_target_fps: 0
  tracking_iou_threshold: 0.3
  duplicate_iou_threshold: 0.5
  visual_duration_ms: 2000
  maximum_track_misses: 2
  minimum_blur_score: 5
  minimum_mean_brightness: 5
  maximum_mean_brightness: 250
  image_quality_status_cooldown_ms: 10000
```

| Parameter | Default | Range | Description |
| :--- | :--- | :--- | :--- |
| `processing_target_fps` | `0` | $\ge 0$ | Target frame rate for video AI processing. Set to `0` to process **100% of frames at native video FPS**. Set to `10` to sample ~10 FPS for CPU savings. |
| `tracking_iou_threshold` | `0.3` | $0.0 - 1.0$ | Minimum Intersection-over-Union (IoU) overlap to track the same object across consecutive video frames. |
| `duplicate_iou_threshold` | `0.5` | $0.0 - 1.0$ | Overlap threshold to suppress duplicate alert bounding boxes when multi-model fusion produces overlapping detections. |
| `visual_duration_ms` | `2000` | $\ge 0$ | Time (in milliseconds) that an active alert banner remains visible on screen once triggered. |
| `maximum_track_misses` | `2` | $\ge 0$ | Maximum consecutive frames an object can be missing before its spatial tracking record is deleted. |
| `minimum_blur_score` | `5` | $\ge 0$ | Minimum Laplacian variance score for image sharpness. Scores below this trigger a low visibility warning. |
| `minimum_mean_brightness` | `5` | $0 - 255$ | Minimum acceptable average pixel brightness (prevents alerts in pitch-black video). |
| `maximum_mean_brightness` | `250` | $0 - 255$ | Maximum acceptable average pixel brightness (prevents alerts in overexposed video). |
| `image_quality_status_cooldown_ms` | `10000` | $\ge 0$ | Cooldown period (in ms) between camera visibility warning popups. |

---

## 🚦 Road Sign Filter (`road_sign_filter`)

Filters and temporal confirmation criteria for road sign detection models (Speed Limit, Stop, Red/Green Light, Warning Signs).

```yaml
road_sign_filter:
  minimum_confidence: 0.15
  nms_iou_threshold: 0.45
  minimum_aspect_ratio: 0.2
  maximum_aspect_ratio: 2.5
  minimum_frame_area_ratio: 0.0001
  window_frames: 3
  minimum_hits: 1
```

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `minimum_confidence` | `0.15` | Minimum YOLO model confidence threshold (15%) to accept road sign detections. |
| `nms_iou_threshold` | `0.45` | Non-Maximum Suppression (NMS) IoU threshold to eliminate overlapping duplicate boxes for the same sign. |
| `minimum_aspect_ratio` | `0.2` | Minimum bounding box width-to-height ratio (filters invalid thin vertical artifacts). |
| `maximum_aspect_ratio` | `2.5` | Maximum bounding box width-to-height ratio (filters invalid wide horizontal artifacts). |
| `minimum_frame_area_ratio` | `0.0001` | Minimum box area relative to total image area (filters tiny background noise). |
| `window_frames` | `3` | Number of consecutive frames evaluated in the temporal tracking window. |
| `minimum_hits` | `1` | Minimum positive detections required within `window_frames` to confirm a sign. |

---

## 🕳️ Pothole & Road Damage Filter (`pothole_filter`)

Filtering criteria for road surface hazards (Potholes, Alligator Cracks, Longitudinal Cracks, Transverse Cracks).

```yaml
pothole_filter:
  minimum_confidence: 0.15
  nms_iou_threshold: 0.4
  window_frames: 3
  minimum_hits: 2
```

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `minimum_confidence` | `0.15` | Minimum confidence threshold (15%) for road damage detections. |
| `nms_iou_threshold` | `0.4` | NMS IoU threshold for duplicate pothole bounding box suppression. |
| `window_frames` | `3` | Temporal tracking window size in frames. |
| `minimum_hits` | `2` | Minimum detections required in the temporal window to confirm a pothole hazard. |

---

## 🚨 Road Anomaly Filter (`anomaly_filter`)

Filtering criteria for critical road incidents (Accidents, Car Fires, Fighting, Snatching).

```yaml
anomaly_filter:
  minimum_confidence: 0.5
  nms_iou_threshold: 0.4
  window_frames: 3
  minimum_hits: 1
```

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `minimum_confidence` | `0.5` | Minimum confidence threshold (50%) for critical road anomaly detections. |
| `nms_iou_threshold` | `0.4` | NMS IoU threshold for duplicate anomaly box suppression. |
| `window_frames` | `3` | Temporal tracking window size in frames. |
| `minimum_hits` | `1` | Minimum detections required in the temporal window to trigger an incident alert. |

---

## 🛣️ Lane Departure & Centering (`lane`)

Configuration for vehicle lane position estimation, polynomial fitting, and departure warning hysteresis.

```yaml
lane:
  minimum_confidence: 0.15
  ema_alpha: 0.7
  trigger_offset_ratio: 0.35
  clear_offset_ratio: 0.2
  minimum_duration_ms: 1500
```

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `minimum_confidence` | `0.15` | Minimum confidence threshold (15%) for lane line segment detection. |
| `ema_alpha` | `0.7` | Exponential Moving Average (EMA) smoothing weight ($0 < \alpha \le 1$) applied to lane position offset ratio. |
| `trigger_offset_ratio` | `0.35` | Vehicle drift threshold (35% of lane width from center) to **trigger** a Lane Departure Warning. |
| `clear_offset_ratio` | `0.2` | Vehicle drift threshold (20% of lane width from center) to **clear/reset** a Lane Departure Warning (Hysteresis). |
| `minimum_duration_ms` | `1500` | Minimum continuous lane drift duration (1.5 seconds) required before issuing a departure warning. |
