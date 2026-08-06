import io
import os
import cv2
import json
import time
import base64
import asyncio
import tempfile
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# Load environment variables from .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles

from ml_pipeline.fusion_layer import FusionManager, sanitize_for_json

SUPPORTED_IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
    ".avif",
)


def decode_uploaded_image(contents: bytes) -> Optional[np.ndarray]:
    """Decode common image formats, using Pillow when OpenCV cannot read them."""
    if not contents:
        return None

    frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if frame is not None:
        return frame

    try:
        with Image.open(io.BytesIO(contents)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, ValueError):
        return None

app = FastAPI(
    title="Dashcam Road Safety Monitoring API",
    description="Backend service for multi-model YOLOv8 road safety hazard detection and real-time streaming.",
    version="1.0.0"
)

# Robust CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static video export directory exists and mount static route
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
VIDEOS_DIR = os.path.join(STATIC_DIR, "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Instantiate central ML fusion manager
fusion_manager = FusionManager()


@app.get("/")
def read_root():
    cfg_data = fusion_manager.rule_config.data
    return {
        "status": "online",
        "service": "Dashcam Road Safety Monitor API",
        "available_models": ["road_sign", "pothole", "lane_line", "anomaly"],
        "model_confidence_thresholds": {
            "road_sign": cfg_data["road_sign_filter"]["minimum_confidence"],
            "pothole": cfg_data["pothole_filter"]["minimum_confidence"],
            "anomaly": cfg_data["anomaly_filter"]["minimum_confidence"],
            "lane_line": cfg_data["lane"]["minimum_confidence"],
        },
        "detection_confidence_thresholds": {
            "road_sign": cfg_data["road_sign_filter"]["minimum_confidence"],
            "pothole": cfg_data["pothole_filter"]["minimum_confidence"],
            "anomaly": cfg_data["anomaly_filter"]["minimum_confidence"],
            "lane_line": cfg_data["lane"]["minimum_confidence"],
        },
        "rule_version": cfg_data["version"],
        "priority_policy": cfg_data["priority_policy"],
        "configured_rule_count": len(fusion_manager.rule_config.rules),
    }


@app.get("/api/rules")
def get_alert_rules():
    """Return the complete editable rule set; higher numeric priority wins."""
    return fusion_manager.rule_config.public_data()


@app.put("/api/rules")
def update_alert_rules(rule_set: Dict[str, Any]):
    """Validate, persist, and activate a complete replacement rule set."""
    try:
        fusion_manager.replace_rules(rule_set)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=422, detail=str(error))
    return {
        "status": "updated",
        "version": fusion_manager.rule_config.data["version"],
        "priority_policy": fusion_manager.rule_config.data["priority_policy"],
        "rule_count": len(fusion_manager.rule_config.rules),
    }


@app.post("/api/process-image")
async def process_image(
    file: UploadFile = File(...),
    models: str = Form("road_sign,pothole,lane_line,anomaly"),
    turn_signal: str = Form("off"),
    simulated_vehicle_speed_kmh: Optional[float] = Form(None, ge=0, le=300),
):
    filename = (file.filename or "").lower()
    has_image_type = bool(
        file.content_type and file.content_type.startswith("image/")
    )
    has_image_extension = filename.endswith(SUPPORTED_IMAGE_EXTENSIONS)
    if not (has_image_type or has_image_extension):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        frame = decode_uploaded_image(contents)

        if frame is None:
            file_type = file.content_type or "unknown type"
            raise HTTPException(
                status_code=400,
                detail=(
                    f'Could not decode "{file.filename or "uploaded image"}" '
                    f"({file_type}). Upload a valid JPEG, PNG, WebP, BMP, TIFF, "
                    "GIF, or AVIF image. Convert HEIC/HEIF photos to JPEG first."
                ),
            )

        active_models_list = [m.strip() for m in models.split(",") if m.strip()]

        annotated_frame, detections, highest_priority, audio_trigger = fusion_manager.process_frame(
            frame=frame,
            active_models=active_models_list,
            is_video=False,
            turn_signal=turn_signal,
            vehicle_speed_kmh=simulated_vehicle_speed_kmh,
        )

        _, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{jpg_as_text}"

        return JSONResponse({
            "status": "success",
            "active_models": active_models_list,
            "highest_priority": highest_priority,
            "audio_trigger": audio_trigger,
            "total_detections": len(detections),
            "detections": sanitize_for_json(detections),
            "primary_alert": sanitize_for_json(fusion_manager.last_primary_alert),
            "image_quality": sanitize_for_json(fusion_manager.last_image_quality),
            "annotated_image": data_url
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")


@app.post("/api/process-video")
async def process_video(
    file: UploadFile = File(...),
    models: str = Form("road_sign,pothole,lane_line,anomaly"),
    turn_signal: str = Form("off"),
    simulated_vehicle_speed_kmh: Optional[float] = Form(None, ge=0, le=300),
):
    if not (file.content_type.startswith("video/") or file.filename.endswith((".mp4", ".avi", ".mov", ".webm"))):
        raise HTTPException(status_code=400, detail="Uploaded file must be a video.")

    temp_video_path = None
    writer = None
    try:
        suffix = os.path.splitext(file.filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_video_path = tmp.name

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Failed to open video file.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        active_models_list = [m.strip() for m in models.split(",") if m.strip()]

        out_filename = f"annotated_{int(time.time())}.mp4"
        out_filepath = os.path.join(VIDEOS_DIR, out_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_filepath, fourcc, fps, (width, height))

        fusion_manager.reset_state()
        timeline_alerts = []
        overall_highest_priority = "normal"
        overall_highest_rank = 99
        audio_trigger_count = 0
        category_counts = {"anomaly": 0, "lane_line": 0, "pothole": 0, "road_sign": 0}

        frame_idx = 0
        target_fps = fusion_manager.rule_config.system["processing_target_fps"]
        sample_stride = max(1, round(fps / target_fps))
        latest_detections = []
        latest_primary_alert = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = frame_idx / fps * 1000
            if frame_idx % sample_stride == 0:
                annotated_frame, detections, highest_priority, audio_trigger = fusion_manager.process_frame(
                    frame=frame,
                    active_models=active_models_list,
                    is_video=True,
                    timestamp_ms=timestamp_ms,
                    turn_signal=turn_signal,
                    vehicle_speed_kmh=simulated_vehicle_speed_kmh,
                )
                latest_detections = detections
                latest_primary_alert = fusion_manager.last_primary_alert
            else:
                visible_alert = (
                    latest_primary_alert
                    if latest_primary_alert
                    and latest_primary_alert.get("visible_until_ms", 0) >= timestamp_ms
                    else None
                )
                annotated_frame = fusion_manager.render_annotations(
                    frame,
                    latest_detections,
                    visible_alert,
                    fusion_manager.last_image_quality,
                )
                detections = []
                highest_priority = visible_alert["category"] if visible_alert else "normal"
                audio_trigger = False

            if writer:
                writer.write(annotated_frame)

            if audio_trigger:
                audio_trigger_count += 1

            for d in detections:
                cat = d.get("category", "")
                if cat in category_counts:
                    category_counts[cat] += 1
                rank = d.get("priority_rank", 99)
                if rank < overall_highest_rank:
                    overall_highest_rank = rank
                    overall_highest_priority = cat

            if frame_idx % sample_stride == 0 and (detections or frame_idx % (sample_stride * 5) == 0):
                timestamp = frame_idx / fps
                _, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                out_b64 = base64.b64encode(buffer).decode("utf-8")
                data_url = f"data:image/jpeg;base64,{out_b64}"

                timeline_alerts.append({
                    "frame_idx": frame_idx,
                    "timestamp": round(timestamp, 2),
                    "highest_priority": highest_priority,
                    "audio_trigger": audio_trigger,
                    "total_detections": len(detections),
                    "detections": sanitize_for_json(detections),
                    "primary_alert": sanitize_for_json(fusion_manager.last_primary_alert),
                    "image_quality": sanitize_for_json(fusion_manager.last_image_quality),
                    "annotated_frame": data_url
                })

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        annotated_video_url = f"/static/videos/{out_filename}"

        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "processed_samples": len(timeline_alerts),
            "active_models": active_models_list,
            "highest_priority": overall_highest_priority,
            "audio_trigger_count": audio_trigger_count,
            "category_counts": category_counts,
            "annotated_video_url": annotated_video_url,
            "timeline_alerts": timeline_alerts
        })

    except Exception as e:
        if writer:
            writer.release()
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass


@app.websocket("/ws/video")
async def websocket_video_endpoint(websocket: WebSocket):
    await websocket.accept()
    fusion_manager.reset_state()
    print("[WebSocket] Client connected for video stream processing.")

    try:
        await websocket.send_json({"status": "connected", "message": "WebSocket Safety Stream Ready"})
    except Exception:
        pass

    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                
                command = payload.get("command", "process_frame")

                if command == "ping":
                    await websocket.send_json({"status": "pong"})
                    continue

                if command == "reset":
                    fusion_manager.reset_state()
                    await websocket.send_json({"status": "reset_complete"})
                    continue

                active_models = payload.get("active_models", ["road_sign", "pothole", "lane_line", "anomaly"])
                frame_idx = payload.get("frame_idx", 0)
                timestamp = payload.get("timestamp", 0.0)
                turn_signal = payload.get("turn_signal", "off")
                include_annotated_frame = payload.get("include_annotated_frame", True)
                vehicle_speed_kmh = payload.get("simulated_vehicle_speed_kmh")
                if vehicle_speed_kmh is not None:
                    vehicle_speed_kmh = float(vehicle_speed_kmh)
                    if not 0 <= vehicle_speed_kmh <= 300:
                        await websocket.send_json(
                            {"status": "error", "message": "Simulated speed must be between 0 and 300 km/h"}
                        )
                        continue

                frame_b64 = payload.get("frame_b64")
                if not frame_b64:
                    await websocket.send_json({"error": "Missing frame_b64 in payload"})
                    continue

                if "," in frame_b64:
                    frame_b64 = frame_b64.split(",")[1]

                img_bytes = base64.b64decode(frame_b64)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    await websocket.send_json({"error": "Invalid image frame encoding"})
                    continue

                await asyncio.sleep(0.001)

                annotated_frame, detections, highest_priority, audio_trigger = fusion_manager.process_frame(
                    frame=frame,
                    active_models=active_models,
                    is_video=True,
                    timestamp_ms=float(timestamp) * 1000,
                    turn_signal=turn_signal,
                    vehicle_speed_kmh=vehicle_speed_kmh,
                )

                data_url = None
                if include_annotated_frame:
                    _, buffer = cv2.imencode(
                        ".jpg",
                        annotated_frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 80],
                    )
                    out_b64 = base64.b64encode(buffer).decode("utf-8")
                    data_url = f"data:image/jpeg;base64,{out_b64}"

                response_payload = {
                    "status": "success",
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                    "active_models": active_models,
                    "highest_priority": highest_priority,
                    "audio_trigger": audio_trigger,
                    "detections": sanitize_for_json(detections),
                    "primary_alert": sanitize_for_json(fusion_manager.last_primary_alert),
                    "image_quality": sanitize_for_json(fusion_manager.last_image_quality),
                    "annotated_frame": data_url
                }

                await websocket.send_json(response_payload)
            except Exception as frame_err:
                print(f"[WebSocket] Inner frame processing error: {frame_err}")
                try:
                    await websocket.send_json({"status": "error", "message": str(frame_err)})
                except Exception:
                    pass

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected.")
    except Exception as e:
        print(f"[WebSocket] Outer disconnect: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
