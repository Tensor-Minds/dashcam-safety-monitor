import io
import os
import cv2
import json
import time
import base64
import asyncio
import tempfile
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles

from ml_pipeline.fusion_layer import FusionManager, sanitize_for_json

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
    return {
        "status": "online",
        "service": "Dashcam Road Safety Monitor API",
        "available_models": ["road_sign", "pothole", "lane_line", "anomaly"]
    }


@app.post("/api/process-image")
async def process_image(
    file: UploadFile = File(...),
    models: str = Form("road_sign,pothole,lane_line,anomaly")
):
    if not (file.content_type and file.content_type.startswith("image/")):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to decode image file.")

        active_models_list = [m.strip() for m in models.split(",") if m.strip()]

        annotated_frame, detections, highest_priority, audio_trigger = fusion_manager.process_frame(
            frame=frame,
            active_models=active_models_list,
            is_video=False
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
            "annotated_image": data_url
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")


@app.post("/api/process-video")
async def process_video(
    file: UploadFile = File(...),
    models: str = Form("road_sign,pothole,lane_line,anomaly")
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
        sample_stride = max(1, int(fps / 10))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, detections, highest_priority, audio_trigger = fusion_manager.process_frame(
                frame=frame,
                active_models=active_models_list,
                is_video=True
            )

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
                    is_video=True
                )

                _, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
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
