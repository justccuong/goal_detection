import io
import os
import cv2
import numpy as np
import base64
import uuid
import shutil
import subprocess
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import uvicorn
from highlight_algorithm import analyze_frame_highlight, merge_time_windows

app = FastAPI(title="SmartPlay Football Highlight API", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "temp_uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "temp_outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# In-memory database of active video analysis jobs
jobs = {}

# Load custom YOLOv8 model for ball and goalpost detection
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
try:
    model = YOLO(MODEL_PATH)
    print(f"Successfully loaded custom model from {MODEL_PATH}")
except Exception as e:
    print(f"Failed to load custom model from {MODEL_PATH}, trying local fallback...")
    if os.path.exists(os.path.join(BASE_DIR, "best.pt")):
        model = YOLO(os.path.join(BASE_DIR, "best.pt"))
    elif os.path.exists("best.pt"):
        model = YOLO("best.pt")
    else:
        print("Could not find best.pt custom model locally, trying standard yolov8s.pt fallback.")
        model = YOLO("yolov8s.pt")

# Custom class mapping for best.pt (0: ball, 1: goalpost)
CLASS_MAPPINGS = {
    0: {"name": "Ball", "type": "ball", "color": (0, 255, 255)},        # Neon Yellow-ish (BGR)
    1: {"name": "Goalpost", "type": "goalpost", "color": (255, 255, 0)}, # Cyan (BGR)
}

def draw_hud_box(img, x1, y1, x2, y2, label, conf, color, is_threat):
    # Draw bounding box outline
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    
    # Draw corner brackets (HUD Targeting style)
    bracket_len = min(20, int((x2 - x1) * 0.2), int((y2 - y1) * 0.2))
    if bracket_len > 2:
        cv2.line(img, (x1, y1), (x1 + bracket_len, y1), color, 2)
        cv2.line(img, (x1, y1), (x1, y1 + bracket_len), color, 2)
        cv2.line(img, (x2, y1), (x2 - bracket_len, y1), color, 2)
        cv2.line(img, (x2, y1), (x2, y1 + bracket_len), color, 2)
        cv2.line(img, (x1, y2), (x1 + bracket_len, y2), color, 2)
        cv2.line(img, (x1, y2), (x1, y2 - bracket_len), color, 2)
        cv2.line(img, (x2, y2), (x2 - bracket_len, y2), color, 2)
        cv2.line(img, (x2, y2), (x2, y2 - bracket_len), color, 2)

    # Draw background for text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    text = f"{label.upper()} {conf:.0%}"
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    tx = x1
    ty = y1 - 6 if y1 - 6 > text_h + 4 else y1 + text_h + 6
    
    cv2.rectangle(img, (tx, ty - text_h - 4), (tx + text_w + 6, ty + baseline), color, -1)
    cv2.putText(img, text, (tx + 3, ty - 2), font, font_scale, (0, 0, 0) if sum(color) > 380 else (255, 255, 255), thickness, cv2.LINE_AA)

def process_video_background(job_id: str, video_path: str):
    jobs[job_id] = {
        "status": "processing",
        "progress": 0.0,
        "video_path": video_path,
        "duration": 0.0,
        "fps": 0.0,
        "intervals": [],
        "error": None
    }
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Failed to open video file")
            
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        jobs[job_id]["duration"] = duration
        jobs[job_id]["fps"] = fps
        
        # Process at 2 frames per second (fps / 2) to keep scanning speed high
        sample_interval = max(1, int(fps / 2))
        
        highlight_frames = []
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % sample_interval == 0:
                current_time = frame_idx / fps
                
                results = model(frame, verbose=False)
                detections = []
                
                for result in results:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        if cls_id in CLASS_MAPPINGS:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            detections.append({
                                "class_id": cls_id,
                                "confidence": conf,
                                "bbox": [x1, y1, x2, y2]
                            })
                
                highlight_level, _ = analyze_frame_highlight(detections)
                if highlight_level in ["GOAL", "NEAR_MISS"]:
                    highlight_frames.append({
                        "time": current_time,
                        "type": highlight_level,
                        "conf": max([d["confidence"] for d in detections]) if detections else 0.0,
                        "frame": frame.copy(),
                        "detections": detections
                    })
            
            frame_idx += 1
            jobs[job_id]["progress"] = min(0.99, frame_idx / frame_count)
            
        cap.release()
        
        # Segment clustering using the new algorithm
        intervals = merge_time_windows(highlight_frames, window_duration=8.0, total_duration=duration)
                
        # Format segments
        formatted_intervals = []
        for i, inter in enumerate(intervals):
            peak_img = inter["peak_frame"].copy()
            for det in inter["peak_detections"]:
                x1, y1, x2, y2 = det["bbox"]
                mapping = CLASS_MAPPINGS[det["class_id"]]
                draw_hud_box(peak_img, x1, y1, x2, y2, mapping["name"], det["confidence"], mapping["color"], False)
                
            _, buffer = cv2.imencode('.jpg', peak_img)
            thumbnail_base64 = base64.b64encode(buffer).decode('utf-8')
            
            desc = ""
            if inter["type"] == "GOAL":
                desc = f"Goal event. Ball entered goalmouth area. (Conf: {inter['peak_conf']:.0%})"
            else:
                desc = f"Goal attempt. Ball close to goalpost. (Conf: {inter['peak_conf']:.0%})"
                
            formatted_intervals.append({
                "id": i,
                "start": round(inter["start"], 1),
                "end": round(inter["end"], 1),
                "type": inter["type"],
                "confidence": inter["peak_conf"],
                "thumbnail": f"data:image/jpeg;base64,{thumbnail_base64}",
                "description": desc
            })
            
        jobs[job_id]["intervals"] = formatted_intervals
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

class IntervalExport(BaseModel):
    start: float
    end: float

class ExportPayload(BaseModel):
    job_id: str
    intervals: List[IntervalExport]

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "model": "YOLOv8 Custom Goal/Ball Tracker",
        "classes_mapped": [info["name"] for info in CLASS_MAPPINGS.values()]
    }

@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        annotated_img = img.copy()
        results = model(img, verbose=False)
        detections = []
        ball_count = 0
        goalpost_count = 0
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id in CLASS_MAPPINGS:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    mapping = CLASS_MAPPINGS[cls_id]
                    
                    if cls_id == 0:
                        ball_count += 1
                    else:
                        goalpost_count += 1
                    
                    detections.append({
                        "class_id": cls_id,
                        "original_name": model.names[cls_id],
                        "mapped_name": mapping["name"],
                        "type": mapping["type"],
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2]
                    })
                    
                    draw_hud_box(
                        annotated_img, 
                        x1, y1, x2, y2, 
                        mapping["name"], 
                        conf, 
                        mapping["color"], 
                        False
                    )
        
        highlight_level, instruction = analyze_frame_highlight(detections)
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "threat_level": highlight_level,
            "threat_count": ball_count,
            "ally_count": goalpost_count,
            "detections": detections,
            "annotated_image": f"data:image/jpeg;base64,{img_base64}",
            "instruction": instruction
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    try:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".mp4", ".avi", ".mov", ".mkv"]:
            raise HTTPException(status_code=400, detail="Unsupported video format")
            
        job_id = str(uuid.uuid4())
        video_filename = f"{job_id}{ext}"
        video_path = os.path.join(UPLOAD_DIR, video_filename)
        
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        background_tasks.add_task(process_video_background, job_id, video_path)
        
        return {
            "job_id": job_id,
            "status": "pending",
            "message": "Video uploaded successfully. Scanning started."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status_video/{job_id}")
def status_video(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.post("/api/export_video")
async def export_video(payload: ExportPayload):
    try:
        job_id = payload.job_id
        intervals = payload.intervals
        
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
            
        video_path = jobs[job_id]["video_path"]
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Original video file not found on server")
            
        if not intervals:
            raise HTTPException(status_code=400, detail="No intervals selected for export")
            
        export_id = str(uuid.uuid4())
        export_subfolder = os.path.join(OUTPUT_DIR, export_id)
        os.makedirs(export_subfolder, exist_ok=True)
        
        clip_files = []
        for idx, interval in enumerate(intervals):
            start = interval.start
            end = interval.end
            
            clip_name = f"clip_{idx}.mp4"
            clip_path = os.path.join(export_subfolder, clip_name)
            
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-to", str(end),
                "-i", video_path,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-avoid_negative_ts", "make_zero",
                "-fflags", "+genpts",
                clip_path
            ]
            
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                raise Exception(f"Failed to cut segment {start} - {end}: {result.stderr}")
                
            clip_files.append(clip_path)
            
        output_filename = f"highlight_{export_id}.mp4"
        output_path = os.path.join(export_subfolder, output_filename)
        
        if len(clip_files) == 1:
            shutil.copy2(clip_files[0], output_path)
        else:
            list_path = os.path.join(export_subfolder, "list.txt")
            with open(list_path, "w") as f:
                for clip in clip_files:
                    f.write(f"file '{os.path.basename(clip)}'\n")
                    
            concat_cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", list_path,
                "-c", "copy",
                output_path
            ]
            
            concat_result = subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if concat_result.returncode != 0:
                raise Exception(f"Failed to concatenate segments: {concat_result.stderr}")
                
        if not os.path.exists(output_path):
            raise Exception("Exported file was not created successfully")
            
        return FileResponse(
            path=output_path,
            media_type="video/mp4",
            filename="smartplay_highlights.mp4"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
