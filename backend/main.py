import io
import os
import zipfile
import cv2
import numpy as np
import base64
import uuid
import shutil
import subprocess
import threading
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import uvicorn
from highlight_algorithm import analyze_frame_highlight, merge_time_windows

model_lock = threading.Lock()

app = FastAPI(title="SmartPlay Football Highlight API", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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

def convert_to_h264(input_path: str) -> str:
    """
    Convert any uploaded video to H264 MP4 for maximum OpenCV compatibility.
    Returns path to converted video.
    """

    converted_path = os.path.splitext(input_path)[0] + "_h264.mp4"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,

        # Video codec
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",

        # Audio codec
        "-c:a", "aac",

        # Better compatibility
        "-movflags", "+faststart",

        converted_path
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise Exception(f"FFmpeg conversion failed:\n{result.stderr}")

    if not os.path.exists(converted_path):
        raise Exception("Converted video was not created")

    return converted_path

def draw_hud_box(img, x1, y1, x2, y2, label, conf, color, is_threat):
    # Draw bounding box outline (increased thickness from 1 to 3)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    
    # Draw corner brackets (HUD Targeting style - increased thickness from 2 to 3)
    bracket_len = min(20, int((x2 - x1) * 0.2), int((y2 - y1) * 0.2))
    if bracket_len > 2:
        cv2.line(img, (x1, y1), (x1 + bracket_len, y1), color, 3)
        cv2.line(img, (x1, y1), (x1, y1 + bracket_len), color, 3)
        cv2.line(img, (x2, y1), (x2 - bracket_len, y1), color, 3)
        cv2.line(img, (x2, y1), (x2, y1 + bracket_len), color, 3)
        cv2.line(img, (x1, y2), (x1 + bracket_len, y2), color, 3)
        cv2.line(img, (x1, y2), (x1, y2 - bracket_len), color, 3)
        cv2.line(img, (x2, y2), (x2 - bracket_len, y2), color, 3)
        cv2.line(img, (x2, y2), (x2, y2 - bracket_len), color, 3)

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

def render_bounded_video_segment(video_path: str, start: float, end: float, output_path: str):
    """
    Renders a specific segment of the video with bounding boxes on-the-fly.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Failed to open original video for segment rendering")
        
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame = int(start * fps)
    end_frame = int(end * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    temp_segment_raw = output_path + "_raw_seg.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_segment_raw, fourcc, fps, (width, height))
    
    current_frame = start_frame
    try:
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            with model_lock:
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
                        
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                mapping = CLASS_MAPPINGS[det["class_id"]]
                draw_hud_box(frame, x1, y1, x2, y2, mapping["name"], det["confidence"], mapping["color"], False)
                
            writer.write(frame)
            current_frame += 1
    finally:
        cap.release()
        writer.release()
        
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_segment_raw,
        "-ss", str(start),
        "-to", str(end),
        "-i", video_path,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        output_path
    ]
    
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if os.path.exists(temp_segment_raw):
        os.remove(temp_segment_raw)
        
    if res.returncode != 0:
        raise Exception(f"FFmpeg segment transcode/merge failed: {res.stderr}")

def render_bounded_video_background(job_id: str, video_path: str):
    """
    Renders a version of the video with bounding boxes in the background.
    Saves it to: f"{job_id}_bounded.mp4" in the UPLOAD_DIR.
    """
    try:
        jobs[job_id]["bounded_status"] = "rendering"
        jobs[job_id]["bounded_progress"] = 0.0
        print(f"Starting bounded video rendering for job {job_id}...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Failed to open video for bounded rendering")
            
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        temp_raw_path = os.path.join(UPLOAD_DIR, f"{job_id}_temp_bounded_raw.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_raw_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            with model_lock:
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
            
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                mapping = CLASS_MAPPINGS[det["class_id"]]
                draw_hud_box(frame, x1, y1, x2, y2, mapping["name"], det["confidence"], mapping["color"], False)
                
            writer.write(frame)
            frame_idx += 1
            
            if frame_count > 0:
                jobs[job_id]["bounded_progress"] = frame_idx / frame_count
                
        cap.release()
        writer.release()
        
        bounded_final_path = os.path.join(UPLOAD_DIR, f"{job_id}_bounded.mp4")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_raw_path,
            "-i", video_path,
            "-map", "0:v:0",
            "-map", "1:a?",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            bounded_final_path
        ]
        
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if os.path.exists(temp_raw_path):
            os.remove(temp_raw_path)
            
        if res.returncode == 0 and os.path.exists(bounded_final_path):
            jobs[job_id]["bounded_status"] = "completed"
            jobs[job_id]["bounded_progress"] = 1.0
            jobs[job_id]["bounded_video_path"] = bounded_final_path
            print(f"Bounded video rendering completed for job {job_id}")
        else:
            raise Exception(f"FFmpeg bounded transcode failed: {res.stderr}")
            
    except Exception as e:
        print(f"Bounded rendering failed for {job_id}: {e}")
        jobs[job_id]["bounded_status"] = "failed"
        jobs[job_id]["bounded_error"] = str(e)

def process_video_background(job_id: str, video_path: str):
    jobs[job_id] = {
        "status": "processing",
        "progress": 0.0,
        "video_path": video_path,
        "duration": 0.0,
        "fps": 0.0,
        "intervals": [],
        "error": None,
        "bounded_status": "pending",
        "bounded_progress": 0.0,
        "bounded_error": None
    }
    
    try:
        # Transcode the video first to ensure compatibility (H.264 / AAC MP4)
        transcoded_filename = f"{job_id}_transcoded.mp4"
        transcoded_path = os.path.join(UPLOAD_DIR, transcoded_filename)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-c:v", "libx264",
            "-preset", "superfast",
            "-crf", "23",
            "-c:a", "aac",
            "-strict", "experimental",
            transcoded_path
        ]
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0 and os.path.exists(transcoded_path):
                video_path = transcoded_path
                jobs[job_id]["video_path"] = transcoded_path
                print(f"Successfully transcoded {job_id} to standard MP4 H.264")
            else:
                print(f"Transcoding failed with code {result.returncode}. Stderr: {result.stderr}. Falling back to original.")
        except Exception as trans_ex:
            print(f"Transcoding process error: {trans_ex}. Falling back to original.")
            
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
        
        # Setup VideoWriter for bounded video rendering in parallel
        temp_raw_path = os.path.join(UPLOAD_DIR, f"{job_id}_temp_bounded_raw.mp4")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        writer = None
        bounded_enabled = False
        try:
            writer = cv2.VideoWriter(temp_raw_path, fourcc, fps, (width, height))
            if writer.isOpened():
                bounded_enabled = True
                jobs[job_id]["bounded_status"] = "rendering"
            else:
                jobs[job_id]["bounded_status"] = "failed"
                jobs[job_id]["bounded_error"] = "Failed to open VideoWriter"
        except Exception as writer_ex:
            jobs[job_id]["bounded_status"] = "failed"
            jobs[job_id]["bounded_error"] = str(writer_ex)

        highlight_frames = []
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run model on every frame to support smooth bounding box rendering
            with model_lock:
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
            
            # Draw bounding boxes and write to video stream
            if bounded_enabled:
                try:
                    annotated_frame = frame.copy()
                    for det in detections:
                        x1, y1, x2, y2 = det["bbox"]
                        mapping = CLASS_MAPPINGS[det["class_id"]]
                        draw_hud_box(annotated_frame, x1, y1, x2, y2, mapping["name"], det["confidence"], mapping["color"], False)
                    writer.write(annotated_frame)
                except Exception as write_err:
                    print(f"Failed to write bounded frame: {write_err}")
                    bounded_enabled = False
                    jobs[job_id]["bounded_status"] = "failed"
                    jobs[job_id]["bounded_error"] = str(write_err)

            # Analyze highlights at sample intervals
            if frame_idx % sample_interval == 0:
                current_time = frame_idx / fps
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
            progress_val = min(0.99, frame_idx / frame_count)
            jobs[job_id]["progress"] = progress_val
            if bounded_enabled:
                jobs[job_id]["bounded_progress"] = progress_val
            
        cap.release()
        if writer is not None:
            writer.release()
            
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
        
        # Finalize the bounded video using FFmpeg to combine video and audio streams
        if bounded_enabled:
            bounded_final_path = os.path.join(UPLOAD_DIR, f"{job_id}_bounded.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_raw_path,
                "-i", video_path,
                "-map", "0:v:0",
                "-map", "1:a?",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-shortest",
                bounded_final_path
            ]
            
            try:
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if os.path.exists(temp_raw_path):
                    os.remove(temp_raw_path)
                
                if res.returncode == 0 and os.path.exists(bounded_final_path):
                    jobs[job_id]["bounded_status"] = "completed"
                    jobs[job_id]["bounded_progress"] = 1.0
                    jobs[job_id]["bounded_video_path"] = bounded_final_path
                    print(f"Bounded video rendering completed for job {job_id}")
                else:
                    print(f"FFmpeg bounded transcode failed: {res.stderr}")
                    jobs[job_id]["bounded_status"] = "failed"
                    jobs[job_id]["bounded_error"] = f"FFmpeg failed: {res.stderr}"
            except Exception as ffmpeg_ex:
                print(f"FFmpeg bounded transcode exception: {ffmpeg_ex}")
                jobs[job_id]["bounded_status"] = "failed"
                jobs[job_id]["bounded_error"] = str(ffmpeg_ex)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

class IntervalExport(BaseModel):
    id: int
    start: float
    end: float
    type: str
    confidence: float
    correct: bool
    isGoal: bool
    description: str

class ExportPayload(BaseModel):
    job_id: str
    intervals: List[IntervalExport]
    export_markdown: bool = True
    export_clean: bool = True
    export_bounding: bool = True
    actual_goals: int = 0

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
        with model_lock:
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
            
        # background_tasks.add_task(process_video_background, job_id, video_path)
        
        # Convert uploaded video to H264 first
        converted_video_path = convert_to_h264(video_path)

        # Process converted video
        background_tasks.add_task(
            process_video_background,
            job_id,
            converted_video_path
)

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

@app.get("/api/video/{job_id}")
def get_video(job_id: str, bounded: bool = False):
    if bounded:
        bounded_path = os.path.join(UPLOAD_DIR, f"{job_id}_bounded.mp4")
        if os.path.exists(bounded_path):
            return FileResponse(bounded_path, media_type="video/mp4")
        else:
            raise HTTPException(status_code=404, detail="Bounded video not ready yet")

    transcoded_path = os.path.join(UPLOAD_DIR, f"{job_id}_transcoded.mp4")
    if os.path.exists(transcoded_path):
        return FileResponse(transcoded_path, media_type="video/mp4")
        
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
        if os.path.exists(path):
            return FileResponse(path, media_type="video/mp4")
            
    raise HTTPException(status_code=404, detail="Video file not found")

def generate_markdown_report(payload: ExportPayload, job_id: str) -> str:
    import datetime
    total_clips = len(payload.intervals)
    correct_clips = sum(1 for c in payload.intervals if c.correct)
    accuracy_rate = int((correct_clips / total_clips) * 100) if total_clips > 0 else 0
    ai_goals = sum(1 for c in payload.intervals if c.type == 'GOAL')
    
    md_content = f"""# BÁO CÁO PHÂN TÍCH HIGHLIGHT TRẬN ĐẤU
* **Mã Tiến Trình (Job ID):** {job_id or "N/A"}
* **Thời Gian Xuất Báo Cáo:** {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

## 📊 THỐNG KÊ HIỆU SUẤT MÔ HÌNH

* **Số bàn thắng thực tế (Người dùng xác nhận):** {payload.actual_goals}
* **Số bàn thắng dự đoán (AI nhận diện):** {ai_goals}
* **Số phân cảnh AI dự đoán đúng:** {correct_clips}/{total_clips} phân cảnh
* **Độ chính xác của AI:** {accuracy_rate}%

## 🎞 CHI TIẾT CÁC PHÂN CẢNH ĐÃ XUẤT HIGHLIGHT

| Phân cảnh | Trim (Bắt đầu - Kết thúc) | Nhãn AI | Độ tin cậy (AI Conf) | Đánh giá của Bạn | Xác nhận Bàn thắng | Nội dung chi tiết |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
"""
    for clip in payload.intervals:
        eval_str = "✅ Đúng" if clip.correct else "❌ Sai"
        goal_str = "⚽ Có" if clip.isGoal else "❌ Không"
        md_content += f"| #{clip.id + 1} | {clip.start}s - {clip.end}s | {clip.type} | {int(clip.confidence * 100)}% | {eval_str} | {goal_str} | {clip.description} |\n"

    md_content += "\n\n---\n*Báo cáo được tạo tự động bởi hệ thống SmartPlay Football Highlight Analyzer.*"
    return md_content

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
            
        if not (payload.export_clean or payload.export_bounding or payload.export_markdown):
            raise HTTPException(status_code=400, detail="Vui lòng chọn ít nhất một định dạng để xuất.")
            
        export_id = str(uuid.uuid4())
        export_subfolder = os.path.join(OUTPUT_DIR, export_id)
        os.makedirs(export_subfolder, exist_ok=True)
        
        files_to_export = []
        
        # 1. Generate Markdown report
        if payload.export_markdown:
            report_text = generate_markdown_report(payload, job_id)
            report_path = os.path.join(export_subfolder, "report.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            files_to_export.append((report_path, "report.md"))
            
        # 2. Process clips
        for idx, interval in enumerate(intervals):
            start = interval.start
            end = interval.end
            
            # Clean clip
            if payload.export_clean:
                clean_name = f"highlight_{interval.id + 1}_clean.mp4"
                clean_path = os.path.join(export_subfolder, clean_name)
                
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start),
                    "-to", str(end),
                    "-i", video_path,
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    "-avoid_negative_ts", "make_zero",
                    "-fflags", "+genpts",
                    clean_path
                ]
                
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode != 0:
                    raise Exception(f"Failed to cut clean segment {start} - {end}: {result.stderr}")
                    
                files_to_export.append((clean_path, clean_name))
                
            # Bounded clip
            if payload.export_bounding:
                bounded_name = f"highlight_{interval.id + 1}_bounding.mp4"
                bounded_path = os.path.join(export_subfolder, bounded_name)
                
                # Check if the entire video has been fully rendered with bounds in background
                if jobs[job_id].get("bounded_status") == "completed" and "bounded_video_path" in jobs[job_id]:
                    bounded_source_path = jobs[job_id]["bounded_video_path"]
                    cmd = [
                        "ffmpeg", "-y",
                        "-ss", str(start),
                        "-to", str(end),
                        "-i", bounded_source_path,
                        "-c:v", "libx264",
                        "-c:a", "aac",
                        "-avoid_negative_ts", "make_zero",
                        "-fflags", "+genpts",
                        bounded_path
                    ]
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if result.returncode != 0:
                        raise Exception(f"Failed to cut bounded segment from pre-rendered video: {result.stderr}")
                else:
                    # Telemetry video rendering is not finished yet, render segment on-the-fly
                    render_bounded_video_segment(video_path, start, end, bounded_path)
                    
                files_to_export.append((bounded_path, bounded_name))
                
        if len(files_to_export) == 0:
            raise HTTPException(status_code=400, detail="Không có tệp nào được tạo.")
            
        if len(files_to_export) == 1:
            file_path, archive_name = files_to_export[0]
            media_type = "video/mp4"
            if archive_name.endswith(".md"):
                media_type = "text/markdown"
            return FileResponse(
                path=file_path,
                media_type=media_type,
                filename=archive_name
            )
        else:
            zip_filename = f"highlights_{export_id}.zip"
            zip_path = os.path.join(export_subfolder, zip_filename)
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path, archive_name in files_to_export:
                    zipf.write(file_path, archive_name)
                    
            if not os.path.exists(zip_path):
                raise Exception("Exported ZIP file was not created successfully")
                
            return FileResponse(
                path=zip_path,
                media_type="application/zip",
                filename="smartplay_highlights.zip"
            )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
