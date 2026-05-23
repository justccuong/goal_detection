import math
import numpy as np

def iou_area(ball_box, goal_box):
    """
    Calculate intersection area over ball area.
    ball_box, goal_box format: [x1, y1, x2, y2]
    """
    bx1, by1, bx2, by2 = ball_box
    gx1, gy1, gx2, gy2 = goal_box

    # Intersection rectangle
    ix1 = max(bx1, gx1)
    iy1 = max(by1, gy1)
    ix2 = min(bx2, gx2)
    iy2 = min(by2, gy2)

    if ix1 < ix2 and iy1 < iy2:
        inter_area = (ix2 - ix1) * (iy2 - iy1)
    else:
        inter_area = 0

    ball_area = (bx2 - bx1) * (by2 - by1)
    return inter_area / ball_area if ball_area > 0 else 0

def center_distance(box1, box2):
    """Euclidean distance between centers of two boxes."""
    x1 = (box1[0] + box1[2]) / 2
    y1 = (box1[1] + box1[3]) / 2
    x2 = (box2[0] + box2[2]) / 2
    y2 = (box2[1] + box2[3]) / 2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def closest_ball_goalpost(detections):
    """
    detections: list of dicts with keys {'class_id': int, 'bbox': [x1,y1,x2,y2]}
    Returns: (ball_box, goal_box, intersection_ratio, min_dist)
    """
    balls = [d['bbox'] for d in detections if d['class_id'] == 0]
    goals = [d['bbox'] for d in detections if d['class_id'] == 1]
    
    if not balls or not goals:
        return None

    min_dist = float('inf')
    best_pair = None

    for ball in balls:
        for goal in goals:
            dist = center_distance(ball, goal)
            if dist < min_dist:
                min_dist = dist
                best_pair = (ball, goal)

    ball_box, goal_box = best_pair
    ratio = iou_area(ball_box, goal_box)
    return ball_box, goal_box, ratio, min_dist

def analyze_frame_highlight(detections):
    """
    Adapted from Kaggle logic:
    Checks if a frame is a highlight based on IOU and distance.
    Returns: (threat_level, instruction)
    """
    balls = [d for d in detections if d["class_id"] == 0]
    goalposts = [d for d in detections if d["class_id"] == 1]
    
    if not balls:
        return "NO_BALL", "No soccer ball detected in frame. Normal background play."
    if not goalposts:
        return "ACTIVE_PLAY", "Ball is active, but goalposts are out of frame. Midfield buildup."
        
    r = closest_ball_goalpost(detections)
    if not r:
        return "ACTIVE_PLAY", "Ball and goalpost detected, but play is at a safe distance. Standard possession play."
        
    ball_box, goal_box, ratio, min_dist = r
    
    # Kaggle script considers it a goal if ratio > 0.
    if ratio > 0:
        return "GOAL", "CRITICAL HIGHLIGHT: Ball is inside the goalpost area! Potential goal scored or critical goal-line clearance."
        
    # We also keep NEAR_MISS if distance is relatively small for "highlight" tracking
    gx1, gy1, gx2, gy2 = goal_box
    goal_width = gx2 - gx1
    goal_height = gy2 - gy1
    max_dim = max(goal_width, goal_height)
    
    if min_dist < max_dim * 0.8:
        return "NEAR_MISS", "HIGH INTEREST: Ball is close to the goalpost. Review for shot on target, header, or save."
        
    return "ACTIVE_PLAY", "Ball and goalpost detected, but play is at a safe distance. Standard possession play."

def merge_time_windows(highlight_frames, window_duration=8.0, total_duration=0.0):
    """
    Merges overlapping time windows based on Kaggle's logic adapted to seconds.
    """
    if not highlight_frames:
        return []
        
    segments = []
    
    # Sort by time just in case
    highlight_frames.sort(key=lambda x: x["time"])
    
    first_hf = highlight_frames[0]
    start = max(0.0, first_hf["time"] - window_duration)
    end = min(total_duration, first_hf["time"] + window_duration)
    
    current_segment = {
        "start": start,
        "end": end,
        "type": first_hf["type"],
        "peak_conf": first_hf["conf"],
        "peak_frame": first_hf["frame"],
        "peak_detections": first_hf["detections"]
    }

    for hf in highlight_frames[1:]:
        t = hf["time"]
        s = max(0.0, t - window_duration)
        e = min(total_duration, t + window_duration)
        
        # Overlapping
        if s <= current_segment["end"]:
            current_segment["end"] = max(current_segment["end"], e)
            
            # Update segment properties if this frame is "better"
            if hf["type"] == "GOAL" and current_segment["type"] != "GOAL":
                current_segment["type"] = "GOAL"
            if hf["conf"] > current_segment["peak_conf"]:
                current_segment["peak_conf"] = hf["conf"]
                current_segment["peak_frame"] = hf["frame"]
                current_segment["peak_detections"] = hf["detections"]
        else:
            # Gap -> save current segment and start new
            segments.append(current_segment)
            current_segment = {
                "start": s,
                "end": e,
                "type": hf["type"],
                "peak_conf": hf["conf"],
                "peak_frame": hf["frame"],
                "peak_detections": hf["detections"]
            }
            
    segments.append(current_segment)
    return segments
