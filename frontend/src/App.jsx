import React, { useState, useEffect, useRef } from 'react';
import './App.css';

// Import Raw SVGs as strings using Vite's ?raw modifier
import flow1Raw from './assets/flow_1_startup_upload.svg?raw';
import flow2Raw from './assets/flow_2_worker_inference.svg?raw';
import flow3Raw from './assets/flow_3_dashboard_export.svg?raw';

// Node explanations lookup mapping
const SVG_EXPLANATIONS = {
  // Flow 1
  "What does the database store for model metadata?": "The SQL Database stores model metadata, including the file path (e.g., in object storage), version history, creation timestamps, and SHA-256 checksums to ensure file integrity when workers load weights.",
  "Why store the model in object storage instead of database?": "YOLO model weights (best.pt) are binary files. Storing heavy weights directly in databases causes transaction sluggishness, memory overhead, and costly backups. Object storage (S3/R2) is designed for high-throughput, low-cost static files.",
  "What does the worker do at startup?": "At boot time, the worker process requests the active model's metadata, downloads the file from object storage (if not already cached locally), loads it into RAM/GPU memory, and starts listening to the Redis queue.",
  "What does the user send in the upload request?": "The client sends a HTTP POST request with a multipart form containing the raw video/image file, along with optional parameters like target processing FPS and region-of-interest dimensions.",
  "What does the API do on upload?": "FastAPI receives the file, validates its format, generates a unique job_id (UUID v4), writes the file to temporary local or object storage, and pushes the job payload onto the Redis queue before returning a 202 Accepted status immediately.",
  "Why keep the video in object storage until export?": "Keeping the raw video in object storage ensures it is safe from disk failures and accessible by any available worker in the cluster, rather than locking it to a single worker's local storage.",
  "What payload goes into the Redis queue?": "The Redis queue payload contains a JSON string with the job metadata: `{ \"job_id\": \"uuid-123\", \"video_url\": \"s3://video.mp4\", \"fps\": 30 }`.",
  "How often should the client poll for results?": "Clients should poll using exponential backoff (e.g. starting at 1s, doubling to 2s, 4s, up to 10s) to check the endpoint `GET /api/result/{job_id}`, reducing unnecessary server requests.",
  "What does the result store contain?": "The Redis result store maintains the status of the job (`pending`, `processing`, or `completed`), a progress counter (`current_frame / total_frames`), and a list of identified target intervals.",

  // Flow 2
  "What is in the queued job payload?": "It contains the job ID, the direct URL to download/stream the video, and processing configuration parameters (like YOLO confidence threshold and sampling rates).",
  "Should the worker download the video or stream it?": "Downloading is safer for short clips, but streaming using OpenCV's `VideoCapture(stream_url)` is preferred for large videos. Streaming avoids high local disk usage and starts processing immediately without waiting for a full download.",
  "What happens when cap.read() returns False?": "It signifies either the end of the video or a stream disconnection. The frame loop terminates, and the worker cleanup routines close the video decoder and update the job status to 'completed'.",
  "What happens when the loop finishes?": "The worker marks the job as finished in Redis, saves the final list of detected clips, and optionally triggers a webhook or notification, then waits for the next job from the queue.",
  "Why letterbox instead of plain resize for YOLO input?": "Plain resizing stretches or squishes the image, distorting the aspect ratio and reducing model accuracy. Letterboxing adds black borders (padding) to keep the original aspect ratio constant for the 640x640 network input.",
  "What does YOLO output — what format are the bboxes?": "YOLO outputs bounding boxes as `[x_center, y_center, width, height]` (normalized xywhn format) or absolute corner coordinates `[x1, y1, x2, y2]`, along with class IDs and confidence scores (0.0 to 1.0).",
  "How exactly does the bbox intersection check work?": "It computes the Intersection over Union (IoU) of the target bounding boxes (e.g. ball and goalpost) or checks if a target bbox center is inside a region, filtering out coincidental overlaps and confirming target containment.",
  "How to handle interval going below 0 or past video end?": "The clipping interval is mathematically bound to `[0, video_duration]`. The start timestamp is clamped using `max(0, ts - 5)` and the end is clamped using `min(duration, ts + 5)`.",
  "What exactly is saved per interval in Redis?": "The database saves a JSON object representing the start timestamp, end timestamp, the highest detection confidence score, class details, and a URL link to the preview thumbnail image.",

  // Flow 3
  "What interval data is fetched from Redis for the dashboard?": "The dashboard queries a list of identified match clip segments, each including the start/end times, preview image link, and class statistics (e.g. 'Ball detected near goal').",
  "How should the thumbnail be generated for each interval?": "The worker captures the frame with the highest confidence detection inside the interval, compresses it to JPEG, overlays bounding boxes, uploads it to storage, and returns the public link.",
  "What does keep do internally?": "Clicking 'Keep' marks the event interval as verified, adding its timestamps to the final list of clips slated for inclusion in the edited highlight summary reel.",
  "What trim controls should the modify UI show?": "The UI displays a range slider (dual handles) corresponding to the start and end of the clip, allowing the operator to fine-tune the trim boundaries visually frame by frame.",
  "Does discard delete anything from storage?": "No, it simply removes the interval from the highlight render list. The original video remains in object storage until the job lifecycle retention policy deletes it (typically 24 hours).",
  "What payload does POST /export send?": "It sends a JSON array containing the selected clip intervals: `[{ \"start\": 12.5, \"end\": 22.0 }, { \"start\": 85.1, \"end\": 92.4 }]` to initiate compilation.",
  "What is the exact FFmpeg command to cut a clip by timestamps?": "To cut without re-encoding: `ffmpeg -ss [start] -to [end] -i input.mp4 -c copy output_clip.mp4`. This is extremely fast as it copy-pastes the video frames directly without computing new pixels.",
  "When should we concat vs zip the clips?": "Concatenation combines all segments into a single continuous highlight video (useful for quick playback). Zipping compresses separate clip files into one file (ideal if individual segments must be cataloged separately).",
  "How should the download link be served — presigned URL?": "The API generates a secure, time-limited presigned URL from the Object Storage service (e.g., S3 Presigned URL expiring in 15 minutes) so users download it directly from the storage network, offloading API traffic."
};

// Preset images (sports-focused photos that can be checked by YOLO model)
const PRESETS = [
  {
    name: "Penalty Shot (Ball & Goal)",
    url: "https://images.unsplash.com/photo-1508098682722-e99c43a406b2?w=800",
    desc: "A penalty attempt featuring the soccer ball close to the goalposts."
  },
  {
    name: "Goalpost Net Close-up",
    url: "https://images.unsplash.com/photo-1518005020951-eccb494ad742?w=800",
    desc: "Empty soccer goal and pitch markers."
  },
  {
    name: "Match Action Build-up",
    url: "https://images.unsplash.com/photo-1529900748604-07564a03e7a6?w=800",
    desc: "Soccer match in progress with ball active on the pitch."
  }
];

function App() {
  const [activeTab, setActiveTab] = useState('scanner'); // 'scanner', 'schematics', 'logs'
  const [activeSvgTab, setActiveSvgTab] = useState('flow1'); // 'flow1', 'flow2', 'flow3'
  const videoRef = useRef(null);
  
  // Scanning State
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scanResult, setScanResult] = useState(null);
  const [audioEnabled, setAudioEnabled] = useState(true);
  const [errorMsg, setErrorMsg] = useState(null);
  
  // Video Scanning & Exporting State
  const [uploadType, setUploadType] = useState(null); // 'image' or 'video'
  const [videoUrl, setVideoUrl] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [isScanningVideo, setIsScanningVideo] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);
  const [videoIntervals, setVideoIntervals] = useState([]);
  const [isExporting, setIsExporting] = useState(false);
  const [actualGoals, setActualGoals] = useState(0);

  // Bounded telemetry video & export options states
  const [playBounded, setPlayBounded] = useState(false);
  const [boundedStatus, setBoundedStatus] = useState('pending');
  const [boundedProgress, setBoundedProgress] = useState(0);
  const [exportOptions, setExportOptions] = useState({
    markdown: true,
    clean: true,
    bounding: true
  });

  // Custom Video Player & Playback Range Bounding States
  const [isPlaying, setIsPlaying] = useState(false);
  const [videoCurrentTime, setVideoCurrentTime] = useState(0);
  const [videoDuration, setVideoDuration] = useState(0);
  const [activePlayInterval, setActivePlayInterval] = useState(null); // { start, end }

  const formatTime = (secs) => {
    if (isNaN(secs)) return "00:00";
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m < 10 ? '0' + m : m}:${s < 10 ? '0' + s : s}`;
  };

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
    } else {
      videoRef.current.play().catch(e => console.warn("Playback blocked", e));
    }
  };

  const stopVideo = () => {
    if (!videoRef.current) return;
    videoRef.current.pause();
    videoRef.current.currentTime = 0;
    setActivePlayInterval(null);
  };

  const handleScrubberChange = (e) => {
    const newTime = parseFloat(e.target.value);
    if (videoRef.current) {
      videoRef.current.currentTime = newTime;
      setVideoCurrentTime(newTime);
      setActivePlayInterval(null); // Stop segment preview if user seeks
    }
  };

  const handlePlayShort = (start, end) => {
    if (videoRef.current) {
      videoRef.current.currentTime = start;
      setActivePlayInterval({ start, end });
      videoRef.current.play().catch(e => console.warn("Playback blocked", e));
      addLog(`Playing segment review: ${start}s to ${end}s`, "info");
    }
  };

  const handleTimeUpdate = () => {
    if (!videoRef.current) return;
    const currentTime = videoRef.current.currentTime;
    setVideoCurrentTime(currentTime);

    if (activePlayInterval) {
      if (currentTime >= activePlayInterval.end) {
        videoRef.current.pause();
        setActivePlayInterval(null);
        addLog(`Reached end of segment at ${activePlayInterval.end}s. Stopped playback.`, "info");
      }
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setVideoDuration(videoRef.current.duration);
    }
  };

  // SVG click explanation state
  const [selectedNode, setSelectedNode] = useState(null);
  const [explanationText, setExplanationText] = useState("Click on any component node inside the pipeline flowchart to inspect its backend mechanics and architectural details.");
  
  // Terminal log state
  const [logs, setLogs] = useState([
    { time: "22:18:01", type: "system", text: "Initializing SmartPlay Highlight Compiler v1.0.0..." },
    { time: "22:18:02", type: "system", text: "Loading custom best.pt Neural Core into RAM..." },
    { time: "22:18:03", type: "success", text: "YOLOv8 Custom Goal/Ball tracker model loaded." },
    { time: "22:18:03", type: "system", text: "Listening on port 8000 for match uploads." }
  ]);
  
  const logContainerRef = useRef(null);

  // Auto-scroll logs
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  // Audio alert synthesizer using Web Audio API
  const playAlert = (status) => {
    if (!audioEnabled) return;
    try {
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      
      if (status === 'GOAL') {
        // High pitch cheer/chime siren
        const osc1 = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();
        
        osc1.type = 'sine';
        osc1.frequency.setValueAtTime(600, audioCtx.currentTime);
        osc1.frequency.linearRampToValueAtTime(1200, audioCtx.currentTime + 0.3);
        osc1.frequency.linearRampToValueAtTime(800, audioCtx.currentTime + 0.6);
        osc1.frequency.linearRampToValueAtTime(1500, audioCtx.currentTime + 0.9);
        
        gainNode.gain.setValueAtTime(0.15, audioCtx.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 1.2);
        
        osc1.connect(gainNode);
        gainNode.connect(audioCtx.destination);
        osc1.start();
        osc1.stop(audioCtx.currentTime + 1.2);
      } else if (status === 'NEAR_MISS') {
        // Warning chime
        [0, 0.2].forEach((delay) => {
          const osc = audioCtx.createOscillator();
          const gain = audioCtx.createGain();
          osc.type = 'triangle';
          osc.frequency.setValueAtTime(700, audioCtx.currentTime + delay);
          gain.gain.setValueAtTime(0.12, audioCtx.currentTime + delay);
          gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + delay + 0.15);
          
          osc.connect(gain);
          gain.connect(audioCtx.destination);
          osc.start(audioCtx.currentTime + delay);
          osc.stop(audioCtx.currentTime + delay + 0.15);
        });
      } else if (status === 'ACTIVE_PLAY') {
        // Simple click chime
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(440, audioCtx.currentTime);
        gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.1);
        
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.1);
      }
    } catch (e) {
      console.warn("Audio context blocked or unsupported", e);
    }
  };

  const addLog = (text, type = "info") => {
    const timeString = new Date().toTimeString().split(' ')[0];
    setLogs(prev => [...prev, { time: timeString, type, text }]);
  };

  // Drag and drop handlers
  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setErrorMsg(null);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        setUploadType('image');
        setSelectedFile(file);
        setPreviewUrl(URL.createObjectURL(file));
        setScanResult(null);
        setVideoIntervals([]);
        addLog(`Image loaded: ${file.name} (${Math.round(file.size / 1024)} KB)`);
      } else if (file.type.startsWith('video/') || file.name.endsWith('.mp4')) {
        setUploadType('video');
        setSelectedFile(file);
        setPreviewUrl(null);
        setVideoUrl(URL.createObjectURL(file));
        setScanResult(null);
        setVideoIntervals([]);
        setActualGoals(0);
        addLog(`Video loaded: ${file.name}. Click 'ANALYZE MATCH EVENT' to begin scanning.`);
      } else {
        setErrorMsg("Error: Please drop a valid image or MP4 video file.");
      }
    }
  };

  const handleFileChange = (e) => {
    setErrorMsg(null);
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (file.type.startsWith('image/')) {
        setUploadType('image');
        setSelectedFile(file);
        setPreviewUrl(URL.createObjectURL(file));
        setScanResult(null);
        setVideoIntervals([]);
        addLog(`Image selected: ${file.name} (${Math.round(file.size / 1024)} KB)`);
      } else if (file.type.startsWith('video/') || file.name.endsWith('.mp4')) {
        setUploadType('video');
        setSelectedFile(file);
        setPreviewUrl(null);
        setVideoUrl(URL.createObjectURL(file));
        setScanResult(null);
        setVideoIntervals([]);
        setActualGoals(0);
        addLog(`Video selected: ${file.name}. Click 'ANALYZE MATCH EVENT' to begin scanning.`);
      } else {
        setErrorMsg("Error: Please select a valid image or MP4 video file.");
      }
    }
  };

  // Fetch preset image and parse as blob for upload
  const loadPreset = async (preset) => {
    setErrorMsg(null);
    setIsScanning(true);
    setUploadType('image');
    setVideoIntervals([]);
    addLog(`Fetching preset soccer asset: ${preset.name}...`);
    try {
      const response = await fetch(preset.url);
      const blob = await response.blob();
      const file = new File([blob], "preset.jpg", { type: "image/jpeg" });
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setScanResult(null);
      
      await uploadAndScan(file);
    } catch (err) {
      addLog(`Failed to load preset: ${err.message}`, "error");
      setErrorMsg("Failed to download preset image. Try uploading a local file.");
      setIsScanning(false);
    }
  };

  const triggerScan = async () => {
    if (!selectedFile) return;
    if (uploadType === 'video') {
      await uploadAndScanVideo(selectedFile);
    } else {
      setIsScanning(true);
      await uploadAndScan(selectedFile);
    }
  };

  const uploadAndScan = async (file) => {
    addLog("Sending match feed frame to FastAPI backend...", "system");
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/api/detect", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error(`API responded with ${response.status}`);
      }

      const data = await response.json();
      setScanResult(data);
      setIsScanning(false);
      
      addLog(`YOLOv8 best.pt scan finished. Detections: ${data.threat_count} balls, ${data.ally_count} goalposts.`, "success");
      addLog(`Highlight Level: ${data.threat_level}. Suggestion: ${data.instruction}`, data.threat_level === "GOAL" ? "error" : data.threat_level === "NEAR_MISS" ? "warning" : "success");
      
      playAlert(data.threat_level);
    } catch (err) {
      addLog(`Scan failed: ${err.message}. Backend might be offline.`, "error");
      setErrorMsg("Could not connect to FastAPI server. Please ensure the backend is running on port 8000.");
      setIsScanning(false);
    }
  };

  // Video Scanning & Polling Logic
  const uploadAndScanVideo = async (file) => {
    setIsScanningVideo(true);
    setScanProgress(0);
    setVideoIntervals([]);
    setActualGoals(0);
    addLog(`Uploading video file: ${file.name} for highlight analysis...`, "system");
    
    const formData = new FormData();
    formData.append("file", file);
    
    try {
      const response = await fetch("http://localhost:8000/api/upload_video", {
        method: "POST",
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`API returned status ${response.status}`);
      }
      
      const data = await response.json();
      const job_id = data.job_id;
      setJobId(job_id);
      addLog(`Video upload complete. Job registered: ${job_id}. Starting frame scan...`, "success");
      
      // Start polling status
      pollVideoStatus(job_id);
    } catch (err) {
      addLog(`Video upload failed: ${err.message}`, "error");
      setErrorMsg("Failed to upload video to backend. Make sure the backend is active.");
      setIsScanningVideo(false);
    }
  };

  const pollVideoStatus = (job_id) => {
    let scanFinished = false;
    let lastBoundedProgress = -1;
    const timer = setInterval(async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/status_video/${job_id}`);
        if (!response.ok) {
          throw new Error("Failed to poll status");
        }
        
        const data = await response.json();
        
        if (data.bounded_status) {
          setBoundedStatus(data.bounded_status);
          setBoundedProgress(data.bounded_progress || 0);
        }
        
        if (data.status === 'processing') {
          const progressPercent = Math.round(data.progress * 100);
          setScanProgress(data.progress);
          addLog(`Scanning video: ${progressPercent}% complete...`, "info");
        } else if (data.status === 'completed') {
          if (!scanFinished) {
            scanFinished = true;
            setScanProgress(1.0);
            setIsScanningVideo(false);
            
            const mappedIntervals = data.intervals.map(inter => ({ 
              ...inter, 
              keep: true,
              correct: true,
              isGoal: inter.type === 'GOAL'
            }));
            setVideoIntervals(mappedIntervals);
            const predictedGoalsCount = data.intervals.filter(c => c.type === 'GOAL').length;
            setActualGoals(predictedGoalsCount);
            setVideoUrl(`http://localhost:8000/api/video/${job_id}`);
            addLog(`Video scanning finished. Extracted ${mappedIntervals.length} soccer highlight clips!`, "success");
            if (mappedIntervals.length > 0) {
              playAlert("GOAL");
            }
            addLog("Kích hoạt kết xuất video theo dõi YOLOv8 (Bounding Box) trong nền...", "system");
          }
          
          if (data.bounded_status === 'rendering') {
            const bp = Math.round((data.bounded_progress || 0) * 100);
            if (bp !== lastBoundedProgress) {
              lastBoundedProgress = bp;
              addLog(`Đang kết xuất video Bounding Box trong nền: ${bp}%...`, "system");
            }
          }
          
          if (data.bounded_status === 'completed' || data.bounded_status === 'failed') {
            clearInterval(timer);
            if (data.bounded_status === 'completed') {
              addLog(`Kết xuất video Bounding Box hoàn tất! Luồng video YOLO hiện đã sẵn sàng.`, "success");
            } else {
              addLog(`Kết xuất video Bounding Box thất bại: ${data.bounded_error}`, "error");
            }
          }
        } else if (data.status === 'failed') {
          clearInterval(timer);
          setIsScanningVideo(false);
          setErrorMsg(`Video scan failed: ${data.error}`);
          addLog(`Video analysis job failed: ${data.error}`, "error");
        }
      } catch (err) {
        clearInterval(timer);
        setIsScanningVideo(false);
        setErrorMsg(`Polling connection error: ${err.message}`);
      }
    }, 1500);
  };

  const handleToggleBounded = () => {
    if (!jobId || boundedStatus !== 'completed') return;
    const nextPlayBounded = !playBounded;
    setPlayBounded(nextPlayBounded);
    
    if (videoRef.current) {
      const curTime = videoRef.current.currentTime;
      const wasPlaying = !videoRef.current.paused;
      
      const newUrl = nextPlayBounded 
        ? `http://localhost:8000/api/video/${jobId}?bounded=true`
        : `http://localhost:8000/api/video/${jobId}`;
      
      videoRef.current.src = newUrl;
      videoRef.current.load();
      videoRef.current.currentTime = curTime;
      
      if (wasPlaying) {
        videoRef.current.oncanplay = () => {
          videoRef.current.play().catch(e => console.warn("Playback blocked", e));
          videoRef.current.oncanplay = null;
        };
      }
      
      addLog(`Swapped stream to: ${nextPlayBounded ? "YOLO Bounded" : "Clean original"}`, "info");
    }
  };

  const handleSeek = (time) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      videoRef.current.play().catch(e => console.warn("Autoplay prevented", e));
    }
  };

  const downloadReportFile = (selectedClips) => {
    try {
      const fileName = selectedFile?.name || "match_video.mp4";
      const totalClips = videoIntervals.length;
      const correctClips = videoIntervals.filter(c => c.correct).length;
      const accuracyRate = totalClips > 0 ? Math.round((correctClips / totalClips) * 100) : 0;
      const aiGoals = videoIntervals.filter(c => c.type === 'GOAL').length;

      let mdContent = `# BÁO CÁO PHÂN TÍCH HIGHLIGHT TRẬN ĐẤU\n\n`;
      mdContent += `* **Tên Video Gốc:** ${fileName}\n`;
      mdContent += `* **Mã Tiến Trình (Job ID):** ${jobId || "N/A"}\n`;
      mdContent += `* **Thời Gian Xuất Báo Cáo:** ${new Date().toLocaleString('vi-VN')}\n`;
      mdContent += `\n## 📊 THỐNG KÊ HIỆU SUẤT MÔ HÌNH\n\n`;
      mdContent += `* **Số bàn thắng thực tế (Người dùng xác nhận):** ${actualGoals}\n`;
      mdContent += `* **Số bàn thắng dự đoán (AI nhận diện):** ${aiGoals}\n`;
      mdContent += `* **Số phân cảnh AI dự đoán đúng:** ${correctClips}/${totalClips} phân cảnh\n`;
      mdContent += `* **Độ chính xác của AI:** ${accuracyRate}%\n`;
      mdContent += `\n## 🎞 CHI TIẾT CÁC PHÂN CẢNH ĐÃ XUẤT HIGHLIGHT\n\n`;
      mdContent += `| Phân cảnh | Trim (Bắt đầu - Kết thúc) | Nhãn AI | Độ tin cậy (AI Conf) | Đánh giá của Bạn | Xác nhận Bàn thắng | Nội dung chi tiết |\n`;
      mdContent += `| :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n`;

      selectedClips.forEach((clip) => {
        mdContent += `| #${clip.id + 1} | ${clip.start}s - ${clip.end}s | ${clip.type} | ${Math.round(clip.confidence * 100)}% | ${clip.correct ? "✅ Đúng" : "❌ Sai"} | ${clip.isGoal ? "⚽ Có" : "❌ Không"} | ${clip.description} |\n`;
      });

      mdContent += `\n\n---\n*Báo cáo được tạo tự động bởi hệ thống SmartPlay Football Highlight Analyzer.*`;

      // Trigger markdown file download
      const blob = new Blob([mdContent], { type: 'text/markdown;charset=utf-8;' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `smartplay_highlight_report_${jobId || "export"}.md`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      addLog("Analysis report (smartplay_highlight_report.md) downloaded.", "success");
    } catch (e) {
      console.error("Failed to generate report", e);
      addLog("Failed to generate markdown report: " + e.message, "error");
    }
  };

  // Export video highlight reel
  const exportHighlightReel = async () => {
    const selectedClips = videoIntervals.filter(c => c.keep);
    if (selectedClips.length === 0) {
      setErrorMsg("No highlight clips selected for export.");
      return;
    }
    if (!exportOptions.markdown && !exportOptions.clean && !exportOptions.bounding) {
      setErrorMsg("Vui lòng chọn ít nhất một định dạng để xuất.");
      return;
    }
    
    setIsExporting(true);
    addLog(`Initiating highlight compilation for ${selectedClips.length} segments...`, "system");
    
    try {
      const response = await fetch("http://localhost:8000/api/export_video", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_id: jobId,
          intervals: selectedClips.map(c => ({
            id: parseInt(c.id),
            start: parseFloat(c.start),
            end: parseFloat(c.end),
            type: c.type || "ACTIVE_PLAY",
            confidence: parseFloat(c.confidence || 0),
            correct: c.correct,
            isGoal: c.isGoal,
            description: c.description || ""
          })),
          export_markdown: exportOptions.markdown,
          export_clean: exportOptions.clean,
          export_bounding: exportOptions.bounding,
          actual_goals: parseInt(actualGoals)
        })
      });
      
      if (!response.ok) {
        throw new Error("FFmpeg compilation request failed.");
      }
      
      const blob = await response.blob();
      const isZip = blob.type === 'application/zip';
      const isMd = blob.type === 'text/markdown' || blob.type.startsWith('text/');
      const url = window.URL.createObjectURL(blob);
      
      const a = document.createElement("a");
      a.href = url;
      if (isZip) {
        a.download = "football_highlights.zip";
      } else if (isMd) {
        a.download = `smartplay_highlight_report_${jobId || "export"}.md`;
      } else {
        a.download = exportOptions.bounding ? "football_highlights_bounding.mp4" : "football_highlights_clean.mp4";
      }
      document.body.appendChild(a);
      a.click();
      a.remove();
      
      setIsExporting(false);
      addLog("Xuất dữ liệu highlights thành công!", "success");
    } catch (err) {
      addLog(`Export failed: ${err.message}`, "error");
      setErrorMsg(`FFmpeg compilation error: ${err.message}`);
      setIsExporting(false);
    }
  };

  // Intercept SVG flowchart clicks
  const handleSvgClick = (e) => {
    let target = e.target;
    while (target && target !== e.currentTarget) {
      const onClickAttr = target.getAttribute('onclick');
      if (onClickAttr && onClickAttr.startsWith('sendPrompt(')) {
        e.preventDefault();
        e.stopPropagation();
        
        const match = onClickAttr.match(/sendPrompt\('(.*)'\)/);
        if (match && match[1]) {
          const prompt = match[1];
          setSelectedNode(prompt);
          
          if (SVG_EXPLANATIONS[prompt]) {
            setExplanationText(SVG_EXPLANATIONS[prompt]);
            addLog(`Querying Node: "${prompt}"`, "system");
          } else {
            setExplanationText("No explanation loaded for this component query.");
          }
        }
        break;
      }
      target = target.parentElement;
    }
  };

  const getSvgContent = () => {
    switch (activeSvgTab) {
      case 'flow1': return flow1Raw;
      case 'flow2': return flow2Raw;
      case 'flow3': return flow3Raw;
      default: return flow1Raw;
    }
  };

  return (
    <div className="app-container">
      <div className="cyber-grid"></div>
      
      {/* HEADER SECTION */}
      <header className="dashboard-header glass-panel">
        <div className="header-brand">
          <div className="brand-icon-glow sports-glow">
            <span className="blink-dot green"></span>
          </div>
          <div>
            <h1>SMARTPLAY HIGHLIGHT ANALYZER</h1>
            <p className="system-subtitle">YOLOv8 CUSTOM MATCH EVENT TRACKER & CLIP COORDINATOR</p>
          </div>
        </div>
        
        <div className="header-status">
          <div className="status-item">
            <span className="status-label">SYSTEM STATE:</span>
            <span className="status-val text-green">ACTIVE</span>
          </div>
          <div className="status-item">
            <span className="status-label">MODEL WEIGHTS:</span>
            <span className="status-val text-purple">best.pt</span>
          </div>
          <div className="status-item">
            <span className="status-label">HIGHLIGHT STATUS:</span>
            <span className={`status-val ${scanResult?.threat_level === 'GOAL' || videoIntervals.some(c => c.keep && c.type === 'GOAL') ? 'text-red font-bold' : 'text-green'}`}>
              {uploadType === 'video' ? (videoIntervals.length > 0 ? "HIGHLIGHTS FOUND" : "ANALYZING VIDEO") : (scanResult ? scanResult.threat_level : "ANALYZING FEED")}
            </span>
          </div>
        </div>
      </header>

      {/* NAVIGATION TABS */}
      <nav className="dashboard-tabs">
        <button 
          className={`tab-btn glass-panel ${activeTab === 'scanner' ? 'active' : ''}`}
          onClick={() => setActiveTab('scanner')}
        >
          MATCH EVENT SCANNER
        </button>
        <button 
          className={`tab-btn glass-panel ${activeTab === 'schematics' ? 'active' : ''}`}
          onClick={() => {
            setActiveTab('schematics');
            addLog("Viewing system pipeline schematics...", "system");
          }}
        >
          PROCESSING PIPELINE
        </button>
        <button 
          className={`tab-btn glass-panel ${activeTab === 'logs' ? 'active' : ''}`}
          onClick={() => setActiveTab('logs')}
        >
          MODEL SPECS
        </button>
      </nav>

      {/* MAIN CONTAINER */}
      <main className="dashboard-body">
        
        {/* TAB 1: EVENT SCANNER */}
        {activeTab === 'scanner' && (
          <div className="scanner-layout">
            
            {/* LEFT PANEL: UPLOAD AND SCAN */}
            <div className="scanner-control-panel glass-panel">
              <h2>EVENT STREAM SCANNER</h2>
              <p className="panel-desc">Upload a match photo (image) or video file (MP4). The custom YOLOv8 model will run inference to track soccer balls and goalposts, compiling highlight reels automatically.</p>
              
              <div 
                className={`dropzone ${previewUrl || videoUrl ? 'has-preview' : ''}`}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
              >
                {previewUrl ? (
                  <div className="preview-container">
                    <img 
                      src={scanResult ? scanResult.annotated_image : previewUrl} 
                      alt="Match Feed" 
                      className="feed-img"
                    />
                    {isScanning && (
                      <div className="scanning-overlay">
                        <div className="scanning-bar"></div>
                        <div className="scanning-text">RUNNING CUSTOM YOLOv8 INFERENCE...</div>
                      </div>
                    )}
                  </div>
                ) : videoUrl ? (
                  <div className="preview-container video-player-container" style={{ display: 'flex', flexDirection: 'column' }}>
                    <video 
                      ref={videoRef} 
                      src={videoUrl} 
                      className="feed-img" 
                      onTimeUpdate={handleTimeUpdate}
                      onLoadedMetadata={handleLoadedMetadata}
                      onPlay={() => setIsPlaying(true)}
                      onPause={() => setIsPlaying(false)}
                      style={{ width: '100%', height: 'auto', maxHeight: '420px' }}
                    />
                    {isScanningVideo && (
                      <div className="scanning-overlay">
                        <div className="scanning-bar"></div>
                        <div className="scanning-text">SCANNING VIDEO FRAMES ({Math.round(scanProgress * 100)}%)...</div>
                      </div>
                    )}
                    <div className="custom-player-controls" style={{ display: 'flex', alignItems: 'center', width: '100%', background: '#0a0a14', padding: '8px 12px', boxSizing: 'border-box', borderTop: '1px solid var(--color-border)' }}>
                      <button className="control-btn play-pause-btn" onClick={togglePlay} style={{ background: 'none', border: 'none', color: '#fff', cursor: 'pointer', display: 'flex', alignItems: 'center', padding: '4px 8px' }}>
                        {isPlaying ? (
                          <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                            <rect x="6" y="4" width="4" height="16" rx="1" />
                            <rect x="14" y="4" width="4" height="16" rx="1" />
                          </svg>
                        ) : (
                          <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                            <path d="M8 5v14l11-7z" />
                          </svg>
                        )}
                      </button>
                      
                      <button className="control-btn stop-btn" onClick={stopVideo} title="Stop playback" style={{ background: 'none', border: 'none', color: '#fff', cursor: 'pointer', display: 'flex', alignItems: 'center', padding: '4px 8px', marginRight: '8px' }}>
                        <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                          <rect x="4" y="4" width="16" height="16" rx="1" />
                        </svg>
                      </button>
                      
                      <input 
                        type="range" 
                        min={0} 
                        max={videoDuration || 100} 
                        step={0.1}
                        value={videoCurrentTime} 
                        onChange={handleScrubberChange} 
                        className="player-scrubber"
                        style={{ flex: 1, margin: '0 12px', cursor: 'pointer', height: '4px', background: '#27272a' }}
                      />
                      
                      <div className="player-time-display" style={{ fontFamily: 'monospace', fontSize: '0.78rem', color: '#a1a1aa', minWidth: '85px', textAlign: 'right' }}>
                        {formatTime(videoCurrentTime)} / {formatTime(videoDuration)}
                      </div>
                      
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        marginLeft: '12px',
                        padding: '2px 8px',
                        borderRadius: '4px',
                        background: playBounded && boundedStatus === 'completed' ? 'rgba(6, 182, 212, 0.05)' : 'rgba(255, 255, 255, 0.01)',
                        border: '1px solid',
                        borderColor: boundedStatus === 'completed' ? (playBounded ? 'var(--secondary)' : 'var(--color-border)') : '#27272a',
                        height: '24px',
                        boxSizing: 'border-box',
                        boxShadow: playBounded && boundedStatus === 'completed' ? '0 0 8px rgba(6, 182, 212, 0.15)' : 'none',
                        transition: 'all 0.2s ease-in-out'
                      }}>
                        <label style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '6px',
                          cursor: boundedStatus === 'completed' ? 'pointer' : 'not-allowed',
                          fontSize: '0.72rem',
                          fontFamily: 'monospace',
                          color: boundedStatus === 'completed' ? (playBounded ? 'var(--secondary)' : '#a1a1aa') : '#4b5563',
                          margin: 0,
                          userSelect: 'none'
                        }}>
                          <input
                            type="checkbox"
                            checked={playBounded}
                            disabled={boundedStatus !== 'completed'}
                            onChange={handleToggleBounded}
                            style={{
                              cursor: boundedStatus === 'completed' ? 'pointer' : 'not-allowed',
                              accentColor: 'var(--secondary)',
                              width: '13px',
                              height: '13px',
                              margin: 0
                            }}
                          />
                          <span style={{
                            fontWeight: playBounded && boundedStatus === 'completed' ? 'bold' : 'normal',
                            textShadow: playBounded && boundedStatus === 'completed' ? '0 0 4px rgba(6, 182, 212, 0.4)' : 'none'
                          }}>
                            {boundedStatus === 'pending' && "BBOX (STANDBY)"}
                            {boundedStatus === 'rendering' && `BBOX (${Math.round(boundedProgress * 100)}%)`}
                            {boundedStatus === 'completed' && "BBOX"}
                            {boundedStatus === 'failed' && "BBOX (FAILED)"}
                          </span>
                        </label>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="dropzone-placeholder">
                    <p>DRAG & DROP IMAGE OR MP4 VIDEO HERE</p>
                    <span>OR CLICK TO BROWSE LOCAL ASSETS</span>
                  </div>
                )}
                <input 
                  type="file" 
                  accept="image/*,video/mp4" 
                  onChange={handleFileChange} 
                  className="file-input"
                />
              </div>

              {errorMsg && <div className="error-banner">{errorMsg}</div>}

              <div className="action-row">
                <button 
                  className="neon-btn" 
                  onClick={triggerScan}
                  disabled={!selectedFile || isScanning || isScanningVideo}
                >
                  {isScanning || isScanningVideo ? "SCANNING / EXTRACTING..." : "ANALYZE MATCH EVENT / EXTRACT EVENTS"}
                </button>
                
                <button 
                  className={`neon-btn secondary ${audioEnabled ? 'active' : ''}`}
                  onClick={() => setAudioEnabled(!audioEnabled)}
                  title="Toggle Alert Audio"
                >
                  {audioEnabled ? "CHIME ON" : "CHIME MUTED"}
                </button>
              </div>

              {/* Preset Scenarios */}
              <div className="presets-section">
                <h3>SIMULATE MATCH SCENARIOS (IMAGES)</h3>
                <div className="presets-list">
                  {PRESETS.map((preset, idx) => (
                    <button 
                      key={idx} 
                      className="preset-card glass-panel"
                      onClick={() => loadPreset(preset)}
                      disabled={isScanning || isScanningVideo}
                    >
                      <div className="preset-name">{preset.name}</div>
                      <div className="preset-desc">{preset.desc}</div>
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* RIGHT PANEL: DETAILS OR TIMELINE */}
            <div className="scanner-results-panel">
              
              {uploadType === 'video' ? (
                // VIDEO HIGHLIGHT REEL COMPILER TIMELINE
                <div className="threat-status-card glass-panel" style={{ flex: 1 }}>
                  <h3>SUGGESTED HIGHLIGHT CLIPS</h3>
                  
                  {/* Statistics Display Panel */}
                  {videoIntervals.length > 0 && !isScanningVideo && (
                    <div className="stats-dashboard" style={{ display: 'flex', gap: '16px', marginBottom: '16px', background: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                      <div className="stats-dash-card" style={{ flex: 1, textAlign: 'center' }}>
                        <div className="stats-dash-num" style={{ fontSize: '1.4rem', fontWeight: 'bold', color: 'var(--secondary)' }}>
                          {videoIntervals.filter(c => c.correct).length} / {videoIntervals.length}
                        </div>
                        <div className="stats-dash-label" style={{ fontSize: '0.65rem', color: 'var(--text-muted)', letterSpacing: '0.05em', marginTop: '4px' }}>DỰ ĐOÁN ĐÚNG</div>
                        <div className="stats-dash-sub" style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                          Tỉ lệ: {videoIntervals.length > 0 ? Math.round((videoIntervals.filter(c => c.correct).length / videoIntervals.length) * 100) : 0}%
                        </div>
                      </div>
                      <div className="stats-dash-card" style={{ flex: 1, textAlign: 'center', borderLeft: '1px solid rgba(255,255,255,0.1)' }}>
                        <div className="stats-dash-input-container">
                          <input 
                            type="number" 
                            min="0"
                            value={actualGoals}
                            onChange={(e) => setActualGoals(Math.max(0, parseInt(e.target.value) || 0))}
                            className="actual-goals-input"
                            title="Nhập số bàn thắng thực tế"
                          />
                        </div>
                        <div className="stats-dash-label" style={{ fontSize: '0.65rem', color: 'var(--text-muted)', letterSpacing: '0.05em', marginTop: '4px' }}>BÀN THẮNG TRONG VIDEO</div>
                        <div className="stats-dash-sub" style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '2px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                          <span>Xác nhận thực tế</span>
                          <span className={`stats-comparison-badge ${actualGoals === videoIntervals.filter(c => c.type === 'GOAL').length ? 'match' : 'mismatch'}`}>
                            {actualGoals === videoIntervals.filter(c => c.type === 'GOAL').length ? 'Khớp với AI' : `AI đoán: ${videoIntervals.filter(c => c.type === 'GOAL').length}`}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {isScanningVideo ? (
                    <div className="empty-results">
                      <p>PITCH ANALYSIS IN PROGRESS</p>
                      <div style={{ width: '80%', background: '#0a1d12', height: '10px', borderRadius: '5px', overflow: 'hidden', margin: '15px auto' }}>
                        <div style={{ background: '#22c55e', height: '100%', width: `${scanProgress * 100}%`, transition: 'width 0.3s' }}></div>
                      </div>
                      <span>Analyzing frames: {Math.round(scanProgress * 100)}% complete...</span>
                    </div>
                  ) : videoIntervals.length === 0 ? (
                    <div className="empty-results">
                      <p>NO HIGHLIGHTS DETECTED</p>
                      <span>YOLO scanned the video but found no events matching goalmouth or shot patterns.</span>
                    </div>
                  ) : (
                    <div className="summary-details" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                      
                      <div className="detections-list-card" style={{ flex: 1 }}>
                        <div className="detections-scroll" style={{ maxHeight: '380px' }}>
                          {videoIntervals.map((clip) => (
                            <div 
                              key={clip.id} 
                              className={`det-item ${clip.keep ? 'ball-border' : 'empty-list-text'}`}
                              style={{ display: 'flex', gap: '12px', alignItems: 'center', padding: '10px', background: clip.keep ? 'rgba(8, 20, 14, 0.9)' : 'rgba(20, 20, 20, 0.4)', opacity: clip.keep ? 1 : 0.6 }}
                            >
                              <div style={{ position: 'relative', cursor: 'pointer' }} onClick={() => handlePlayShort(clip.start, clip.end)} title="Click to play segment review">
                                <img src={clip.thumbnail} alt="Clip Preview" style={{ width: '80px', height: '45px', objectFit: 'cover', borderRadius: '4px', border: '1px solid var(--color-border)' }} />
                                <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', background: 'rgba(0,0,0,0.6)', borderRadius: '50%', width: '20px', height: '20px', color: 'white', fontSize: '9px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>▶</div>
                              </div>
                              
                              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '4px' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                  <span style={{ fontWeight: 'bold', fontSize: '0.82rem', color: clip.type === 'GOAL' ? '#eab308' : '#06b6d4' }}>
                                    SEGMENT #{clip.id + 1} ({clip.type})
                                  </span>
                                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                                    CONF: {Math.round(clip.confidence * 100)}%
                                  </span>
                                </div>
                                <p style={{ fontSize: '0.74rem', color: 'var(--text-secondary)', margin: 0 }}>{clip.description}</p>
                                
                                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginTop: '6px', flexWrap: 'wrap' }}>
                                  <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                    <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Trim:</span>
                                    <input 
                                      type="number" 
                                      value={clip.start} 
                                      step="0.1"
                                      onChange={(e) => {
                                        const val = parseFloat(e.target.value);
                                        setVideoIntervals(prev => prev.map(c => c.id === clip.id ? { ...c, start: val } : c));
                                      }}
                                      style={{ width: '50px', background: '#020205', border: '1px solid var(--color-border)', color: '#fff', fontSize: '0.72rem', padding: '2px' }}
                                    />
                                    <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>to</span>
                                    <input 
                                      type="number" 
                                      value={clip.end} 
                                      step="0.1"
                                      onChange={(e) => {
                                        const val = parseFloat(e.target.value);
                                        setVideoIntervals(prev => prev.map(c => c.id === clip.id ? { ...c, end: val } : c));
                                      }}
                                      style={{ width: '50px', background: '#020205', border: '1px solid var(--color-border)', color: '#fff', fontSize: '0.72rem', padding: '2px' }}
                                    />
                                  </div>

                                  <label style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.72rem', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                                    <input 
                                      type="checkbox" 
                                      checked={clip.correct} 
                                      onChange={(e) => {
                                        const checked = e.target.checked;
                                        setVideoIntervals(prev => prev.map(c => c.id === clip.id ? { ...c, correct: checked } : c));
                                      }}
                                    />
                                    Dự đoán đúng
                                  </label>

                                  <label style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.72rem', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                                    <input 
                                      type="checkbox" 
                                      checked={clip.isGoal} 
                                      onChange={(e) => {
                                        const checked = e.target.checked;
                                        setVideoIntervals(prev => prev.map(c => {
                                          if (c.id === clip.id) {
                                            const isAiGoal = c.type === 'GOAL';
                                            return { ...c, isGoal: checked, correct: checked === isAiGoal };
                                          }
                                          return c;
                                        }));
                                      }}
                                    />
                                    Có bàn thắng
                                  </label>
                                </div>
                              </div>
                              
                              <button 
                                onClick={() => setVideoIntervals(prev => prev.map(c => c.id === clip.id ? { ...c, keep: !c.keep } : c))}
                                style={{ background: clip.keep ? '#082f1b' : '#27272a', border: '1px solid var(--color-border)', color: clip.keep ? '#22c55e' : '#a1a1aa', borderRadius: '4px', padding: '6px 10px', fontSize: '0.7rem', cursor: 'pointer' }}
                              >
                                {clip.keep ? "KEEP" : "SKIP"}
                              </button>
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      <div style={{ marginTop: '15px', borderTop: '1px solid rgba(34, 197, 94, 0.1)', paddingTop: '15px' }}>
                        <div className="export-settings-panel" style={{ marginBottom: '15px', padding: '12px', background: 'rgba(255, 255, 255, 0.02)', border: '1px solid rgba(255, 255, 255, 0.05)', borderRadius: '6px' }}>
                          <h4 style={{ margin: '0 0 10px 0', fontSize: '0.8rem', color: 'var(--secondary)', letterSpacing: '0.05em', textTransform: 'uppercase' }}>Cấu hình định dạng xuất</h4>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.76rem', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                              <input 
                                type="checkbox" 
                                checked={exportOptions.markdown} 
                                onChange={(e) => setExportOptions(prev => ({ ...prev, markdown: e.target.checked }))}
                                style={{ accentColor: 'var(--secondary)' }}
                              />
                              Xuất báo cáo chi tiết (.md) - Kết quả đánh giá của bạn
                            </label>
                            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.76rem', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                              <input 
                                type="checkbox" 
                                checked={exportOptions.clean} 
                                onChange={(e) => setExportOptions(prev => ({ ...prev, clean: e.target.checked }))}
                                style={{ accentColor: 'var(--secondary)' }}
                              />
                              Xuất video highlights sạch (Không vẽ Bounding Box)
                            </label>
                            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.76rem', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                              <input 
                                type="checkbox" 
                                checked={exportOptions.bounding} 
                                onChange={(e) => setExportOptions(prev => ({ ...prev, bounding: e.target.checked }))}
                                style={{ accentColor: 'var(--secondary)' }}
                              />
                              Xuất video highlights có Bounding Box (YOLO v8)
                            </label>
                          </div>
                        </div>

                        <button 
                          className="neon-btn" 
                          onClick={exportHighlightReel}
                          disabled={isExporting || videoIntervals.filter(c => c.keep).length === 0 || (!exportOptions.markdown && !exportOptions.clean && !exportOptions.bounding)}
                          style={{ width: '100%' }}
                        >
                          {isExporting ? "EXPORTING & COMPILING..." : (
                            !exportOptions.markdown && !exportOptions.clean && !exportOptions.bounding ? "VUI LÒNG CHỌN ĐỊNH DẠNG XUẤT" :
                            exportOptions.markdown && !exportOptions.clean && !exportOptions.bounding ? "XUẤT BÁO CÁO PHÂN TÍCH (.MD)" :
                            !exportOptions.markdown && exportOptions.clean && !exportOptions.bounding ? "XUẤT HIGHLIGHTS SẠCH (MP4)" :
                            !exportOptions.markdown && !exportOptions.clean && exportOptions.bounding ? "XUẤT HIGHLIGHTS BOUNDING BOX (MP4)" :
                            "XUẤT DỮ LIỆU HIGHLIGHTS (ZIP)"
                          )}
                        </button>
                        {isExporting && (
                          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textAlign: 'center', marginTop: '8px', fontStyle: 'italic' }}>
                            ⚡ Cutting and stitching clips on server. Please stand by...
                          </div>
                        )}
                      </div>
                      
                    </div>
                  )}
                </div>
              ) : (
                // IMAGE STATIC SCAN SUMMARY
                <div className="threat-status-card glass-panel">
                  <h3>EVENT ANALYSIS SUMMARY</h3>
                  
                  {scanResult ? (
                    <div className="summary-details">
                      <div className="threat-gauge">
                        <div className={`gauge-reading ${scanResult.threat_level}`}>
                          {scanResult.threat_level}
                        </div>
                        <p className="gauge-label">HIGHLIGHT PROBABILITY</p>
                      </div>

                      <div className="count-stats">
                        <div className="stat-box yellow-glow">
                          <span className="stat-count">{scanResult.threat_count}</span>
                          <span className="stat-label">BALLS DETECTED</span>
                        </div>
                        <div className="stat-box cyan-glow">
                          <span className="stat-count">{scanResult.ally_count}</span>
                          <span className="stat-label">GOALPOSTS DETECTED</span>
                        </div>
                      </div>

                      <div className="instruction-box">
                        <div className="instr-header">EDITOR ACTION / RECOMMENDATION:</div>
                        <p className="instr-text">"{scanResult.instruction}"</p>
                      </div>

                      <div className="detections-list-card">
                        <h4>CLASSIFIED EVENT OBJECTS</h4>
                        <div className="detections-scroll">
                          {scanResult.detections.length === 0 ? (
                            <div className="empty-list-text">No ball or goalpost objects detected in this frame.</div>
                          ) : (
                            scanResult.detections.map((det, i) => (
                              <div key={i} className={`det-item ${det.class_id === 0 ? 'ball-border' : 'goalpost-border'}`}>
                                <span className="det-name">{det.mapped_name}</span>
                                <span className="det-orig">Class: {det.class_id}</span>
                                <span className="det-conf">CONF: {Math.round(det.confidence * 100)}%</span>
                              </div>
                            ))
                          )}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="empty-results">
                      <p>ANALYSIS ENGINE STANDBY</p>
                      <span>Upload a soccer match photo or simulate a preset scenario to view highlight analysis logs.</span>
                    </div>
                  )}
                </div>
              )}

              {/* MOCK TACTICAL LOG TERMINAL */}
              <div className="console-panel glass-panel">
                <div className="console-header">
                  <span className="blink-dot green"></span>
                  <span className="console-title">HIGHLIGHT_COMPILER_LOG.SH</span>
                </div>
                <div className="console-log-area" ref={logContainerRef}>
                  {logs.map((log, i) => (
                    <div key={i} className={`log-line ${log.type}`}>
                      <span className="log-time">[{log.time}]</span>{' '}
                      <span className="log-prefix">{log.type.toUpperCase()}:</span>{' '}
                      <span className="log-text">{log.text}</span>
                    </div>
                  ))}
                </div>
              </div>

            </div>

          </div>
        )}

        {/* TAB 2: PIPELINE SCHEMATICS */}
        {activeTab === 'schematics' && (
          <div className="schematics-layout">
            
            {/* PIPELINE CONTROL CARD */}
            <div className="pipeline-view glass-panel">
              <div className="pipeline-header">
                <h2>SYSTEM FLOW SCHEMATICS</h2>
                <div className="flow-selector-tabs">
                  <button 
                    className={`flow-tab-btn ${activeSvgTab === 'flow1' ? 'active' : ''}`}
                    onClick={() => {
                      setActiveSvgTab('flow1');
                      setSelectedNode(null);
                      setExplanationText("Click on any component node inside the pipeline flowchart to inspect its backend mechanics and architectural details.");
                      addLog("Switched flowchart: Stage 1 - Startup & Upload", "system");
                    }}
                  >
                    PHASE 1: UPLOAD PIPELINE
                  </button>
                  <button 
                    className={`flow-tab-btn ${activeSvgTab === 'flow2' ? 'active' : ''}`}
                    onClick={() => {
                      setActiveSvgTab('flow2');
                      setSelectedNode(null);
                      setExplanationText("Click on any component node inside the pipeline flowchart to inspect its backend mechanics and architectural details.");
                      addLog("Switched flowchart: Stage 2 - Worker Inference", "system");
                    }}
                  >
                    PHASE 2: WORKER INFERENCE
                  </button>
                  <button 
                    className={`flow-tab-btn ${activeSvgTab === 'flow3' ? 'active' : ''}`}
                    onClick={() => {
                      setActiveSvgTab('flow3');
                      setSelectedNode(null);
                      setExplanationText("Click on any component node inside the pipeline flowchart to inspect its backend mechanics and architectural details.");
                      addLog("Switched flowchart: Stage 3 - Dashboard Export", "system");
                    }}
                  >
                    PHASE 3: DASHBOARD EXPORT
                  </button>
                </div>
              </div>

              {/* RENDER THE INTERACTIVE INJECTED SVG */}
              <div className="svg-viewport-container">
                <p className="viewport-tip">Interactive Diagram: Click on individual flowchart elements below to inspect their code mechanics.</p>
                <div 
                  className="interactive-svg-wrapper"
                  dangerouslySetInnerHTML={{ __html: getSvgContent() }}
                  onClick={handleSvgClick}
                />
              </div>
            </div>

            {/* DIAGNOSTIC OPERATOR CARD */}
            <div className="operator-panel glass-panel">
              <div className="panel-header">
                <span className="terminal-prompt">AI_OPERATOR&gt;</span>
                <h3>PIPELINE NODE ANALYZER</h3>
              </div>
              
              <div className="operator-content">
                {selectedNode ? (
                  <div className="node-diagnostic-readout">
                    <div className="node-query-bubble">
                      <span className="bubble-label">OPERATOR QUERY:</span>
                      <p>"{selectedNode}"</p>
                    </div>
                    
                    <div className="node-answer-bubble">
                      <span className="bubble-label">SYSTEM METRICS / RESPONSE:</span>
                      <p className="answer-text">{explanationText}</p>
                    </div>
                    
                    <div className="node-metadata">
                      <div className="meta-row">
                        <span>COMPONENT STATE:</span>
                        <span className="text-green">ONLINE</span>
                      </div>
                      <div className="meta-row">
                        <span>LATENCY:</span>
                        <span>&lt; 5ms</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="operator-idle">
                    <p>CONSOLE STANDBY</p>
                    <span>Click any element block in the flowchart (like 'Database', 'FastAPI', 'YOLO inference') to analyze its detailed pipeline parameters.</span>
                  </div>
                )}
              </div>
            </div>

          </div>
        )}

        {/* TAB 3: LOGS & SPECS */}
        {activeTab === 'logs' && (
          <div className="logs-layout glass-panel">
            <h2>SYSTEM CONFIGURATION & MODEL SPECIFICATIONS</h2>
            <p className="panel-desc">Technical specifications of the YOLOv8 custom soccer match goal and ball tracking neural network.</p>
            
            <div className="specs-table-wrapper">
              <table className="specs-table">
                <thead>
                  <tr>
                    <th>Specification Component</th>
                    <th>Configuration Value</th>
                    <th>Purpose / Details</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Detection Framework</td>
                    <td>Ultralytics YOLOv8 Custom</td>
                    <td>Custom trained to detect football balls and goalpost structures.</td>
                  </tr>
                  <tr>
                    <td>Neural Architecture Weights</td>
                    <td>best.pt (Custom weights)</td>
                    <td>Located in the web directory. Optimized for sports telemetry and highlight generation.</td>
                  </tr>
                  <tr>
                    <td>Mapped Ball Class</td>
                    <td>Class 0 (ball)</td>
                    <td>Yellow bounding boxes. Monitored for speed, trajectory, and proximity.</td>
                  </tr>
                  <tr>
                    <td>Mapped Goalpost Class</td>
                    <td>Class 1 (goalpost)</td>
                    <td>Cyan/Blue bounding boxes. Monitored for goal mouth events.</td>
                  </tr>
                  <tr>
                    <td>Highlight Trigger Metrics</td>
                    <td>Distance & Intersection check</td>
                    <td>Checks whether the ball bounding box overlaps or resides within the goalpost bounding box.</td>
                  </tr>
                  <tr>
                    <td>Inference Runtime Environment</td>
                    <td>PyTorch 2.12.0 / CPU (CUDA accelerated if available)</td>
                    <td>Runs in local python interpreter with ultralytics packages.</td>
                  </tr>
                  <tr>
                    <td>Backend API Server</td>
                    <td>FastAPI + Uvicorn (Port 8000)</td>
                    <td>High-performance, async web framework for handling multipart binary file uploads and base64 replies.</td>
                  </tr>
                  <tr>
                    <td>Frontend UI Server</td>
                    <td>React (Vite) + Vanilla CSS (Port 5173)</td>
                    <td>Asynchronous, responsive visual dashboard utilizing glassmorphism aesthetics and custom Web Audio API alarms.</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="logs-terminal-large">
              <h3>HIGHLIGHT COMPILER SEQUENCE DIAGNOSTIC LOGS</h3>
              <pre className="large-console">
{`[SYSTEM BOOT] COMMAND: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
[SYSTEM BOOT] INFO:     Will watch for changes in these directories: ['E:\\_FPT_UNI_\\Smartlab\\web\\backend']
[SYSTEM BOOT] INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
[SYSTEM BOOT] INFO:     Started reloader process [12492] using StatReload
[SYSTEM BOOT] INFO:     Loading custom best.pt weights...
[SYSTEM BOOT] torch.cuda.is_available() = False (Fallback to CPU mode)
[SYSTEM BOOT] best.pt loaded. 2 classes registered.
[SYSTEM BOOT] Application health checklist:
   - [OK] FastAPI Router initialized
   - [OK] CORS Middleware configured for origins: [*]
   - [OK] Ball & Goalpost proximity heuristic compiler active
   - [OK] Base64 image compiler functional
[SYSTEM BOOT] API ready to compile soccer match highlight frames. Status: STANDBY.`}
              </pre>
            </div>
          </div>
        )}
        
      </main>

      <footer className="dashboard-footer-info">
        <p>© 2026 Smartlab Football Highlight Analyzer. Powered by YOLOv8 and FastAPI. Authorized personnel only.</p>
      </footer>
    </div>
  );
}

export default App;
