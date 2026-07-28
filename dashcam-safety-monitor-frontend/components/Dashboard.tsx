"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Volume2,
  VolumeX,
  ShieldAlert,
  Play,
  Pause,
  RefreshCw,
  Activity,
  Layers,
  AlertTriangle,
  Radio,
  List,
  Filter,
  Flame,
  FileVideo,
  Download,
  Eye,
  Zap,
  CheckCircle2,
  Sparkles,
  Loader2,
  Clock,
  Cpu,
  ShieldCheck
} from "lucide-react";

export interface Detection {
  bbox: [number, number, number, number];
  confidence: number;
  class_name: string;
  category: "anomaly" | "lane_line" | "pothole" | "road_sign";
  color: [number, number, number];
  priority_rank?: number;
  priority_level?: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";
}

export interface ProcessedImageResult {
  status: string;
  active_models: string[];
  highest_priority: string;
  audio_trigger: boolean;
  total_detections: number;
  detections: Detection[];
  annotated_image: string;
}

export interface ProcessedVideoResult {
  status: string;
  filename: string;
  total_frames: number;
  fps: number;
  processed_samples: number;
  active_models: string[];
  highest_priority: string;
  audio_trigger_count: number;
  category_counts: Record<string, number>;
  annotated_video_url?: string;
  timeline_alerts: {
    frame_idx: number;
    timestamp: number;
    highest_priority: string;
    audio_trigger: boolean;
    total_detections: number;
    detections: Detection[];
    annotated_frame: string;
  }[];
}

const DASHBOARD_MODELS = [
  { id: "anomaly", label: "P1: Anomaly", activeClass: "bg-red-500/20 text-red-400 border-red-500/50 shadow-sm shadow-red-500/20" },
  { id: "pothole", label: "P2: Potholes", activeClass: "bg-orange-500/20 text-orange-400 border-orange-500/50" },
  { id: "lane_line", label: "P3: Lane Lines", activeClass: "bg-cyan-500/20 text-cyan-400 border-cyan-500/50" },
  { id: "road_sign", label: "P4: Road Signs", activeClass: "bg-yellow-500/20 text-yellow-400 border-yellow-500/50" }
];

interface DashboardProps {
  mediaType: "image" | "video" | null;
  mediaFile: File | null;
  selectedModels: string[];
  onSelectedModelsChange: (models: string[]) => void;
  imageResult: ProcessedImageResult | null;
  videoResult: ProcessedVideoResult | null;
  onReset: () => void;
  onRunServerVideoProcess?: () => void;
  isServerProcessingVideo?: boolean;
}

export const Dashboard: React.FC<DashboardProps> = ({
  mediaType,
  mediaFile,
  selectedModels,
  onSelectedModelsChange,
  imageResult,
  videoResult,
  onReset,
  onRunServerVideoProcess,
  isServerProcessingVideo = false
}) => {
  // Sound controls
  const [soundEnabled, setSoundEnabled] = useState(true);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const lastAudioTriggerRef = useRef<number>(0);

  // Video Streaming State
  const [isPlaying, setIsPlaying] = useState(false);
  const [wsStatus, setWsStatus] = useState<"disconnected" | "connecting" | "connected" | "error">("disconnected");
  const [wsErrorDetail, setWsErrorDetail] = useState<string | null>(null);
  const [liveAnnotatedFrame, setLiveAnnotatedFrame] = useState<string | null>(null);
  const [showOverlay, setShowOverlay] = useState<boolean>(true);

  // Alert Feed State & Priority Filter
  const [priorityFilter, setPriorityFilter] = useState<string>("ALL");
  const [alertsFeed, setAlertsFeed] = useState<
    { id: string; timestamp: string; priority: string; priorityRank: number; detections: Detection[]; audioTrigger: boolean }[]
  >([]);
  const [currentPriority, setCurrentPriority] = useState<string>("normal");
  const [stats, setStats] = useState({ totalFrames: 0, totalAlerts: 0 });

  // Processing Timer Counter for Server Video processing
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  const lastAlertFeedUpdateRef = useRef<number>(0);
  const isPlayingRef = useRef<boolean>(false);
  isPlayingRef.current = isPlaying;

  // Timeline slider for server processed video
  const [selectedTimelineIdx, setSelectedTimelineIdx] = useState<number>(0);

  // Video element & canvas references
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Timer effect for server processing loader
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isServerProcessingVideo) {
      setElapsedSeconds(0);
      interval = setInterval(() => {
        setElapsedSeconds((prev) => prev + 1);
      }, 1000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isServerProcessingVideo]);

  // Pre-load audio alert element
  useEffect(() => {
    const audio = new Audio("/alert.mp3");
    audio.preload = "auto";
    audioRef.current = audio;

    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
      }
    };
  }, []);

  // Non-blocking Audio playback function
  const playAlertSound = (priorityLevel?: string) => {
    if (!soundEnabled || !audioRef.current) return;
    const now = Date.now();
    const cooldown = priorityLevel === "CRITICAL" ? 1200 : 1800;
    if (now - lastAudioTriggerRef.current > cooldown) {
      lastAudioTriggerRef.current = now;
      try {
        audioRef.current.currentTime = 0;
        const playPromise = audioRef.current.play();
        if (playPromise !== undefined) {
          playPromise.catch(() => { });
        }
      } catch {
        // Ignore audio errors
      }
    }
  };

  // Format timer into MM:SS format
  const formatTimer = (totalSec: number) => {
    const m = Math.floor(totalSec / 60);
    const s = totalSec % 60;
    const mStr = m < 10 ? `0${m}` : `${m}`;
    const sStr = s < 10 ? `0${s}` : `${s}`;
    return `${mStr}:${sStr}s`;
  };

  // Trigger audio on image processing result
  useEffect(() => {
    if (mediaType === "image" && imageResult?.audio_trigger) {
      playAlertSound(imageResult.highest_priority === "anomaly" ? "CRITICAL" : "HIGH");
    }
  }, [imageResult, mediaType]);

  // Connect WebSocket helper function
  const connectWebSocket = () => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws/video";
    setWsStatus("connecting");
    setWsErrorDetail(null);

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setWsStatus("connected");
        setWsErrorDetail(null);
        console.log("[Dashboard WS] Connected to backend endpoint successfully.");
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.status === "connected") return;

          if (data.status === "success") {
            if (data.annotated_frame) {
              setLiveAnnotatedFrame(data.annotated_frame);
            }
            setCurrentPriority(data.highest_priority);
            setStats((prev) => ({
              totalFrames: prev.totalFrames + 1,
              totalAlerts: prev.totalAlerts + (data.detections.length > 0 ? 1 : 0)
            }));

            if (data.audio_trigger) {
              playAlertSound(data.highest_priority === "anomaly" ? "CRITICAL" : "HIGH");
            }

            const now = Date.now();
            if (data.detections && data.detections.length > 0 && now - lastAlertFeedUpdateRef.current > 350) {
              lastAlertFeedUpdateRef.current = now;
              const pRank = data.highest_priority === "anomaly" ? 1 : data.highest_priority === "pothole" ? 2 : data.highest_priority === "lane_line" ? 3 : 4;
              const newAlert = {
                id: `${now}-${Math.random()}`,
                timestamp: new Date().toLocaleTimeString(),
                priority: data.highest_priority,
                priorityRank: pRank,
                detections: data.detections,
                audioTrigger: data.audio_trigger
              };
              setAlertsFeed((prev) => [newAlert, ...prev.slice(0, 19)]);
            }
          }
        } catch (err) {
          console.error("[Dashboard WS] Frame parse error:", err);
        }
      };

      ws.onerror = () => {
        console.warn("[Dashboard WS] Connection error or backend unreachable.");
        setWsStatus("error");
        setWsErrorDetail(`WebSocket server unreachable at ${wsUrl}. Ensure the backend is running and reachable.`);
      };

      ws.onclose = () => {
        setWsStatus((prev) => (prev === "error" ? "error" : "disconnected"));
        if (isPlayingRef.current) {
          setTimeout(connectWebSocket, 500);
        }
      };
    } catch (err: any) {
      setWsStatus("error");
      setWsErrorDetail(err.message || "Failed to initialize WebSocket connection.");
    }
  };

  // Video WebSocket setup
  useEffect(() => {
    if (mediaType !== "video" || !mediaFile || videoResult) return;

    const videoUrl = URL.createObjectURL(mediaFile);
    if (videoRef.current) {
      videoRef.current.src = videoUrl;
      videoRef.current.load();
    }

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.onopen = null;
        wsRef.current.onmessage = null;
        wsRef.current.onerror = null;
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      URL.revokeObjectURL(videoUrl);
    };
  }, [mediaType, mediaFile, videoResult]);

  // Frame capture loop over WebSocket (~15 fps)
  const streamFrameLoop = () => {
    if (!videoRef.current || !canvasRef.current) {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (video.paused || video.ended) {
      setIsPlaying(false);
      return;
    }

    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      if (isPlayingRef.current) {
        connectWebSocket();
        setTimeout(() => {
          if (isPlayingRef.current) {
            animationFrameRef.current = requestAnimationFrame(streamFrameLoop);
          }
        }, 150);
      }
      return;
    }

    const ctx = canvas.getContext("2d");
    if (ctx && video.videoWidth > 0 && video.videoHeight > 0) {
      canvas.width = Math.min(640, video.videoWidth);
      canvas.height = Math.min(360, video.videoHeight);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const frame_b64 = canvas.toDataURL("image/jpeg", 0.75);

      const payload = {
        command: "process_frame",
        active_models: selectedModels,
        frame_idx: video.currentTime,
        timestamp: video.currentTime,
        frame_b64: frame_b64
      };

      wsRef.current.send(JSON.stringify(payload));
    }

    setTimeout(() => {
      if (isPlayingRef.current) {
        animationFrameRef.current = requestAnimationFrame(streamFrameLoop);
      }
    }, 65);
  };

  const toggleVideoPlay = () => {
    if (!videoRef.current) return;

    if (wsStatus !== "connected") {
      connectWebSocket();
    }

    if (isPlaying) {
      videoRef.current.pause();
      setIsPlaying(false);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    } else {
      videoRef.current.play().then(() => {
        setIsPlaying(true);
        animationFrameRef.current = requestAnimationFrame(streamFrameLoop);
      }).catch((err) => {
        console.error("Video play error:", err);
      });
    }
  };

  const toggleModelPill = (id: string) => {
    let next: string[];
    if (selectedModels.includes(id)) {
      next = selectedModels.filter((m) => m !== id);
    } else {
      next = [...selectedModels, id];
    }
    onSelectedModelsChange(next);
  };

  const runAllModelsParallel = () => {
    onSelectedModelsChange(["anomaly", "lane_line", "pothole", "road_sign"]);
  };

  const getPriorityBadge = (priority: string) => {
    switch (priority) {
      case "anomaly":
        return <span className="px-3 py-1 rounded-md text-xs font-extrabold bg-red-500/20 text-red-400 border border-red-500/50 shadow-sm shadow-red-500/20 flex items-center gap-1.5"><Flame className="w-3.5 h-3.5 text-red-400" /> PRIORITY 1: CRITICAL ANOMALY</span>;
      case "pothole":
        return <span className="px-3 py-1 rounded-md text-xs font-extrabold bg-orange-500/20 text-orange-400 border border-orange-500/50 flex items-center gap-1.5"><AlertTriangle className="w-3.5 h-3.5 text-orange-400" /> PRIORITY 2: POTHOLE HAZARD</span>;
      case "lane_line":
        return <span className="px-3 py-1 rounded-md text-xs font-extrabold bg-cyan-500/20 text-cyan-400 border border-cyan-500/50 flex items-center gap-1.5"><Activity className="w-3.5 h-3.5 text-cyan-400" /> PRIORITY 3: LANE DEPARTURE</span>;
      case "road_sign":
        return <span className="px-3 py-1 rounded-md text-xs font-extrabold bg-yellow-500/20 text-yellow-400 border border-yellow-500/50 flex items-center gap-1.5"><Layers className="w-3.5 h-3.5 text-yellow-400" /> PRIORITY 4: ROAD SIGN</span>;
      default:
        return <span className="px-3 py-1 rounded-md text-xs font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40">NORMAL DRIVING</span>;
    }
  };

  const filteredAlerts = alertsFeed.filter((a) => {
    if (priorityFilter === "ALL") return true;
    return a.priority.toUpperCase() === priorityFilter.toUpperCase();
  });

  const apiHost = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  return (
    <div className="space-y-6 relative">
      <canvas ref={canvasRef} className="hidden" />

      {/* ULTRA-MODERN HIGH-TECH SERVER PROCESSING MODAL OVERLAY */}
      {isServerProcessingVideo && (
        <div className="fixed inset-0 z-50 bg-slate-950/85 backdrop-blur-xl flex items-center justify-center p-4">
          <div className="bg-slate-900/90 border border-indigo-500/50 rounded-3xl p-8 max-w-lg w-full shadow-2xl space-y-6 relative overflow-hidden ring-1 ring-indigo-500/30">

            {/* Pulsing neon top accent bar */}
            <div className="absolute top-0 inset-x-0 h-1.5 bg-gradient-to-r from-violet-600 via-indigo-500 to-cyan-400 animate-pulse" />

            {/* Central Animated Radar / AI Scanner graphic */}
            <div className="relative flex items-center justify-center py-3">
              <div className="absolute w-28 h-28 rounded-full border border-indigo-500/20 animate-ping" />
              <div className="absolute w-20 h-20 rounded-full border-2 border-indigo-500/40 border-t-indigo-400 animate-spin" />
              <div className="p-5 bg-gradient-to-tr from-indigo-600 to-violet-600 rounded-3xl shadow-xl shadow-indigo-500/30 relative z-10 flex items-center justify-center">
                <Cpu className="w-8 h-8 text-white animate-pulse" />
              </div>
            </div>

            <div className="text-center space-y-1">
              <h3 className="text-xl font-extrabold text-white tracking-wide">
                Server Rendering Full Bounding Box Video
              </h3>
              <p className="text-xs text-slate-400">
                Executing multi-model PyTorch pipeline with OpenCV priority overlay
              </p>
            </div>

            {/* Live Metrics & Timer Bar */}
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 bg-slate-950/70 border border-slate-800 rounded-2xl flex items-center justify-between">
                <div>
                  <span className="text-[10px] text-slate-400 block font-semibold">Elapsed Time</span>
                  <span className="text-lg font-extrabold text-indigo-400 font-mono">
                    {formatTimer(elapsedSeconds)}
                  </span>
                </div>
                <Clock className="w-5 h-5 text-indigo-400 opacity-60" />
              </div>

              <div className="p-3 bg-slate-950/70 border border-slate-800 rounded-2xl flex items-center justify-between">
                <div>
                  <span className="text-[10px] text-slate-400 block font-semibold">Engine Speed</span>
                  <span className="text-xs font-bold text-emerald-400">Stride 3 (Fast)</span>
                </div>
                <Zap className="w-5 h-5 text-emerald-400 opacity-60" />
              </div>
            </div>

            {/* Step-by-Step Live Processing Checklist */}
            <div className="space-y-3 bg-slate-950/80 p-4 rounded-2xl border border-slate-800">
              <div className="flex items-center gap-3 text-xs font-semibold text-emerald-400">
                <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-400" />
                <span>Uploaded Video File Payload to FastAPI Backend</span>
              </div>
              <div className="flex items-center gap-3 text-xs font-semibold text-indigo-300">
                <Loader2 className="w-4 h-4 shrink-0 text-indigo-400 animate-spin" />
                <span>Evaluating Active Models ({selectedModels.join(", ")})</span>
              </div>
              <div className="flex items-center gap-3 text-xs text-slate-400">
                <ShieldCheck className="w-4 h-4 shrink-0 text-slate-600" />
                <span>Generating Annotated MP4 Video with Download Link</span>
              </div>
            </div>

            {/* Glowing Shimmer Progress Bar */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs text-slate-400 font-mono">
                <span className="flex items-center gap-1">
                  <Sparkles className="w-3.5 h-3.5 text-indigo-400" /> Rendering Frames...
                </span>
                <span className="text-indigo-400 font-bold">Please Wait</span>
              </div>
              <div className="w-full bg-slate-950 rounded-full h-2.5 overflow-hidden border border-slate-800 relative">
                <div className="h-full bg-gradient-to-r from-violet-600 via-indigo-500 to-cyan-400 rounded-full animate-pulse w-4/5" />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Top Header Bar */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 sm:p-6 shadow-xl backdrop-blur-md flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-indigo-600/20 border border-indigo-500/30 rounded-xl text-indigo-400">
            <Activity className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-xl font-bold text-white tracking-wide">
                Live Road Safety Monitor
              </h2>
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/40">
                {mediaType?.toUpperCase()} MODE
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Running Parallel Pipeline: <code className="text-indigo-300 font-bold">{selectedModels.length === 0 ? "ALL 4 MODELS" : selectedModels.join(", ")}</code>
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setSoundEnabled(!soundEnabled)}
            className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold border transition-all ${soundEnabled
              ? "bg-indigo-600/20 text-indigo-300 border-indigo-500/40 hover:bg-indigo-600/30"
              : "bg-slate-800 text-slate-400 border-slate-700 hover:bg-slate-700"
              }`}
          >
            {soundEnabled ? <Volume2 className="w-4 h-4 text-indigo-400" /> : <VolumeX className="w-4 h-4" />}
            <span>Web Audio {soundEnabled ? "ON" : "MUTED"}</span>
          </button>

          <button
            onClick={onReset}
            className="flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold bg-slate-800 text-slate-300 border border-slate-700 hover:bg-slate-700 transition-all"
          >
            <RefreshCw className="w-4 h-4" /> Reset Workspace
          </button>
        </div>
      </div>

      {/* Dynamic Model Control Bar (Change active models live after upload) */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 shadow-xl backdrop-blur-md flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-2 text-xs font-bold text-white">
          <Zap className="w-4 h-4 text-indigo-400" />
          <span>Active Models Filter:</span>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {DASHBOARD_MODELS.map((m) => {
            const active = selectedModels.includes(m.id);
            return (
              <button
                key={m.id}
                type="button"
                onClick={() => toggleModelPill(m.id)}
                className={`px-3 py-1.5 rounded-xl text-xs font-extrabold border transition-all ${active
                  ? m.activeClass
                  : "bg-slate-950 text-slate-500 border-slate-800 opacity-50 hover:opacity-90"
                  }`}
              >
                {active ? "✓ " : "+ "}{m.label}
              </button>
            );
          })}

          <button
            type="button"
            onClick={runAllModelsParallel}
            className="px-3.5 py-1.5 rounded-xl text-xs font-extrabold bg-gradient-to-r from-violet-600 to-indigo-600 text-white shadow-md hover:from-violet-500 hover:to-indigo-500 transition-all ml-1"
          >
            ⚡ Run All Parallel
          </button>
        </div>
      </div>

      {/* IMAGE MODE VIEW */}
      {mediaType === "image" && imageResult && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 bg-slate-900/80 border border-slate-800 rounded-2xl p-4 shadow-xl backdrop-blur-md">
            <div className="flex items-center justify-between mb-3 px-2">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Layers className="w-4 h-4 text-indigo-400" /> YOLOv8 Priority-Annotated Image Output
              </h3>
              {getPriorityBadge(imageResult.highest_priority)}
            </div>

            <div className="relative rounded-xl overflow-hidden bg-black border border-slate-800 flex items-center justify-center min-h-[360px]">
              <img
                src={imageResult.annotated_image}
                alt="YOLO annotated output"
                className="max-h-[500px] w-full object-contain"
              />
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl backdrop-blur-md">
              <h3 className="text-sm font-bold text-white mb-4 flex items-center gap-2">
                <ShieldAlert className="w-4 h-4 text-indigo-400" /> Per-Class Priority Breakdown
              </h3>

              <div className="grid grid-cols-2 gap-3 mb-4">
                <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-xl">
                  <span className="text-xs text-slate-400 block">Total Detections</span>
                  <span className="text-2xl font-bold text-white">{imageResult.total_detections}</span>
                </div>
                <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-xl">
                  <span className="text-xs text-slate-400 block">Audio Alert Flag</span>
                  <span className={`text-sm font-bold ${imageResult.audio_trigger ? "text-red-400" : "text-emerald-400"}`}>
                    {imageResult.audio_trigger ? "TRIGGERED" : "CLEAR"}
                  </span>
                </div>
              </div>

              <div className="space-y-2.5 max-h-[340px] overflow-y-auto pr-1">
                {imageResult.detections.length === 0 ? (
                  <p className="text-xs text-slate-400 italic text-center py-6">
                    No hazards detected in current image frame for active models.
                  </p>
                ) : (
                  imageResult.detections.map((det, idx) => (
                    <div
                      key={idx}
                      className="p-3 bg-slate-950/70 border border-slate-800 rounded-xl space-y-1.5"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-white text-xs">{det.class_name}</span>
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${det.priority_level === "CRITICAL" ? "bg-red-500/20 text-red-400 border border-red-500/40" :
                          det.priority_level === "HIGH" ? "bg-orange-500/20 text-orange-400 border border-orange-500/40" :
                            det.priority_level === "MEDIUM" ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/40" :
                              "bg-yellow-500/20 text-yellow-400 border border-yellow-500/40"
                          }`}>
                          {det.priority_level || "LOW"}
                        </span>
                      </div>
                      <div className="flex items-center justify-between text-[11px] text-slate-400">
                        <span>Category: {det.category}</span>
                        <span className="font-semibold text-indigo-300">{(det.confidence * 100).toFixed(0)}% Conf</span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* VIDEO MODE VIEW */}
      {mediaType === "video" && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-4">
            {/* Action Bar for Server-Side Video Processing vs Live Streaming */}
            <div className="p-5 bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-indigo-500/30 rounded-2xl flex flex-wrap items-center justify-between gap-4 shadow-xl">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-indigo-400" />
                  <h3 className="text-sm font-bold text-white">Full Video Bounding Box Generator</h3>
                </div>
                <p className="text-xs text-slate-400 max-w-md">
                  Accelerated server pipeline with OpenCV priority bounding boxes & MP4 video download.
                </p>
              </div>

              {onRunServerVideoProcess && (
                <button
                  onClick={onRunServerVideoProcess}
                  disabled={isServerProcessingVideo}
                  className="px-5 py-3 rounded-xl font-bold text-xs text-white bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 disabled:opacity-50 shadow-lg shadow-indigo-500/20 flex items-center gap-2.5 transition-all transform active:scale-95"
                >
                  <FileVideo className="w-4 h-4 text-white" />
                  <span>Generate Full Bounding Box Video on Server</span>
                </button>
              )}
            </div>

            {/* Server-Processed Video Result View */}
            {videoResult ? (
              <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 shadow-xl backdrop-blur-md space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-bold text-white">Server-Processed Annotated Video</h3>
                    <p className="text-xs text-slate-400">Total Frames: {videoResult.total_frames} • FPS: {videoResult.fps} • Active: {videoResult.active_models.join(", ")}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    {videoResult.annotated_video_url && (
                      <a
                        href={`${apiHost}${videoResult.annotated_video_url}`}
                        download
                        className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg bg-indigo-600/30 text-indigo-300 border border-indigo-500/40 hover:bg-indigo-600/50 transition-all"
                      >
                        <Download className="w-4 h-4" /> Download MP4
                      </a>
                    )}
                    {getPriorityBadge(videoResult.highest_priority)}
                  </div>
                </div>

                {videoResult.annotated_video_url ? (
                  <div className="relative rounded-xl overflow-hidden bg-black border border-slate-800 min-h-[380px]">
                    <video
                      src={`${apiHost}${videoResult.annotated_video_url}`}
                      controls
                      autoPlay
                      loop
                      playsInline
                      className="w-full max-h-[460px] object-contain mx-auto"
                    />
                  </div>
                ) : videoResult.timeline_alerts.length > 0 && (
                  <div className="space-y-3">
                    <div className="relative rounded-xl overflow-hidden bg-black border border-slate-800 min-h-[380px] flex items-center justify-center">
                      <img
                        src={videoResult.timeline_alerts[selectedTimelineIdx]?.annotated_frame}
                        alt="Server processed frame"
                        className="max-h-[460px] w-full object-contain"
                      />
                    </div>

                    <div className="p-3 bg-slate-950/80 border border-slate-800 rounded-xl space-y-2">
                      <div className="flex items-center justify-between text-xs text-slate-300">
                        <span>Timeline Slider (Frame {videoResult.timeline_alerts[selectedTimelineIdx]?.frame_idx})</span>
                        <span>Timestamp: {videoResult.timeline_alerts[selectedTimelineIdx]?.timestamp}s</span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={videoResult.timeline_alerts.length - 1}
                        value={selectedTimelineIdx}
                        onChange={(e) => setSelectedTimelineIdx(Number(e.target.value))}
                        className="w-full accent-indigo-500 cursor-pointer"
                      />
                    </div>
                  </div>
                )}
              </div>
            ) : (
              /* Live Streaming Player View */
              <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 shadow-xl backdrop-blur-md space-y-3">
                <div className="flex items-center justify-between px-2">
                  <div className="flex items-center gap-2">
                    <Radio className={`w-4 h-4 ${wsStatus === "connected" ? "text-emerald-400 animate-pulse" : "text-amber-400"}`} />
                    <h3 className="text-sm font-bold text-white">
                      Live Video Stream ({wsStatus.toUpperCase()})
                    </h3>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setShowOverlay(!showOverlay)}
                      className="flex items-center gap-1 px-2.5 py-1 text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg border border-slate-700"
                    >
                      <Eye className="w-3.5 h-3.5 text-indigo-400" />
                      <span>{showOverlay ? "YOLO BBox Overlay ON" : "Overlay OFF"}</span>
                    </button>
                    {getPriorityBadge(currentPriority)}
                  </div>
                </div>

                {wsErrorDetail && (
                  <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-300 text-xs flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4 shrink-0" />
                      <span>{wsErrorDetail}</span>
                    </div>
                    <button
                      onClick={connectWebSocket}
                      className="px-2.5 py-1 text-xs bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 border border-rose-500/40 rounded-lg shrink-0"
                    >
                      Retry Connection
                    </button>
                  </div>
                )}

                {/* Visible Video Player + Real-Time YOLO Overlay */}
                <div className="relative rounded-xl overflow-hidden bg-black border border-slate-800 min-h-[380px] flex items-center justify-center">
                  <video
                    ref={videoRef}
                    playsInline
                    muted
                    loop
                    controls
                    className="w-full max-h-[460px] object-contain block"
                  />

                  {showOverlay && liveAnnotatedFrame && (
                    <img
                      src={liveAnnotatedFrame}
                      alt="Live YOLO WebSocket stream overlay"
                      className="absolute inset-0 w-full h-full object-contain pointer-events-none z-10 opacity-90"
                    />
                  )}
                </div>

                <div className="flex items-center justify-between gap-4">
                  <button
                    onClick={toggleVideoPlay}
                    className="flex-1 py-3 px-6 rounded-xl font-bold text-white bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 shadow-lg shadow-indigo-600/20 flex items-center justify-center gap-2 transition-all"
                  >
                    {isPlaying ? (
                      <>
                        <Pause className="w-5 h-5" /> Pause Video Stream
                      </>
                    ) : (
                      <>
                        <Play className="w-5 h-5 fill-current" /> Play Video & Start Safety Stream
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}
          </div>

          <div className="space-y-6">
            {/* Priority-Ordered Alert Feed Log */}
            <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl backdrop-blur-md">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <List className="w-4 h-4 text-indigo-400" /> Prioritized Hazard Feed Log
                </h3>

                {/* Priority Filter dropdown */}
                <div className="flex items-center gap-1 bg-slate-950 border border-slate-800 rounded-lg p-1">
                  <Filter className="w-3.5 h-3.5 text-slate-400 ml-1" />
                  <select
                    value={priorityFilter}
                    onChange={(e) => setPriorityFilter(e.target.value)}
                    className="bg-transparent text-xs text-slate-200 outline-none pr-2 cursor-pointer font-semibold"
                  >
                    <option value="ALL" className="bg-slate-900 text-white">All Priorities</option>
                    <option value="ANOMALY" className="bg-slate-900 text-red-400">P1: Critical</option>
                    <option value="POTHOLE" className="bg-slate-900 text-orange-400">P2: High</option>
                    <option value="LANE_LINE" className="bg-slate-900 text-cyan-400">P3: Medium</option>
                    <option value="ROAD_SIGN" className="bg-slate-900 text-yellow-400">P4: Low</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 mb-4">
                <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-xl">
                  <span className="text-xs text-slate-400 block">Evaluated Frames</span>
                  <span className="text-xl font-bold text-white">{videoResult ? videoResult.total_frames : stats.totalFrames}</span>
                </div>
                <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-xl">
                  <span className="text-xs text-slate-400 block">Active Hazard Alerts</span>
                  <span className="text-xl font-bold text-indigo-400">{videoResult ? videoResult.timeline_alerts.length : stats.totalAlerts}</span>
                </div>
              </div>

              <div className="space-y-2.5 max-h-[380px] overflow-y-auto pr-1">
                {videoResult ? (
                  videoResult.timeline_alerts.map((item, idx) => (
                    <div
                      key={idx}
                      onClick={() => setSelectedTimelineIdx(idx)}
                      className={`p-3 rounded-xl border cursor-pointer transition-all ${selectedTimelineIdx === idx
                        ? "bg-indigo-600/20 border-indigo-500/60 shadow-md"
                        : "bg-slate-950/70 border-slate-800 hover:border-slate-700"
                        }`}
                    >
                      <div className="flex items-center justify-between text-[11px] mb-1">
                        <span className="text-slate-400 font-mono">Frame {item.frame_idx} ({item.timestamp}s)</span>
                        {getPriorityBadge(item.highest_priority)}
                      </div>
                      {item.detections.map((d, i) => (
                        <div key={i} className="text-xs text-white flex items-center justify-between pt-1">
                          <span className="font-semibold">{d.class_name}</span>
                          <span className="text-indigo-300 font-mono text-[11px]">{(d.confidence * 100).toFixed(0)}%</span>
                        </div>
                      ))}
                    </div>
                  ))
                ) : filteredAlerts.length === 0 ? (
                  <p className="text-xs text-slate-400 italic text-center py-8">
                    Stream video or run server video analysis to populate live priority feed...
                  </p>
                ) : (
                  filteredAlerts.map((alert) => (
                    <div
                      key={alert.id}
                      className="p-3 bg-slate-950/70 border border-slate-800 rounded-xl space-y-1.5"
                    >
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-slate-400 font-mono">{alert.timestamp}</span>
                        {getPriorityBadge(alert.priority)}
                      </div>
                      {alert.detections.map((d, i) => (
                        <div key={i} className="text-xs text-white flex items-center justify-between pt-1">
                          <span className="font-semibold">{d.class_name}</span>
                          <span className="text-slate-400 font-mono">{(d.confidence * 100).toFixed(0)}%</span>
                        </div>
                      ))}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
