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
  Flame,
  FileVideo,
  Download,
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
  primary_alert?: PrimaryAlert | null;
  image_quality?: { usable: boolean; message?: string | null; blur_score: number };
  annotated_image: string;
}

export interface PrimaryAlert {
  rule_id: string;
  category: "anomaly" | "lane_line" | "pothole" | "road_sign";
  message: string;
  priority: number;
  audio_key: "anomaly" | "lane_departure" | "pothole" | "road_sign" | null;
  audio_trigger: boolean;
  timestamp_ms: number;
  visible_until_ms: number;
  evidence: Record<string, unknown>;
}

interface ConfiguredRule {
  id: string;
  enabled: boolean;
  module: string;
  labels: string[];
  priority: number;
  message: string;
  temporal: {
    window_frames?: number;
    minimum_hits?: number;
    minimum_duration_ms?: number;
    maximum_misses?: number;
  } | null;
}

interface RuleConfigurationData extends Record<string, unknown> {
  rules: ConfiguredRule[];
}

const RULE_MODEL_GROUPS = [
  { id: "anomaly", label: "Road anomaly" },
  { id: "lane_line", label: "Lane departure" },
  { id: "pothole", label: "Road damage / pothole" },
  { id: "road_sign", label: "Road signs" },
];

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
    primary_alert?: PrimaryAlert | null;
    image_quality?: { usable: boolean; message?: string | null; blur_score: number };
    annotated_frame: string;
  }[];
}

const DASHBOARD_MODELS = [
  { id: "anomaly", label: "P1: Anomaly", activeClass: "bg-red-500/20 text-red-400 border-red-500/50 shadow-sm shadow-red-500/20" },
  { id: "lane_line", label: "P2: Lane Lines", activeClass: "bg-cyan-500/20 text-cyan-400 border-cyan-500/50" },
  { id: "pothole", label: "P3: Potholes", activeClass: "bg-orange-500/20 text-orange-400 border-orange-500/50" },
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
  onRunServerVideoProcess?: (
    turnSignal: "off" | "left" | "right",
    simulatedSpeedKmh: number
  ) => void;
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
  const [audioWarningsEnabled, setAudioWarningsEnabled] = useState(true);
  const [turnSignal, setTurnSignal] = useState<"off" | "left" | "right">("off");
  const [simulatedSpeedKmh, setSimulatedSpeedKmh] = useState(50);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioWarningsEnabledRef = useRef(true);
  const speechUtteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
  const audioUnlockConfirmedRef = useRef(false);
  const lastAudioTriggerRef = useRef<number>(0);
  const lastAudioPriorityRef = useRef<number>(0);

  // Video Streaming State
  const [isPlaying, setIsPlaying] = useState(false);
  const [wsStatus, setWsStatus] = useState<"disconnected" | "connecting" | "connected" | "error">("disconnected");
  const [wsErrorDetail, setWsErrorDetail] = useState<string | null>(null);
  const [liveDetections, setLiveDetections] = useState<Detection[]>([]);
  const [liveFrameSize, setLiveFrameSize] = useState({ width: 640, height: 360 });

  // Alert feed state
  const [alertsFeed, setAlertsFeed] = useState<
    { id: string; timestamp: string; priority: string; detections: Detection[] }[]
  >([]);
  const [currentPriority, setCurrentPriority] = useState<string>("normal");
  const [currentPrimaryAlert, setCurrentPrimaryAlert] = useState<PrimaryAlert | null>(null);
  const [configuredRules, setConfiguredRules] = useState<ConfiguredRule[]>([]);
  const [ruleConfiguration, setRuleConfiguration] = useState<RuleConfigurationData | null>(null);
  const [isSavingRules, setIsSavingRules] = useState(false);
  const [ruleSaveStatus, setRuleSaveStatus] = useState<string | null>(null);
  const [stats, setStats] = useState({ totalFrames: 0, totalAlerts: 0 });

  // Processing Timer Counter for Server Video processing
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  const lastAlertFeedUpdateRef = useRef<number>(0);
  const isPlayingRef = useRef<boolean>(false);
  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

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
      interval = setInterval(() => {
        setElapsedSeconds((prev) => prev + 1);
      }, 1000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isServerProcessingVideo]);

  useEffect(() => {
    return () => {
      audioContextRef.current?.close();
      audioContextRef.current = null;
      window.speechSynthesis?.cancel();
      speechUtteranceRef.current = null;
    };
  }, []);

  const ensureAudioReady = () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext();
    }
    if (audioContextRef.current.state === "suspended") {
      void audioContextRef.current.resume();
    }
  };

  const speakWarning = (message: string) => {
    if (
      !audioWarningsEnabledRef.current
      || !("speechSynthesis" in window)
      || !("SpeechSynthesisUtterance" in window)
    ) {
      return;
    }

    const synthesizer = window.speechSynthesis;
    synthesizer.cancel();
    synthesizer.resume();

    const utterance = new SpeechSynthesisUtterance(message);
    const preferredVoice = synthesizer
      .getVoices()
      .find((voice) => voice.lang.toLowerCase().startsWith("en"));
    if (preferredVoice) utterance.voice = preferredVoice;
    utterance.lang = preferredVoice?.lang || "en-US";
    utterance.rate = 0.95;
    utterance.pitch = 1;
    utterance.volume = 1;
    utterance.onend = () => {
      speechUtteranceRef.current = null;
    };
    utterance.onerror = () => {
      speechUtteranceRef.current = null;
    };
    speechUtteranceRef.current = utterance;
    synthesizer.speak(utterance);
  };

  const playAlertSound = (
    audioKey: string = "road_sign",
    spokenMessage?: string,
    priority: number = 0
  ) => {
    if (!audioWarningsEnabledRef.current) return;
    ensureAudioReady();
    const now = Date.now();
    const higherPriorityInterrupt = priority > lastAudioPriorityRef.current;
    if (now - lastAudioTriggerRef.current > 5000 || higherPriorityInterrupt) {
      lastAudioTriggerRef.current = now;
      lastAudioPriorityRef.current = priority;
      const audio = audioContextRef.current;
      const frequencies =
        audioKey === "anomaly" ? [880, 1040, 880] :
          audioKey === "lane_departure" ? [330, 330] :
            audioKey === "pothole" ? [560] : [720];
      if (audio) {
        frequencies.forEach((frequency, index) => {
          const oscillator = audio.createOscillator();
          const gain = audio.createGain();
          const start = audio.currentTime + index * 0.16;
          oscillator.type = audioKey === "anomaly" ? "square" : "sine";
          oscillator.frequency.value = frequency;
          gain.gain.setValueAtTime(0.12, start);
          gain.gain.exponentialRampToValueAtTime(0.001, start + 0.13);
          oscillator.connect(gain).connect(audio.destination);
          oscillator.start(start);
          oscillator.stop(start + 0.14);
        });
      }
      if (spokenMessage) speakWarning(spokenMessage);
    }
  };

  const toggleAudioWarnings = () => {
    const enabled = !audioWarningsEnabledRef.current;
    audioWarningsEnabledRef.current = enabled;
    setAudioWarningsEnabled(enabled);

    if (enabled) {
      lastAudioTriggerRef.current = 0;
      lastAudioPriorityRef.current = 0;
      ensureAudioReady();
      const activeAlert = currentPrimaryAlert || imageResult?.primary_alert;
      if (activeAlert) {
        playAlertSound(
          activeAlert.audio_key || activeAlert.category,
          activeAlert.message,
          activeAlert.priority
        );
      } else {
        playAlertSound("road_sign", "Audio warnings enabled");
      }
      audioUnlockConfirmedRef.current = true;
    } else {
      window.speechSynthesis?.cancel();
      speechUtteranceRef.current = null;
      audioUnlockConfirmedRef.current = false;
      void audioContextRef.current?.suspend();
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
      playAlertSound(
        imageResult.primary_alert?.audio_key || imageResult.highest_priority,
        imageResult.primary_alert?.message,
        imageResult.primary_alert?.priority || 0
      );
    }
  }, [imageResult, mediaType]);

  // Connect WebSocket helper function
  const connectWebSocket = () => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const wsProtocol =
      typeof window !== "undefined" && window.location.protocol === "https:"
        ? "wss:"
        : "ws:";
    const wsHost =
      typeof window !== "undefined" ? window.location.hostname : "localhost";
    const wsUrl =
      process.env.NEXT_PUBLIC_WS_URL ||
      `${wsProtocol}//${wsHost}:8000/ws/video`;
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
            setLiveDetections(data.detections || []);
            setCurrentPriority(data.highest_priority);
            setCurrentPrimaryAlert(data.primary_alert || null);
            setStats((prev) => ({
              totalFrames: prev.totalFrames + 1,
              totalAlerts: prev.totalAlerts + (data.detections.length > 0 ? 1 : 0)
            }));

            if (data.audio_trigger) {
              playAlertSound(
                data.primary_alert?.audio_key || data.highest_priority,
                data.primary_alert?.message,
                data.primary_alert?.priority || 0
              );
            }

            const now = Date.now();
            if (data.detections && data.detections.length > 0 && now - lastAlertFeedUpdateRef.current > 350) {
              lastAlertFeedUpdateRef.current = now;
              const newAlert = {
                id: `${now}-${Math.random()}`,
                timestamp: new Date().toLocaleTimeString(),
                priority: data.highest_priority,
                detections: data.detections
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
        setWsErrorDetail(`WebSocket server unreachable at ${wsUrl}. Ensure backend is running on port 8000.`);
      };

      ws.onclose = () => {
        setWsStatus((prev) => (prev === "error" ? "error" : "disconnected"));
        if (isPlayingRef.current) {
          setTimeout(connectWebSocket, 500);
        }
      };
    } catch (err: unknown) {
      setWsStatus("error");
      setWsErrorDetail(err instanceof Error ? err.message : "Failed to initialize WebSocket connection.");
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

  // Frame capture loop over WebSocket (target ~10 fps, matching the report).
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
      const captureScale = Math.min(
        1,
        1280 / video.videoWidth,
        720 / video.videoHeight
      );
      canvas.width = Math.max(1, Math.round(video.videoWidth * captureScale));
      canvas.height = Math.max(1, Math.round(video.videoHeight * captureScale));
      setLiveFrameSize((current) =>
        current.width === canvas.width && current.height === canvas.height
          ? current
          : { width: canvas.width, height: canvas.height }
      );
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const frame_b64 = canvas.toDataURL("image/jpeg", 0.75);

      const payload = {
        command: "process_frame",
        active_models: selectedModels,
        frame_idx: video.currentTime,
        timestamp: video.currentTime,
        turn_signal: turnSignal,
        simulated_vehicle_speed_kmh: simulatedSpeedKmh,
        include_annotated_frame: false,
        frame_b64: frame_b64
      };

      wsRef.current.send(JSON.stringify(payload));
    }

    setTimeout(() => {
      if (isPlayingRef.current) {
        animationFrameRef.current = requestAnimationFrame(streamFrameLoop);
      }
    }, 33);
  };

  const toggleVideoPlay = (e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    if (!videoRef.current) return;
    ensureAudioReady();
    if (
      audioWarningsEnabledRef.current
      && !audioUnlockConfirmedRef.current
    ) {
      speakWarning("Audio warnings enabled");
      audioUnlockConfirmedRef.current = true;
    }

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

  const handleVideoPlayEvent = () => {
    ensureAudioReady();
    setIsPlaying(true);
    if (!animationFrameRef.current) {
      animationFrameRef.current = requestAnimationFrame(streamFrameLoop);
    }
  };

  const handleVideoPauseEvent = () => {
    setIsPlaying(false);
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
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

  const runAllModels = () => {
    onSelectedModelsChange(["anomaly", "lane_line", "pothole", "road_sign"]);
  };

  const getPriorityBadge = (priority: string) => {
    switch (priority) {
      case "anomaly":
        return <span className="px-3 py-1 rounded-md text-xs font-extrabold bg-red-500/20 text-red-400 border border-red-500/50 shadow-sm shadow-red-500/20 flex items-center gap-1.5"><Flame className="w-3.5 h-3.5 text-red-400" /> PRIORITY 1: CRITICAL ANOMALY</span>;
      case "pothole":
        return <span className="px-3 py-1 rounded-md text-xs font-extrabold bg-orange-500/20 text-orange-400 border border-orange-500/50 flex items-center gap-1.5"><AlertTriangle className="w-3.5 h-3.5 text-orange-400" /> PRIORITY 3: POTHOLE HAZARD</span>;
      case "lane_line":
        return <span className="px-3 py-1 rounded-md text-xs font-extrabold bg-cyan-500/20 text-cyan-400 border border-cyan-500/50 flex items-center gap-1.5"><Activity className="w-3.5 h-3.5 text-cyan-400" /> PRIORITY 2: LANE DEPARTURE</span>;
      case "road_sign":
        return <span className="px-3 py-1 rounded-md text-xs font-extrabold bg-yellow-500/20 text-yellow-400 border border-yellow-500/50 flex items-center gap-1.5"><Layers className="w-3.5 h-3.5 text-yellow-400" /> PRIORITY 4: ROAD SIGN</span>;
      default:
        return <span className="px-3 py-1 rounded-md text-xs font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40">NORMAL DRIVING</span>;
    }
  };

  const apiProtocol =
    typeof window !== "undefined" ? window.location.protocol : "http:";
  const apiHostname =
    typeof window !== "undefined" ? window.location.hostname : "localhost";
  const apiHost =
    process.env.NEXT_PUBLIC_API_URL ||
    `${apiProtocol}//${apiHostname}:8000`;

  useEffect(() => {
    let cancelled = false;
    fetch(`${apiHost}/api/rules`)
      .then((response) => response.ok ? response.json() : Promise.reject(new Error("Rule configuration unavailable")))
      .then((data: RuleConfigurationData) => {
        if (!cancelled) {
          setRuleConfiguration(data);
          setConfiguredRules(
            [...data.rules].sort((left, right) => right.priority - left.priority)
          );
        }
      })
      .catch(() => {
        if (!cancelled) setConfiguredRules([]);
      });
    return () => {
      cancelled = true;
    };
  }, [apiHost]);

  const updateModelRules = (
    module: string,
    update: (rule: ConfiguredRule) => ConfiguredRule
  ) => {
    if (!ruleConfiguration) return;
    const rules = ruleConfiguration.rules.map((rule) =>
      rule.module === module ? update(rule) : rule
    );
    setRuleConfiguration({ ...ruleConfiguration, rules });
    setConfiguredRules(
      [...rules].sort((left, right) => right.priority - left.priority)
    );
    setRuleSaveStatus(null);
  };

  const updateModelVoting = (
    module: string,
    field: "window_frames" | "minimum_hits" | "minimum_duration_ms",
    rawValue: number
  ) => {
    const value = Math.max(1, Math.round(rawValue || 1));
    updateModelRules(module, (rule) => {
      if (!rule.temporal) return rule;
      if (field !== "minimum_duration_ms" && !rule.temporal.window_frames) {
        return rule;
      }
      const temporal = { ...rule.temporal, [field]: value };
      if (
        temporal.window_frames
        && temporal.minimum_hits
        && temporal.minimum_hits > temporal.window_frames
      ) {
        if (field === "window_frames") {
          temporal.minimum_hits = temporal.window_frames;
        } else {
          temporal.minimum_hits = temporal.window_frames;
        }
      }
      return { ...rule, temporal };
    });
  };

  const saveRuleConfiguration = async () => {
    if (!ruleConfiguration) return;
    setIsSavingRules(true);
    setRuleSaveStatus(null);
    try {
      const response = await fetch(`${apiHost}/api/rules`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(ruleConfiguration),
      });
      if (!response.ok) {
        const error = await response.json().catch(() => null);
        throw new Error(error?.detail || "Unable to save rule configuration");
      }
      setRuleSaveStatus("Rules saved and activated");
    } catch (error) {
      setRuleSaveStatus(
        error instanceof Error ? error.message : "Unable to save rules"
      );
    } finally {
      setIsSavingRules(false);
    }
  };

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

            {/* Live processing timer */}
            <div className="p-3 bg-slate-950/70 border border-slate-800 rounded-2xl flex items-center justify-between">
              <div>
                <span className="text-[10px] text-slate-400 block font-semibold">Elapsed Time</span>
                <span className="text-lg font-extrabold text-indigo-400 font-mono">
                  {formatTimer(elapsedSeconds)}
                </span>
              </div>
              <Clock className="w-5 h-5 text-indigo-400 opacity-60" />
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
              Running multi-model pipeline: <code className="text-indigo-300 font-bold">{selectedModels.length === 0 ? "ALL 4 MODELS" : selectedModels.join(", ")}</code>
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={toggleAudioWarnings}
            className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold border transition-all ${
              audioWarningsEnabled
                ? "bg-indigo-600/20 text-indigo-300 border-indigo-500/40 hover:bg-indigo-600/30"
                : "bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700"
            }`}
          >
            {audioWarningsEnabled ? (
              <Volume2 className="w-4 h-4 text-indigo-400" />
            ) : (
              <VolumeX className="w-4 h-4" />
            )}
            <span>
              {audioWarningsEnabled
                ? "Audio warnings enabled"
                : "Enable audio warnings"}
            </span>
          </button>

          <label className="flex items-center gap-2 px-3 py-2 rounded-xl text-xs bg-slate-800 border border-slate-700 text-slate-300">
            Demo turn signal
            <select
              value={turnSignal}
              onChange={(event) => setTurnSignal(event.target.value as "off" | "left" | "right")}
              className="bg-slate-900 rounded px-2 py-1"
            >
              <option value="off">Off</option>
              <option value="left">Left</option>
              <option value="right">Right</option>
            </select>
          </label>

          <label className="flex items-center gap-2 px-3 py-2 rounded-xl text-xs bg-slate-800 border border-slate-700 text-slate-300">
            Demo speed
            <input
              type="number"
              min={0}
              max={300}
              value={simulatedSpeedKmh}
              onChange={(event) =>
                setSimulatedSpeedKmh(
                  Math.min(300, Math.max(0, Number(event.target.value) || 0))
                )
              }
              className="w-16 bg-slate-900 rounded px-2 py-1"
            />
            km/h
          </label>

          <button
            onClick={onReset}
            className="flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold bg-slate-800 text-slate-300 border border-slate-700 hover:bg-slate-700 transition-all"
          >
            <RefreshCw className="w-4 h-4" /> Reset Workspace
          </button>
        </div>
      </div>

      {currentPrimaryAlert && (
        <div className="rounded-2xl border border-red-500/50 bg-red-500/15 px-5 py-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-[10px] uppercase tracking-widest text-red-300">
              Highest-priority active rule
            </p>
            <p className="font-bold text-white text-lg">{currentPrimaryAlert.message}</p>
          </div>
          <div className="text-right text-xs text-slate-300">
            <p>Rule: <strong>{currentPrimaryAlert.rule_id}</strong></p>
            <p>Numeric priority: <strong>{currentPrimaryAlert.priority}</strong></p>
          </div>
        </div>
      )}

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
            onClick={runAllModels}
            className="px-3.5 py-1.5 rounded-xl text-xs font-extrabold bg-gradient-to-r from-violet-600 to-indigo-600 text-white shadow-md hover:from-violet-500 hover:to-indigo-500 transition-all ml-1"
          >
            ⚡ Run All Models
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
                  onClick={() =>
                    onRunServerVideoProcess(turnSignal, simulatedSpeedKmh)
                  }
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

                  {getPriorityBadge(currentPriority)}
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
                    onPlay={handleVideoPlayEvent}
                    onPause={handleVideoPauseEvent}
                    className="w-full max-h-[460px] object-contain block"
                  />

                  {liveDetections.length > 0 && (
                    <svg
                      viewBox={`0 0 ${liveFrameSize.width} ${liveFrameSize.height}`}
                      preserveAspectRatio="xMidYMid meet"
                      aria-label="Live detection overlay"
                      className="absolute inset-0 w-full h-full pointer-events-none z-10"
                    >
                      {liveDetections.map((detection, index) => {
                        if (!detection || !detection.bbox || detection.bbox.length < 4) return null;
                        const [x1, y1, x2, y2] = detection.bbox;
                        const rawColor = Array.isArray(detection.color) && detection.color.length === 3 ? detection.color : [0, 255, 0];
                        const colour = `rgb(${rawColor[2]}, ${rawColor[1]}, ${rawColor[0]})`;
                        const label = `${detection.class_name || "Hazard"} ${((detection.confidence || 0) * 100).toFixed(0)}%`;
                        return (
                          <g key={`${detection.category || 'det'}-${index}`}>
                            <rect
                              x={x1}
                              y={y1}
                              width={Math.max(0, x2 - x1)}
                              height={Math.max(0, y2 - y1)}
                              fill="transparent"
                              stroke={colour}
                              strokeWidth={detection.priority_level === "CRITICAL" ? 3 : 2}
                              vectorEffect="non-scaling-stroke"
                            />
                            <text
                              x={x1 + 3}
                              y={Math.max(14, y1 - 5)}
                              fill={colour}
                              stroke="black"
                              strokeWidth="0.8"
                              paintOrder="stroke"
                              fontSize="12"
                              fontWeight="700"
                            >
                              {label}
                            </text>
                          </g>
                        );
                      })}
                    </svg>
                  )}
                </div>

                {/* Single Primary Control Action Button */}
                <div className="flex items-center justify-between gap-4">
                  <button
                    onClick={(e) => toggleVideoPlay(e)}
                    className="flex-1 py-3.5 px-6 rounded-xl font-bold text-white bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 disabled:opacity-40 shadow-lg shadow-indigo-600/25 flex items-center justify-center gap-2.5 transition-all transform active:scale-[0.99]"
                  >
                    {isPlaying ? (
                      <>
                        <Pause className="w-5 h-5 fill-current" /> Pause Live Video Stream
                      </>
                    ) : (
                      <>
                        <Play className="w-5 h-5 fill-current" /> Start Live Video Safety Stream
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
                ) : alertsFeed.length === 0 ? (
                  <p className="text-xs text-slate-400 italic text-center py-8">
                    Stream video or run server video analysis to populate live priority feed...
                  </p>
                ) : (
                  alertsFeed.map((alert) => (
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

      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl">
        <div className="flex items-center justify-between gap-3 mb-4">
          <div>
            <h3 className="text-sm font-bold text-white">Configured Alert Rules</h3>
            <p className="text-xs text-slate-400">
              Enable each model and adjust its temporal confirmation before saving to rules.yml.
            </p>
          </div>
          <div className="flex items-center gap-3">
            {ruleSaveStatus && (
              <span className="text-xs text-indigo-300">{ruleSaveStatus}</span>
            )}
            <button
              type="button"
              disabled={!ruleConfiguration || isSavingRules}
              onClick={saveRuleConfiguration}
              className="rounded-lg border border-indigo-500/40 bg-indigo-600/20 px-3 py-1.5 text-xs font-semibold text-indigo-200 hover:bg-indigo-600/30 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isSavingRules ? "Saving..." : "Save rule settings"}
            </button>
            <span className="text-xs text-indigo-300">{configuredRules.length} rules</span>
          </div>
        </div>

        <div className="grid md:grid-cols-2 xl:grid-cols-4 gap-3 mb-5">
          {RULE_MODEL_GROUPS.map((model) => {
            const rules = ruleConfiguration?.rules.filter(
              (rule) => rule.module === model.id
            ) || [];
            const votingRule = rules.find(
              (rule) =>
                rule.temporal?.window_frames !== undefined
                && rule.temporal?.minimum_hits !== undefined
            );
            const durationRule = rules.find(
              (rule) =>
                rule.temporal?.minimum_duration_ms !== undefined
                && rule.temporal?.window_frames === undefined
            );
            const enabledCount = rules.filter((rule) => rule.enabled).length;
            const modelEnabled = enabledCount > 0;

            return (
              <div
                key={model.id}
                className="rounded-xl border border-slate-700 bg-slate-950/70 p-3"
              >
                <label className="flex items-center justify-between gap-3 text-xs font-semibold text-white">
                  <span>{model.label}</span>
                  <input
                    type="checkbox"
                    checked={modelEnabled}
                    disabled={rules.length === 0}
                    onChange={(event) =>
                      updateModelRules(model.id, (rule) => ({
                        ...rule,
                        enabled: event.target.checked,
                      }))
                    }
                    className="h-4 w-4 accent-indigo-500"
                  />
                </label>
                <p className="mt-1 text-[10px] text-slate-500">
                  {enabledCount}/{rules.length} rules enabled
                </p>

                {votingRule?.temporal && (
                  <div className="mt-3 grid grid-cols-2 gap-2">
                    <label className="text-[10px] text-slate-400">
                      Required hits
                      <input
                        type="number"
                        min={1}
                        max={votingRule.temporal.window_frames}
                        value={votingRule.temporal.minimum_hits}
                        onChange={(event) =>
                          updateModelVoting(
                            model.id,
                            "minimum_hits",
                            Number(event.target.value)
                          )
                        }
                        className="mt-1 w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-white"
                      />
                    </label>
                    <label className="text-[10px] text-slate-400">
                      Window frames
                      <input
                        type="number"
                        min={1}
                        max={120}
                        value={votingRule.temporal.window_frames}
                        onChange={(event) =>
                          updateModelVoting(
                            model.id,
                            "window_frames",
                            Number(event.target.value)
                          )
                        }
                        className="mt-1 w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-white"
                      />
                    </label>
                  </div>
                )}

                {durationRule?.temporal && (
                  <label className="mt-3 block text-[10px] text-slate-400">
                    Confirmation duration (ms)
                    <input
                      type="number"
                      min={1}
                      max={60000}
                      step={100}
                      value={durationRule.temporal.minimum_duration_ms}
                      onChange={(event) =>
                        updateModelVoting(
                          model.id,
                          "minimum_duration_ms",
                          Number(event.target.value)
                        )
                      }
                      className="mt-1 w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-white"
                    />
                  </label>
                )}
              </div>
            );
          })}
        </div>

        <div className="grid sm:grid-cols-2 xl:grid-cols-3 gap-2">
          {configuredRules.map((rule) => (
            <div key={rule.id} className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
              <div className="flex items-center justify-between gap-2">
                <code className="text-xs text-indigo-300">{rule.id}</code>
                <span className="rounded bg-indigo-500/20 border border-indigo-500/30 px-2 py-0.5 text-xs font-bold text-indigo-200">
                  Priority {rule.priority}
                </span>
              </div>
              <p className="text-xs text-slate-300 mt-2">{rule.message}</p>
              <p className="text-[10px] text-slate-500 mt-1">
                {rule.module} · {rule.labels.join(", ")} · {rule.enabled ? "enabled" : "disabled"}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
