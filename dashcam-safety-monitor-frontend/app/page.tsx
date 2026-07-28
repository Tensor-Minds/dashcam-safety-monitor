"use client";

import React, { useState } from "react";
import { ModelSelector } from "@/components/ModelSelector";
import { MediaUploader } from "@/components/MediaUploader";
import { Dashboard, ProcessedImageResult, ProcessedVideoResult } from "@/components/Dashboard";
import { Shield, Cpu, Zap, Activity } from "lucide-react";

export default function Home() {
  const [selectedModels, setSelectedModels] = useState<string[]>([
    "anomaly",
    "lane_line",
    "pothole",
    "road_sign"
  ]);

  const [mediaFile, setMediaFile] = useState<File | null>(null);
  const [mediaType, setMediaType] = useState<"image" | "video" | null>(null);
  const [imageResult, setImageResult] = useState<ProcessedImageResult | null>(null);
  const [videoResult, setVideoResult] = useState<ProcessedVideoResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isServerProcessingVideo, setIsServerProcessingVideo] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const getApiHost = () => {
    if (process.env.NEXT_PUBLIC_API_URL) return process.env.NEXT_PUBLIC_API_URL;
    if (typeof window !== "undefined") {
      return `http://${window.location.hostname}:8000`;
    }
    return "http://localhost:8000";
  };

  const processImageWithModels = async (file: File, models: string[]) => {
    setIsProcessing(true);
    setErrorMsg(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      // Send models or default to all if empty
      const modelsToSend = models.length > 0 ? models.join(",") : "anomaly,lane_line,pothole,road_sign";
      formData.append("models", modelsToSend);

      const apiHost = getApiHost();
      const response = await fetch(`${apiHost}/api/process-image`, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Failed to process image.");
      }

      const data: ProcessedImageResult = await response.json();
      setImageResult(data);
    } catch (err: any) {
      console.error("Image processing error:", err);
      setErrorMsg(err.message || `Could not connect to FastAPI backend server at ${getApiHost()}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleMediaSubmit = async (file: File, type: "image" | "video") => {
    setMediaFile(file);
    setMediaType(type);
    setImageResult(null);
    setVideoResult(null);
    setErrorMsg(null);

    if (type === "image") {
      await processImageWithModels(file, selectedModels);
    }
  };

  const handleSelectedModelsChange = (models: string[]) => {
    setSelectedModels(models);
    if (mediaFile && mediaType === "image") {
      processImageWithModels(mediaFile, models);
    }
  };

  const handleRunServerVideoProcess = async () => {
    if (!mediaFile || mediaType !== "video") return;
    setIsServerProcessingVideo(true);
    setErrorMsg(null);

    try {
      const formData = new FormData();
      formData.append("file", mediaFile);
      const modelsToSend = selectedModels.length > 0 ? selectedModels.join(",") : "anomaly,lane_line,pothole,road_sign";
      formData.append("models", modelsToSend);

      const apiHost = getApiHost();
      const response = await fetch(`${apiHost}/api/process-video`, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Failed to process video on server.");
      }

      const data: ProcessedVideoResult = await response.json();
      setVideoResult(data);
    } catch (err: any) {
      console.error("Server video processing error:", err);
      setErrorMsg(err.message || `Server-side video processing failed. Make sure FastAPI backend is running on ${getApiHost()}`);
    } finally {
      setIsServerProcessingVideo(false);
    }
  };

  const handleReset = () => {
    setMediaFile(null);
    setMediaType(null);
    setImageResult(null);
    setVideoResult(null);
    setErrorMsg(null);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-indigo-500 selection:text-white">
      {/* Background ambient glow effect */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-40 -left-40 w-96 h-96 bg-indigo-600/15 rounded-full blur-3xl" />
        <div className="absolute top-1/3 -right-40 w-96 h-96 bg-purple-600/15 rounded-full blur-3xl" />
      </div>

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Navigation / Header */}
        <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-800/80">
          <div className="flex items-center gap-3">
            <div className="p-3 rounded-2xl bg-gradient-to-tr from-indigo-600 to-violet-600 shadow-lg shadow-indigo-600/25">
              <Shield className="w-7 h-7 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white">
                  Dashcam Safety Monitor
                </h1>
                <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/40">
                  YOLOv8 AI
                </span>
              </div>
              <p className="text-xs sm:text-sm text-slate-400 mt-0.5">
                Real-Time Multi-Model Vision Pipeline for Hazard Detection & Emergency Audio Alerts
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs text-slate-400">
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800">
              <Cpu className="w-4 h-4 text-indigo-400" />
              <span>FastAPI + PyTorch</span>
            </div>
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800">
              <Zap className="w-4 h-4 text-amber-400" />
              <span>WebSocket & REST API</span>
            </div>
          </div>
        </header>

        {/* Global Error Banner */}
        {errorMsg && (
          <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm flex items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <Activity className="w-5 h-5 text-rose-400 shrink-0" />
              <span>{errorMsg}</span>
            </div>
            <button
              onClick={() => setErrorMsg(null)}
              className="px-3 py-1 text-xs bg-rose-500/20 hover:bg-rose-500/30 rounded-lg text-rose-200 border border-rose-500/40 shrink-0"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Main Dashboard / Setup views */}
        {!mediaFile ? (
          <div className="space-y-8">
            <ModelSelector
              selectedModels={selectedModels}
              onChange={setSelectedModels}
            />

            <MediaUploader
              onMediaSubmit={handleMediaSubmit}
              selectedModels={selectedModels}
              isProcessing={isProcessing}
            />
          </div>
        ) : (
          <Dashboard
            mediaType={mediaType}
            mediaFile={mediaFile}
            selectedModels={selectedModels}
            onSelectedModelsChange={handleSelectedModelsChange}
            imageResult={imageResult}
            videoResult={videoResult}
            onReset={handleReset}
            onRunServerVideoProcess={handleRunServerVideoProcess}
            isServerProcessingVideo={isServerProcessingVideo}
          />
        )}

        {/* Footer */}
        <footer className="pt-8 border-t border-slate-800/60 text-center text-xs text-slate-500 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p>© 2026 Dashcam Road Safety Monitoring System. Built with FastAPI, PyTorch, YOLOv8 & Next.js.</p>
          <div className="flex items-center gap-4 text-slate-400">
            <span>Road Signs</span> • <span>Potholes</span> • <span>Lane Boundaries</span> • <span>Anomalies</span>
          </div>
        </footer>
      </div>
    </div>
  );
}
