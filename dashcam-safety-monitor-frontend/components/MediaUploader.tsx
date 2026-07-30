"use client";

import React, { useState, useRef, ChangeEvent, DragEvent } from "react";
import { UploadCloud, FileImage, FileVideo, X, ArrowRight, AlertCircle } from "lucide-react";

const SUPPORTED_IMAGE_EXTENSIONS = [
  ".jpg",
  ".jpeg",
  ".png",
  ".webp",
  ".bmp",
  ".tif",
  ".tiff",
  ".gif",
  ".avif",
];

interface MediaUploaderProps {
  onMediaSubmit: (file: File, mediaType: "image" | "video") => void;
  selectedModels: string[];
  isProcessing?: boolean;
}

export const MediaUploader: React.FC<MediaUploaderProps> = ({
  onMediaSubmit,
  selectedModels,
  isProcessing = false
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [mediaType, setMediaType] = useState<"image" | "video" | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const processSelectedFile = (file: File) => {
    setErrorMsg(null);
    const type = file.type;
    const lowerName = file.name.toLowerCase();
    const isSupportedImage = SUPPORTED_IMAGE_EXTENSIONS.some((extension) =>
      lowerName.endsWith(extension)
    );
    const isHeicImage =
      type === "image/heic"
      || type === "image/heif"
      || lowerName.endsWith(".heic")
      || lowerName.endsWith(".heif");

    if (isHeicImage) {
      setErrorMsg("HEIC/HEIF photos are not supported by the image engine. Export or convert the photo to JPEG, PNG, or WebP and upload it again.");
    } else if (isSupportedImage) {
      setMediaType("image");
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    } else if (type.startsWith("video/") || file.name.endsWith(".mp4") || file.name.endsWith(".webm") || file.name.endsWith(".mov")) {
      setMediaType("video");
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    } else {
      setErrorMsg("Unsupported file format. Upload a JPEG, PNG, WebP, BMP, TIFF, GIF, or AVIF image, or an MP4, WebM, or MOV video.");
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processSelectedFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleClear = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setMediaType(null);
    setPreviewUrl(null);
    setErrorMsg(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleSubmit = () => {
    if (!selectedFile || !mediaType) return;
    if (selectedModels.length === 0) {
      setErrorMsg("Please select at least one YOLO model before running inference.");
      return;
    }
    onMediaSubmit(selectedFile, mediaType);
  };

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-md">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-wide">Dashcam Media Feed Input</h2>
          <p className="text-sm text-slate-400 mt-0.5">
            Upload single frame image snapshot or continuous dashcam video.
          </p>
        </div>
        {selectedFile && (
          <button
            onClick={handleClear}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-rose-400 bg-rose-500/10 border border-rose-500/30 rounded-lg hover:bg-rose-500/20 transition-all"
          >
            <X className="w-4 h-4" /> Remove Media
          </button>
        )}
      </div>

      {errorMsg && (
        <div className="mb-4 p-3.5 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-300 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {!selectedFile ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200 ${isDragOver
              ? "border-indigo-500 bg-indigo-500/10 scale-[1.01]"
              : "border-slate-800 hover:border-slate-700 bg-slate-950/40 hover:bg-slate-900/50"
            }`}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff,.gif,.avif,.mp4,.webm,.mov,image/jpeg,image/png,image/webp,image/bmp,image/tiff,image/gif,image/avif,video/mp4,video/webm,video/quicktime"
            className="hidden"
          />

          <div className="mx-auto w-14 h-14 rounded-2xl bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400 mb-4 shadow-lg shadow-indigo-500/10">
            <UploadCloud className="w-7 h-7" />
          </div>

          <h3 className="text-base font-semibold text-white mb-1">
            Drag & Drop Dashcam Media Here
          </h3>
          <p className="text-xs text-slate-400 mb-4 max-w-sm mx-auto">
            Supports static images (<code className="text-indigo-300">.jpg, .png, .webp, .avif</code>) or dashcam video files (<code className="text-indigo-300">.mp4, .webm, .mov</code>).
          </p>

          <div className="flex items-center justify-center gap-3">
            <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 text-xs border border-slate-700">
              <FileImage className="w-3.5 h-3.5 text-cyan-400" /> Image Mode
            </span>
            <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 text-xs border border-slate-700">
              <FileVideo className="w-3.5 h-3.5 text-orange-400" /> Video WebSocket Stream
            </span>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="relative rounded-2xl overflow-hidden border border-slate-800 bg-black/60 group">
            {mediaType === "image" && previewUrl && (
              <img
                src={previewUrl}
                alt="Dashcam preview"
                className="w-full max-h-[320px] object-contain mx-auto"
              />
            )}

            {mediaType === "video" && previewUrl && (
              <video
                src={previewUrl}
                controlsList="nodownload"
                className="w-full max-h-[320px] object-contain mx-auto"
              />
            )}

            <div className="p-4 bg-slate-900/90 border-t border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-3">
                {mediaType === "image" ? (
                  <FileImage className="w-5 h-5 text-cyan-400" />
                ) : (
                  <FileVideo className="w-5 h-5 text-orange-400" />
                )}
                <div>
                  <p className="text-sm font-semibold text-white truncate max-w-[240px] sm:max-w-xs">
                    {selectedFile.name}
                  </p>
                  <p className="text-xs text-slate-400">
                    {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • {mediaType?.toUpperCase()}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <span className="px-2.5 py-1 text-xs font-semibold rounded-md bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                  {selectedModels.length} Model{selectedModels.length !== 1 ? "s" : ""} Active
                </span>
              </div>
            </div>
          </div>

          <button
            onClick={handleSubmit}
            disabled={isProcessing || selectedModels.length === 0}
            className="w-full py-3.5 px-6 rounded-xl font-bold text-white bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 disabled:opacity-50 shadow-lg shadow-indigo-600/25 flex items-center justify-center gap-2 transition-all transform active:scale-[0.99]"
          >
            {isProcessing ? (
              <>
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Running Safety Inference Pipeline...</span>
              </>
            ) : (
              <>
                <span>Run {mediaType === "image" ? "Image Safety Analysis" : "WebSocket Video Safety Stream"}</span>
                <ArrowRight className="w-5 h-5" />
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
};
