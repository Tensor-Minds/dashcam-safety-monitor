"use client";

import React from "react";
import { AlertTriangle, Disc, GitCommit, ShieldAlert, CheckSquare, Square, Zap } from "lucide-react";

export interface ModelConfig {
  id: string;
  name: string;
  category: string;
  description: string;
  priority: string;
  color: string;
  icon: React.ComponentType<{ className?: string }>;
}

export const AVAILABLE_MODELS: ModelConfig[] = [
  {
    id: "anomaly",
    name: "Anomaly Detection",
    category: "Critical Hazard",
    description: "Detects debris, sudden obstacles, and dangerous vehicle tailgating.",
    priority: "Priority 1 (Highest)",
    color: "bg-red-500/20 text-red-400 border-red-500/40",
    icon: ShieldAlert
  },
  {
    id: "lane_line",
    name: "Lane Lines & Departure",
    category: "Navigation Safety",
    description: "Monitors lane boundaries and flags vehicle lane drift warnings.",
    priority: "Priority 2",
    color: "bg-cyan-500/20 text-cyan-400 border-cyan-500/40",
    icon: GitCommit
  },
  {
    id: "pothole",
    name: "Pothole & Surface Damage",
    category: "Road Infrastructure",
    description: "Identifies potholes using 4-out-of-6 temporal voting in video streams.",
    priority: "Priority 3",
    color: "bg-orange-500/20 text-orange-400 border-orange-500/40",
    icon: AlertTriangle
  },
  {
    id: "road_sign",
    name: "Road Sign Recognition",
    category: "Traffic Compliance",
    description: "Detects speed limits, stop signs, warning signs, and traffic lights.",
    priority: "Priority 4",
    color: "bg-yellow-500/20 text-yellow-400 border-yellow-500/40",
    icon: Disc
  }
];

interface ModelSelectorProps {
  selectedModels: string[];
  onChange: (models: string[]) => void;
  disabled?: boolean;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  selectedModels,
  onChange,
  disabled = false
}) => {
  const toggleModel = (id: string) => {
    if (disabled) return;
    if (selectedModels.includes(id)) {
      // Keep at least one model selected if desired, or allow empty
      onChange(selectedModels.filter((m) => m !== id));
    } else {
      onChange([...selectedModels, id]);
    }
  };

  const selectAll = () => {
    if (disabled) return;
    onChange(AVAILABLE_MODELS.map((m) => m.id));
  };

  const clearAll = () => {
    if (disabled) return;
    onChange([]);
  };

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-md">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
        <div>
          <div className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-indigo-400" />
            <h2 className="text-xl font-bold text-white tracking-wide">YOLOv8 Parallel Model Pipeline</h2>
          </div>
          <p className="text-sm text-slate-400 mt-1">
            Select active deep learning detection models for inference execution.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={selectAll}
            disabled={disabled || selectedModels.length === AVAILABLE_MODELS.length}
            className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-indigo-600/30 text-indigo-300 border border-indigo-500/40 hover:bg-indigo-600/50 disabled:opacity-40 transition-all"
          >
            Select All
          </button>
          <button
            type="button"
            onClick={clearAll}
            disabled={disabled || selectedModels.length === 0}
            className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-slate-800 text-slate-400 border border-slate-700 hover:bg-slate-700 disabled:opacity-40 transition-all"
          >
            Clear All
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {AVAILABLE_MODELS.map((model) => {
          const isSelected = selectedModels.includes(model.id);
          const IconComponent = model.icon;

          return (
            <div
              key={model.id}
              onClick={() => toggleModel(model.id)}
              className={`relative cursor-pointer rounded-xl p-4 border transition-all duration-200 ${
                isSelected
                  ? "bg-slate-800/90 border-indigo-500/60 shadow-lg shadow-indigo-500/10 ring-1 ring-indigo-500/50"
                  : "bg-slate-900/40 border-slate-800 hover:border-slate-700 opacity-70 hover:opacity-90"
              } ${disabled ? "pointer-events-none opacity-50" : ""}`}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-center gap-3">
                  <div className={`p-2.5 rounded-lg border ${model.color}`}>
                    <IconComponent className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-white text-base">{model.name}</h3>
                    <span className="text-xs text-slate-400">{model.category}</span>
                  </div>
                </div>

                <div className="text-indigo-400">
                  {isSelected ? (
                    <CheckSquare className="w-6 h-6 text-indigo-400" />
                  ) : (
                    <Square className="w-6 h-6 text-slate-600" />
                  )}
                </div>
              </div>

              <p className="text-xs text-slate-300 mt-3 leading-relaxed">
                {model.description}
              </p>

              <div className="mt-3 flex items-center justify-between text-[11px] text-slate-400 border-t border-slate-800/80 pt-2.5">
                <span className="font-mono text-slate-400">Model ID: {model.id}</span>
                <span className={`px-2 py-0.5 rounded border text-[10px] font-semibold ${model.color}`}>
                  {model.priority}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
