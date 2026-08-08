"use client";

export interface PrimaryAlertPayload {
  alert_event_id?: string;
  rule_id?: string;
  message: string;
  audio_key?: string | null;
  audio_trigger?: boolean;
  cooldown_ms?: number;
  priority?: number;
  severity?: string;
}

class AudioAlertManager {
  private isEnabled: boolean = true;
  private audioContext: AudioContext | null = null;
  private playedEvents: Map<string, number> = new Map();
  private activeUtterances: Set<SpeechSynthesisUtterance> = new Set();
  private isUnlocked: boolean = false;

  constructor() {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("dashcam_audio_alerts_enabled");
      if (saved !== null) {
        this.isEnabled = saved === "true";
      }
    }
  }

  public getEnabled(): boolean {
    return this.isEnabled;
  }

  public setEnabled(enabled: boolean): void {
    this.isEnabled = enabled;
    if (typeof window !== "undefined") {
      localStorage.setItem("dashcam_audio_alerts_enabled", enabled ? "true" : "false");
    }
    if (enabled) {
      this.unlockAudio();
    } else {
      this.cancelSpeech();
    }
  }

  public unlockAudio(): void {
    if (typeof window === "undefined") return;

    try {
      if (!this.audioContext) {
        const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        if (AudioCtx) {
          this.audioContext = new AudioCtx();
        }
      }
      if (this.audioContext && this.audioContext.state === "suspended") {
        this.audioContext.resume();
      }

      if ("speechSynthesis" in window && !this.isUnlocked) {
        const silentUtterance = new SpeechSynthesisUtterance("");
        silentUtterance.volume = 0;
        window.speechSynthesis.speak(silentUtterance);
        this.isUnlocked = true;
      }
    } catch (err) {
      console.warn("[AudioAlertManager] Audio unlock error:", err);
    }
  }

  public playWarningTone(frequency = 880, durationMs = 150): void {
    if (!this.isEnabled || typeof window === "undefined") return;

    try {
      if (!this.audioContext) {
        const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        if (AudioCtx) {
          this.audioContext = new AudioCtx();
        }
      }

      if (!this.audioContext) return;

      if (this.audioContext.state === "suspended") {
        this.audioContext.resume();
      }

      const now = this.audioContext.currentTime;
      const osc = this.audioContext.createOscillator();
      const gain = this.audioContext.createGain();

      osc.type = "sine";
      osc.frequency.setValueAtTime(frequency, now);

      gain.gain.setValueAtTime(0.3, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + durationMs / 1000);

      osc.connect(gain);
      gain.connect(this.audioContext.destination);

      osc.start(now);
      osc.stop(now + durationMs / 1000);
    } catch (err) {
      console.warn("[AudioAlertManager] Tone synthesis error:", err);
    }
  }

  public handlePrimaryAlert(alert: PrimaryAlertPayload | null | undefined): boolean {
    if (!this.isEnabled || !alert || !alert.message) return false;
    if (alert.audio_trigger === false) return false;

    // Deduplicate by rule_id and message so repeat detections of the same ongoing hazard warn every 1 second
    const eventKey = `${alert.rule_id || "alert"}-${alert.message}`;
    const nowMs = Date.now();

    const lastPlayedMs = this.playedEvents.get(eventKey);
    const cooldown = alert.cooldown_ms || 1000;

    if (lastPlayedMs && nowMs - lastPlayedMs < cooldown) {
      return false;
    }

    this.playedEvents.set(eventKey, nowMs);
    this.cleanDeduplicationCache(nowMs);

    // Play warning tone chime
    const freq = alert.severity === "critical" ? 987.77 : 783.99; // B5 for critical, G5 for warning
    this.playWarningTone(freq, 200);

    // Spoken SpeechSynthesis output
    this.speakText(alert.message);
    return true;
  }

  public speakText(text: string): void {
    if (!this.isEnabled || typeof window === "undefined") return;

    if (!("speechSynthesis" in window)) {
      // Fallback: Web Audio tone when SpeechSynthesis is unavailable
      this.playWarningTone(600, 300);
      return;
    }

    try {
      // Cancel previous speech if new critical alert arrives
      window.speechSynthesis.cancel();

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.05;
      utterance.pitch = 1.0;
      utterance.volume = 1.0;

      // Keep reference to prevent GC speech truncation bug in Chrome/Safari
      this.activeUtterances.add(utterance);

      utterance.onend = () => {
        this.activeUtterances.delete(utterance);
      };

      utterance.onerror = (e) => {
        console.warn("[AudioAlertManager] Speech error, playing tone fallback:", e);
        this.activeUtterances.delete(utterance);
        this.playWarningTone(600, 300);
      };

      window.speechSynthesis.speak(utterance);
    } catch (err) {
      console.warn("[AudioAlertManager] SpeechSynthesis exception:", err);
      this.playWarningTone(600, 300);
    }
  }

  public cancelSpeech(): void {
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      this.activeUtterances.clear();
    }
  }

  private cleanDeduplicationCache(nowMs: number): void {
    if (this.playedEvents.size > 100) {
      for (const [key, timestamp] of this.playedEvents.entries()) {
        if (nowMs - timestamp > 60000) {
          this.playedEvents.delete(key);
        }
      }
    }
  }
}

export const audioAlertManager = new AudioAlertManager();
