"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { encodeWav } from "@/lib/wav-encoder";
import { decodeAudioFileToWav, needsConversion } from "@/lib/audio-utils";
import {
  EMOTIONS,
  EMOTION_COLORS,
  type PredictResult,
} from "@/lib/emotions";

// ── Colormap for embedding visualization ─────────────────────────────────────

function embedColor(t: number): string {
  const stops: [number, number, number, number][] = [
    [0.0, 0x1e, 0x1b, 0x4b],
    [0.33, 0xa7, 0x8b, 0xfa],
    [0.66, 0xec, 0x48, 0x99],
    [1.0, 0xfb, 0xbf, 0x24],
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    const [p0, r0, g0, b0] = stops[i];
    const [p1, r1, g1, b1] = stops[i + 1];
    if (t >= p0 && t <= p1) {
      const f = (t - p0) / (p1 - p0);
      const r = Math.round(r0 + (r1 - r0) * f);
      const g = Math.round(g0 + (g1 - g0) * f);
      const b = Math.round(b0 + (b1 - b0) * f);
      return `rgb(${r},${g},${b})`;
    }
  }
  return "rgb(251,191,36)";
}

// ── Magma colormap (simplified) for spectrogram ──────────────────────────────

function magmaColor(t: number): string {
  const stops: [number, number, number, number][] = [
    [0.0, 0x00, 0x00, 0x04],
    [0.25, 0x3b, 0x0f, 0x70],
    [0.5, 0xbb, 0x36, 0x5c],
    [0.75, 0xf3, 0x84, 0x5c],
    [1.0, 0xfc, 0xf3, 0x9d],
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    const [p0, r0, g0, b0] = stops[i];
    const [p1, r1, g1, b1] = stops[i + 1];
    if (t >= p0 && t <= p1) {
      const f = (t - p0) / (p1 - p0);
      const r = Math.round(r0 + (r1 - r0) * f);
      const g = Math.round(g0 + (g1 - g0) * f);
      const b = Math.round(b0 + (b1 - b0) * f);
      return `rgb(${r},${g},${b})`;
    }
  }
  return "rgb(252,243,157)";
}

// ── Waveform colors ──────────────────────────────────────────────────────────

const WAVEFORM_PRIMARY = "rgba(167,139,250,0.5)";
const WAVEFORM_SECONDARY = "rgba(236,72,153,0.2)";

// ── Status dot ───────────────────────────────────────────────────────────────

function StatusDot({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2.5 text-sm text-text-muted">
      <span className="status-dot" />
      {label}
    </div>
  );
}

// ── Metric Card ──────────────────────────────────────────────────────────────

function MetricCard({
  label,
  value,
  className = "",
}: {
  label: string;
  value: string | number;
  className?: string;
}) {
  return (
    <div className={`metric-card ${className}`}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
    </div>
  );
}

// ── Confidence Ring ──────────────────────────────────────────────────────────

function ConfidenceRing({ confidence }: { confidence: number }) {
  const radius = 28;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - confidence);
  const color =
    confidence > 0.7
      ? "#34D399"
      : confidence > 0.4
        ? "#A78BFA"
        : "#F87171";

  return (
    <div className="confidence-ring flex flex-col items-center gap-1">
      <svg width="72" height="72" className="drop-shadow-lg">
        <circle className="ring-bg" cx="36" cy="36" r={radius} />
        <circle
          className="ring-fg"
          cx="36"
          cy="36"
          r={radius}
          stroke={color}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
      </svg>
      <span
        className="text-lg font-bold tabular-nums"
        style={{ color }}
      >
        {(confidence * 100).toFixed(0)}%
      </span>
      <span className="text-[10px] text-text-dim uppercase tracking-wider mt-[-2px]">
        confident
      </span>
    </div>
  );
}

// ── Emotion Bar ──────────────────────────────────────────────────────────────

function EmotionBar({
  label,
  prob,
  color,
  isActive,
  delay,
}: {
  label: string;
  prob: number;
  color: string;
  isActive: boolean;
  delay: number;
}) {
  const [animate, setAnimate] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setAnimate(true), delay);
    return () => clearTimeout(t);
  }, [delay]);

  return (
    <div className="emotion-row group">
      <div className="emotion-label">{label}</div>
      <div className="emotion-bar-bg">
        <div
          className="emotion-bar-fill"
          style={{
            width: animate ? `${prob * 100}%` : "0%",
            backgroundColor: color,
            opacity: isActive ? 1 : 0.6 + prob * 0.4,
          }}
        />
      </div>
      <div className={`emotion-pct ${isActive ? "active" : ""}`}>
        {(prob * 100).toFixed(1)}%
      </div>
    </div>
  );
}

// ── Canvas: Embedding Sparkline ──────────────────────────────────────────────

function EmbeddingCanvas({ embedding }: { embedding: number[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    const min = Math.min(...embedding);
    const max = Math.max(...embedding);
    const range = Math.max(max - min, 1e-8);
    const barW = Math.max(1, w / embedding.length);

    for (let i = 0; i < embedding.length; i++) {
      const norm = (embedding[i] - min) / range;
      const bh = Math.max(1, norm * (h - 4));
      ctx.fillStyle = embedColor(norm);
      ctx.fillRect(i * barW, h - bh - 2, Math.ceil(barW) + 1, bh);
    }
  }, [embedding]);

  return (
    <div className="embedding-bar">
      <div className="text-xs text-text-muted uppercase tracking-wider mb-2">
        Embedding Vector ({embedding.length}d)
      </div>
      <canvas
        ref={canvasRef}
        className="w-full block rounded-md"
        style={{ height: "48px" }}
      />
    </div>
  );
}

// ── Canvas: Mel-Spectrogram ──────────────────────────────────────────────────

function SpectrogramCanvas({
  spectrogramData,
  duration,
}: {
  spectrogramData: number[][];
  duration: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setVisible(true), 250);
    return () => clearTimeout(t);
  }, []);

  useEffect(() => {
    if (!visible) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    const nMels = spectrogramData.length;
    const nFrames = spectrogramData[0]?.length ?? 1;
    const cellW = w / nFrames;
    const cellH = h / nMels;

    for (let m = 0; m < nMels; m++) {
      for (let t = 0; t < nFrames; t++) {
        ctx.fillStyle = magmaColor(spectrogramData[m][t]);
        ctx.fillRect(
          t * cellW,
          (nMels - 1 - m) * cellH,
          Math.ceil(cellW) + 1,
          Math.ceil(cellH) + 1
        );
      }
    }

    // Axis labels
    ctx.fillStyle = "#8B95A8";
    ctx.font = "10px monospace";
    ctx.fillText(`0s`, 4, h - 4);
    ctx.fillText(`${duration.toFixed(1)}s`, w - 40, h - 4);
    ctx.fillText("Mel bands", 4, 12);
    ctx.fillText("128", 4, 14 + cellH * nMels);
  }, [spectrogramData, duration, visible]);

  return (
    <div className="canvas-wrap">
      <canvas
        ref={canvasRef}
        className="w-full block"
        style={{ height: "200px" }}
      />
    </div>
  );
}

// ── Canvas: Waveform ─────────────────────────────────────────────────────────

function WaveformCanvas({
  audioUrl,
  color = WAVEFORM_PRIMARY,
}: {
  audioUrl: string;
  color?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const audioCtx = new AudioContext();
    let cancelled = false;

    fetch(audioUrl)
      .then((r) => r.arrayBuffer())
      .then((buf) => audioCtx.decodeAudioData(buf))
      .then((audioBuf) => {
        if (cancelled) return;
        const dpr = window.devicePixelRatio || 1;
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        ctx.scale(dpr, dpr);

        const samples = audioBuf.getChannelData(0);
        const step = Math.max(1, Math.floor(samples.length / w));
        const mid = h / 2;

        ctx.fillStyle = "#0B0E14";
        ctx.fillRect(0, 0, w, h);

        ctx.beginPath();
        ctx.moveTo(0, mid);
        for (let i = 0; i < w; i++) {
          let max = 0;
          for (let j = 0; j < step; j++) {
            const idx = i * step + j;
            if (idx < samples.length) {
              const abs = Math.abs(samples[idx]);
              if (abs > max) max = abs;
            }
          }
          const y = mid - max * mid * 0.9;
          ctx.lineTo(i, y);
        }
        ctx.lineTo(w, mid);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.lineTo(w, h);
        ctx.lineTo(0, h);
        ctx.closePath();
        ctx.fillStyle = WAVEFORM_SECONDARY;
        ctx.fill();
      })
      .catch(() => {});

    return () => {
      cancelled = true;
      audioCtx.close();
    };
  }, [audioUrl, color]);

  return (
    <div className="canvas-wrap">
      <canvas
        ref={canvasRef}
        className="w-full block"
        style={{ height: "100px" }}
      />
    </div>
  );
}

// ── Section wrapper with staggered entrance ──────────────────────────────────

// ── Convert webm blob from MediaRecorder to a WAV file ────────────────────────

async function convertBlobToWav(blob: Blob): Promise<File> {
  const arrayBuffer = await blob.arrayBuffer();
  const audioCtx = new AudioContext();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  await audioCtx.close();

  // Get mono channel (first channel)
  const channelData = audioBuffer.getChannelData(0);
  const samples = new Float32Array(audioBuffer.length);
  samples.set(channelData);

  const wavBlob = encodeWav(samples, audioBuffer.sampleRate);
  return new File([wavBlob], "recording.wav", { type: "audio/wav" });
}

// ── Audio Recorder Component ───────────────────────────────────────────────────

function AudioRecorder({
  onRecording,
}: {
  onRecording: (file: File) => void;
}) {
  const [state, setState] = useState<
    "idle" | "requesting" | "denied" | "recording" | "processing"
  >("idle");
  const [duration, setDuration] = useState(0);
  const [level, setLevel] = useState(0);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const animRef = useRef<number | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
      if (timerRef.current) clearInterval(timerRef.current);
      if (audioCtxRef.current) audioCtxRef.current.close();
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const stopRecording = useCallback(() => {
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== "inactive"
    ) {
      mediaRecorderRef.current.stop();
    }
  }, []);

  const startRecording = useCallback(async () => {
    setState("requesting");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Set up AnalyserNode for live level meter
      const audioCtx = new AudioContext();
      audioCtxRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserRef.current = analyser;

      // Determine best supported mime type
      const mimeType = MediaRecorder.isTypeSupported(
        "audio/webm;codecs=opus"
      )
        ? "audio/webm;codecs=opus"
        : "audio/webm";

      const recorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (e: BlobEvent) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        // Stop level meter animation frame
        if (animRef.current) cancelAnimationFrame(animRef.current);
        if (timerRef.current) clearInterval(timerRef.current);

        // Stop all tracks
        stream.getTracks().forEach((t) => t.stop());
        setState("processing");

        // Decode webm → PCM → encode as WAV
        const blob = new Blob(chunksRef.current, { type: mimeType });
        try {
          const wavFile = await convertBlobToWav(blob);
          await onRecording(wavFile); // wait for prediction to start
        } catch {
          // conversion error — reset to idle
        } finally {
          setState("idle"); // allow re-recording after processing
        }

        // Close audio context after we're done with it
        if (audioCtxRef.current) {
          audioCtxRef.current.close();
          audioCtxRef.current = null;
        }
      };

      recorder.onerror = () => {
        setState("idle");
        cleanupRecorder();
      };

      recorder.start();
      setState("recording");
      setDuration(0);
      setLevel(0);

      // Start timer
      const startTime = Date.now();
      timerRef.current = window.setInterval(() => {
        setDuration((Date.now() - startTime) / 1000);
      }, 100);

      // Start level meter using requestAnimationFrame
      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      const updateLevel = () => {
        if (analyserRef.current) {
          analyserRef.current.getByteFrequencyData(dataArray);
          let sum = 0;
          for (let i = 0; i < dataArray.length; i++) {
            sum += dataArray[i];
          }
          const avg = sum / dataArray.length;
          setLevel(Math.min(1, avg / 128));
          animRef.current = requestAnimationFrame(updateLevel);
        }
      };
      animRef.current = requestAnimationFrame(updateLevel);
    } catch (err) {
      const domErr = err as DOMException;
      if (domErr.name === "NotAllowedError") {
        setState("denied");
      } else {
        setState("idle");
        cleanupRecorder();
      }
    }
  }, [onRecording]);

  const cleanupRecorder = useCallback(() => {
    if (animRef.current) cancelAnimationFrame(animRef.current);
    if (timerRef.current) clearInterval(timerRef.current);
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    mediaRecorderRef.current = null;
  }, []);

  const retry = useCallback(() => {
    setState("idle");
    setLevel(0);
    setDuration(0);
  }, []);

  const fmtTime = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  const isBusy = state === "requesting" || state === "processing";

  return (
    <div className={`recorder-bar ${state === "recording" ? "recording" : ""}`}>
      <button
        className={`btn-record ${state === "recording" ? "recording" : ""}`}
        onClick={state === "recording" ? stopRecording : startRecording}
        disabled={isBusy}
        title={
          state === "recording"
            ? "Stop recording"
            : state === "denied"
              ? "Microphone access denied"
              : "Start recording"
        }
      >
        {state === "recording" ? (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
            <rect x="6" y="6" width="12" height="12" rx="2" />
          </svg>
        ) : (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 14a3 3 0 0 0 3-3V5a3 3 0 1 0-6 0v6a3 3 0 0 0 3 3Z" />
            <path d="M17 11a1 1 0 0 0-2 0 3 3 0 0 1-6 0 1 1 0 0 0-2 0 5 5 0 0 0 4 4.9V19H9a1 1 0 1 0 0 2h6a1 1 0 1 0 0-2h-2v-3.1a5 5 0 0 0 4-4.9Z" />
          </svg>
        )}
      </button>

      <div className="recorder-info">
        <div className="recorder-label">
          {state === "idle"
            ? "Record live"
            : state === "requesting"
              ? "Requesting microphone…"
              : state === "denied"
                ? "Microphone access denied"
                : state === "recording"
                  ? "Recording…"
                  : "Processing recording…"}
        </div>
        <div className="recorder-timer">
          {state === "denied" ? (
            <button
              onClick={retry}
              className="text-xs text-accent-purple hover:underline"
            >
              Click to retry
            </button>
          ) : state === "recording" || state === "processing" ? (
            fmtTime(duration)
          ) : state === "requesting" ? (
            <span className="text-sm text-text-dim animate-pulse">…</span>
          ) : (
            <span className="text-sm text-text-dim">Tap to start</span>
          )}
        </div>
        {state === "recording" && (
          <div className="recorder-meter">
            <div
              className={`recorder-meter-fill recording-level`}
              style={{ width: `${level * 100}%` }}
            />
          </div>
        )}
      </div>
    </div>
  );
}

function Section({
  title,
  stagger,
  children,
  className = "",
}: {
  title: string;
  stagger: number;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`slide-up stagger-${stagger} ${className}`}>
      <div className="section-title">{title}</div>
      {children}
    </div>
  );
}

// ── Main Page ────────────────────────────────────────────────────────────────

export default function Home() {
  const [result, setResult] = useState<PredictResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Cleanup audio URLs on unmount
  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  const handleFile = useCallback(async (file: File) => {
    const name = file.name.toLowerCase();
    if (!name.endsWith(".wav") && !name.endsWith(".mp3")) {
      setError("Only WAV and MP3 files are supported.");
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setError("File too large. Maximum is 20 MB.");
      return;
    }

    setError(null);
    setResult(null);

    // Convert non-WAV files (e.g. MP3) to WAV client-side using the
    // browser's native decoder — no ffmpeg needed on the server.
    let uploadFile = file;
    if (needsConversion(file.name)) {
      try {
        uploadFile = await decodeAudioFileToWav(file);
      } catch {
        setError(
          "Your browser couldn't decode this audio file. Try a WAV file instead."
        );
        return;
      }
    }

    if (audioUrl) URL.revokeObjectURL(audioUrl);

    const url = URL.createObjectURL(uploadFile);
    setAudioUrl(url);
    setFileName(file.name);
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("audio", uploadFile);

      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || `Server error: ${response.status}`);
      }

      const data: PredictResult = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }, [audioUrl]);

  // ── Drag-and-drop handlers ────────────────────────────────────────────────

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(true);
  }, []);

  const onDragLeave = useCallback(() => {
    setDragging(false);
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const sortedEmotions = result
    ? [...EMOTIONS].sort(
        (a, b) =>
          (result.probabilities[b] ?? 0) - (result.probabilities[a] ?? 0)
      )
    : [];

  return (
    <main className="min-h-screen max-w-2xl mx-auto px-3 sm:px-6 py-6 sm:py-10">
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <header className="text-center mb-8 sm:mb-10 slide-up stagger-1">
        <div className="relative inline-block">
          <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold tracking-tight mb-1 bg-gradient-to-r from-accent-purple via-accent-pink to-accent-amber bg-clip-text text-transparent">
            audiovec
          </h1>
        </div>
        <p className="text-xs sm:text-sm text-text-muted tracking-wider mt-2 max-w-md mx-auto leading-relaxed">
          Upload speech audio &middot; Predict emotion &middot; Extract 256-d
          embedding
        </p>
      </header>

      {/* ── File upload ─────────────────────────────────────────────────── */}
      <div className="slide-up stagger-2 mb-8">
        <div
          className={`upload-zone ${dragging ? "dragover" : ""} ${
            loading ? "pointer-events-none opacity-60" : ""
          } ${fileName ? "has-file" : ""}`}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          onClick={() => !loading && fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleFile(file);
            }}
          />
          {fileName ? (
            <div className="space-y-1 relative z-10">
              <div className="text-accent-purple text-base sm:text-lg font-medium truncate max-w-xs mx-auto">
                {fileName}
              </div>
              <div className="text-text-dim text-xs">
                Click or drop to change file
              </div>
            </div>
          ) : (
            <div className="space-y-3 relative z-10">
              <div className="text-4xl sm:text-5xl select-none">🎤</div>
              <div className="text-text-secondary font-medium text-sm sm:text-base">
                Drop a WAV or MP3 file here
              </div>
              <div className="text-text-dim text-xs">
                or click to browse &middot; WAV or MP3
              </div>
            </div>
          )}
        </div>

        {/* ── or divider ──────────────────────────────────────────────── */}
        <div className="flex items-center gap-3 my-4">
          <div className="flex-1 h-px bg-surface-border/50" />
          <span className="text-xs text-text-dim uppercase tracking-widest">
            or
          </span>
          <div className="flex-1 h-px bg-surface-border/50" />
        </div>

        {/* ── Record live ──────────────────────────────────────────────── */}
        <AudioRecorder onRecording={handleFile} />
      </div>

      {/* ── Loading state ────────────────────────────────────────────────── */}
      {loading && (
        <div className="slide-up stagger-3 mb-8">
          <div className="glass-card p-5 sm:p-6">
            <StatusDot label="Processing audio…" />

            {/* Animated progress bar */}
            <div className="mt-4 w-full rounded-full h-1.5 overflow-hidden bg-surface-border/30">
              <div className="h-full rounded-full gradient-sweep w-3/4" />
            </div>

            {/* Skeleton preview cards */}
            <div className="grid grid-cols-3 gap-3 mt-5">
              <div className="h-20 rounded-xl skeleton-block" />
              <div className="h-20 rounded-xl skeleton-block" />
              <div className="h-20 rounded-xl skeleton-block" />
            </div>
          </div>
        </div>
      )}

      {/* ── Error state ──────────────────────────────────────────────────── */}
      {error && !loading && (
        <div className="slide-up stagger-3 mb-8">
          <div className="banner-error">{error}</div>
        </div>
      )}

      {/* ── Results ──────────────────────────────────────────────────────── */}
      {result && !loading && (
        <div className="space-y-6 sm:space-y-8">
          {/* Audio Playback */}
          {audioUrl && (
            <Section title="Audio" stagger={3}>
              <div className="glass-card p-4 sm:p-5">
                <audio src={audioUrl} controls />
                <p className="text-xs text-text-dim mt-2.5">
                  {fileName} &middot; {result.duration.toFixed(1)}s
                </p>
              </div>
            </Section>
          )}

          {/* Prediction → Metrics + Confidence */}
          <Section title="Prediction" stagger={4}>
            <div className="metrics-grid">
              <MetricCard
                label="Emotion"
                value={result.emotion.toUpperCase()}
                className="metric-card"
              />
              <div className="metric-card !p-3">
                <ConfidenceRing confidence={result.confidence} />
              </div>
              <MetricCard
                label="Duration"
                value={`${result.duration.toFixed(1)}s`}
              />
            </div>
          </Section>

          {/* Emotion Probabilities */}
          <Section title="Probabilities" stagger={5}>
            <div className="glass-card p-4 sm:p-5">
              {EMOTIONS.map((emotion, i) => {
                const prob = result.probabilities[emotion] ?? 0;
                return (
                  <EmotionBar
                    key={emotion}
                    label={
                      emotion.charAt(0).toUpperCase() + emotion.slice(1)
                    }
                    prob={prob}
                    color={EMOTION_COLORS[emotion]}
                    isActive={emotion === result.emotion}
                    delay={i * 60}
                  />
                );
              })}
            </div>
          </Section>

          {/* Embedding + Spectrogram side by side on larger screens */}
          <Section title="Features" stagger={6}>
            <div className="glass-card p-4 sm:p-5 space-y-5">
              <EmbeddingCanvas embedding={result.embedding} />
              <SpectrogramCanvas
                spectrogramData={result.spectrogram}
                duration={result.duration}
              />
            </div>
          </Section>

          {/* Waveform */}
          {audioUrl && (
            <Section title="Waveform" stagger={7}>
              <WaveformCanvas audioUrl={audioUrl} />
            </Section>
          )}

          {/* Download + Footer */}
          <div className="slide-up stagger-8 space-y-6">
            <button
              onClick={() => {
                const blob = new Blob(
                  [JSON.stringify(result, null, 2)],
                  { type: "application/json" }
                );
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = "audiovec_embedding.json";
                a.click();
                URL.revokeObjectURL(url);
              }}
              className="btn-download"
            >
              <span className="relative z-10">
                ⬇ Download Embedding as JSON
              </span>
            </button>

            <div className="text-center text-text-dim text-xs pb-4 border-t border-surface-border/50 pt-6">
              audiovec &middot; 256-dimensional audio sentiment embedding
              &middot; RAVDESS dataset
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
