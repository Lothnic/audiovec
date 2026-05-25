// ── Emotion constants (matching EMOTION_MAPPING 1-8) ─────────────────────────

export const EMOTIONS = [
  "neutral",
  "calm",
  "happy",
  "sad",
  "angry",
  "fearful",
  "disgust",
  "surprised",
] as const;

export type Emotion = (typeof EMOTIONS)[number];

export const EMOTION_COLORS: Record<Emotion, string> = {
  neutral: "#A78BFA",
  calm: "#60A5FA",
  happy: "#34D399",
  sad: "#FBBF24",
  angry: "#F87171",
  fearful: "#FB923C",
  disgust: "#E879F9",
  surprised: "#22D3EE",
};

export const EMOTION_COLOR_ARRAY = EMOTIONS.map((e) => EMOTION_COLORS[e]);

// ── Inference result types ───────────────────────────────────────────────────

export interface PredictResult {
  emotion: Emotion;
  confidence: number;
  probabilities: Record<Emotion, number>;
  embedding: number[];
  duration: number;
  /** Mel-spectrogram as 2D array [nMels, nFrames] normalized to [0, 1]. */
  spectrogram: number[][];
}

export interface ApiError {
  error: string;
}
