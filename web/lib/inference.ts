/**
 * inference.ts — ONNX Runtime inference wrapper.
 *
 * Loads the CRNN-Transformer ONNX model and runs predictions.
 * Uses a singleton session pattern — the model is loaded once and reused.
 */

import path from "path";
import * as ort from "onnxruntime-node";
import { audioToSpectrogram } from "./melspectrogram";
import { EMOTIONS, type Emotion, type PredictResult } from "./emotions";

// ── ONNX session (singleton) ────────────────────────────────────────────────

let _session: ort.InferenceSession | null = null;

async function getSession(): Promise<ort.InferenceSession> {
  if (!_session) {
    // Allow overriding the model path via env var.
    // Default: look in the same directory as the built function output
    // (traced via next.config.ts outputFileTracingIncludes).
    const envPath = process.env.MODEL_PATH;
    const modelPath = envPath
      ? path.resolve(process.cwd(), envPath)
      : path.resolve(__dirname, "models", "crnn-transformer.onnx");
    _session = await ort.InferenceSession.create(modelPath);
  }
  return _session;
}

// ── Inference ────────────────────────────────────────────────────────────────

/**
 * Run emotion prediction on raw PCM audio samples.
 *
 * @param samples - Raw PCM audio samples (mono)
 * @param sr - Sample rate of the audio
 * @param durationSec - Duration of the audio in seconds
 * @returns PredictResult with emotion, confidence, probabilities, and embedding
 */
export async function predict(
  samples: Float32Array,
  sr: number,
  durationSec: number
): Promise<PredictResult> {
  const session = await getSession();

  // 1. Compute mel-spectrogram
  const { data: spectrogram } = audioToSpectrogram(samples, sr);

  // 2. Prepare ONNX input: (1, 128, 228, 1) float32
  const inputData = new Float32Array(128 * 228);
  for (let i = 0; i < 128 * 228; i++) {
    inputData[i] = spectrogram[i];
  }

  const inputTensor = new ort.Tensor("float32", inputData, [
    1,
    128,
    228,
    1,
  ]);

  // 3. Run inference
  const results = await session.run({ spectrogram: inputTensor });

  const logits = results.logits.data as Float32Array; // (8,)
  const embedding = results.embedding.data as Float32Array; // (256,)

  // 4. Softmax
  const maxLogit = Math.max(...logits);
  const exps = Array.from(logits).map((v) => Math.exp(v - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((v) => v / sumExps);

  // 5. Reshape spectrogram from flat Float64Array to number[][]
  const nMels = 128;
  const nFrames = 228;
  const spectrogram2d: number[][] = [];
  for (let m = 0; m < nMels; m++) {
    const row: number[] = [];
    for (let t = 0; t < nFrames; t++) {
      row.push(spectrogram[m * nFrames + t]);
    }
    spectrogram2d.push(row);
  }

  // 6. Build result
  const probabilities = Object.fromEntries(
    EMOTIONS.map((e, i) => [e, probs[i]])
  ) as Record<Emotion, number>;

  const topIdx = probs.indexOf(Math.max(...probs));
  const emotion = EMOTIONS[topIdx];
  const confidence = probs[topIdx];

  return {
    emotion,
    confidence,
    probabilities,
    embedding: Array.from(embedding),
    duration: durationSec,
    spectrogram: spectrogram2d,
  };
}
