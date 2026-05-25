/**
 * inference.ts — ONNX Runtime inference wrapper.
 *
 * Loads the CRNN-Transformer ONNX model and runs predictions.
 * Uses a singleton session pattern — the model is loaded once and reused.
 */

import fs from "fs";
import path from "path";
import * as ort from "onnxruntime-node";
import { audioToSpectrogram } from "./melspectrogram";
import { EMOTIONS, type Emotion, type PredictResult } from "./emotions";

// ── Model resolution ─────────────────────────────────────────────────────────
//
// The ONNX model needs to be present on the filesystem at runtime.  Vercel's
// serverless-function file tracing (`outputFileTracingIncludes`) is unreliable
// for binary files inside the app directory, so we fall back to downloading
// the model from GitHub on cold start and caching it in /tmp.

const MODEL_FILENAME = "crnn-transformer.onnx";

/**
 * Public URL for the ONNX model file (tracked in git at
 * `web/app/api/predict/models/crnn-transformer.onnx`).
 * Used as a download fallback when file tracing doesn't include the model.
 */
const MODEL_DOWNLOAD_URL =
  "https://raw.githubusercontent.com/Lothnic/audiovec/master/web/app/api/predict/models/crnn-transformer.onnx";

/**
 * Try to resolve the model file from several locations in order of preference.
 * Returns the first path that exists on disk, or null if not found anywhere.
 */
function resolveModelPath(): string | null {
  // 1. Env-var override (absolute or relative to cwd)
  if (process.env.MODEL_PATH) {
    return path.resolve(process.cwd(), process.env.MODEL_PATH);
  }

  // 2. Next.js file-trace output (__dirname in bundled serverless function)
  const tracedPath = path.resolve(__dirname, "models", MODEL_FILENAME);
  if (fs.existsSync(tracedPath)) {
    return tracedPath;
  }

  // 3. /tmp cache (downloaded on a previous cold start)
  const cachedPath = path.resolve("/tmp", "models", MODEL_FILENAME);
  if (fs.existsSync(cachedPath)) {
    return cachedPath;
  }

  return null;
}

/**
 * Download the model from the public URL and save it to /tmp/models/.
 */
async function downloadModel(): Promise<string> {
  const dir = path.resolve("/tmp", "models");
  fs.mkdirSync(dir, { recursive: true });
  const dest = path.resolve(dir, MODEL_FILENAME);

  console.log(`Downloading model from ${MODEL_DOWNLOAD_URL}`);
  const response = await fetch(MODEL_DOWNLOAD_URL);
  if (!response.ok) {
    throw new Error(
      `Failed to download model: HTTP ${response.status} ${response.statusText}`
    );
  }

  const buffer = Buffer.from(await response.arrayBuffer());
  fs.writeFileSync(dest, buffer);
  console.log(`Model downloaded (${buffer.length} bytes) to ${dest}`);
  return dest;
}

// ── ONNX session (singleton) ────────────────────────────────────────────────

let _session: ort.InferenceSession | null = null;

async function getSession(): Promise<ort.InferenceSession> {
  if (!_session) {
    const localPath = resolveModelPath();
    const modelPath = localPath ?? (await downloadModel());
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
