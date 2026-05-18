#!/usr/bin/env node

/**
 * predict.js — Node.js inference for audiovec ONNX model.
 *
 * Usage:
 *   node predict.js <path-to-wav> [options]
 *
 * Options:
 *   --top-k <n>    Show top-N emotions (default: 3)
 *   --json         Output raw JSON with logits + embedding
 *
 * Example:
 *   node predict.js speech.wav
 *   node predict.js speech.wav --top-k 5
 *   node predict.js speech.wav --json
 *
 * Dependencies:
 *   npm install onnxruntime-node wav-decoder
 */

const fs = require("fs");
const path = require("path");
const ort = require("onnxruntime-node");
const wavDecoder = require("wav-decoder");
const { audioToSpectrogram, N_MELS, MAX_PAD_LEN } = require("./melspectrogram");

// ── Emotion labels (matching EMOTION_MAPPING 1-8) ────────────────────────────

const EMOTIONS = [
  "neutral", "calm", "happy", "sad",
  "angry", "fearful", "disgust", "surprised",
];

const EMOTION_EMOJIS = [
  "😐", "😌", "😊", "😢", "😡", "😰", "🤢", "😲",
];

// ── ONNX session (loaded once) ───────────────────────────────────────────────

let _session = null;

async function getSession() {
  if (!_session) {
    const modelPath = path.join(__dirname, "..", "..", "models", "crnn-transformer.onnx");
    if (!fs.existsSync(modelPath)) {
      console.error(
        `ONNX model not found at: ${modelPath}\n` +
        "Run `uv run python -m audiovec.export_onnx` first to generate it."
      );
      process.exit(1);
    }
    _session = await ort.InferenceSession.create(modelPath);
  }
  return _session;
}

// ── Inference ────────────────────────────────────────────────────────────────

/**
 * Run emotion prediction on a mel-spectrogram.
 *
 * @param {Float64Array} spectrogram - Flattened (128, 228) array.
 * @returns {{ logits: Float64Array, embedding: Float64Array, probs: Float64Array }}
 */
async function predict(spectrogram) {
  const session = await getSession();

  // ONNX expects float32 — shape (1, 128, 228, 1)
  const inputData = new Float32Array(N_MELS * MAX_PAD_LEN);
  for (let i = 0; i < N_MELS * MAX_PAD_LEN; i++) {
    inputData[i] = spectrogram[i];
  }

  const inputTensor = new ort.Tensor("float32", inputData, [1, N_MELS, MAX_PAD_LEN, 1]);

  const feeds = { spectrogram: inputTensor };
  const results = await session.run(feeds);

  const logits = results.logits.data;     // Float32Array(8)
  const embedding = results.embedding.data; // Float32Array(256)

  // Softmax
  const maxLogit = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((v) => v / sumExps);

  return { logits, embedding, probs };
}

// ── CLI ──────────────────────────────────────────────────────────────────────

function parseArgs() {
  const args = process.argv.slice(2);
  if (args.length === 0 || args[0] === "--help" || args[0] === "-h") {
    console.log(`
Usage: node predict.js <path-to-wav> [options]

Options:
  --top-k <n>    Show top-N emotions (default: 3)
  --json         Output raw JSON with logits + embedding
  --help, -h     Show this help

Example:
  node predict.js speech.wav
  node predict.js speech.wav --top-k 5
  node predict.js speech.wav --json
    `.trim());
    process.exit(0);
  }

  const wavPath = args[0];
  let topK = 3;
  let jsonOutput = false;

  for (let i = 1; i < args.length; i++) {
    if (args[i] === "--top-k" && i + 1 < args.length) {
      topK = parseInt(args[++i], 10);
    } else if (args[i] === "--json") {
      jsonOutput = true;
    }
  }

  return { wavPath, topK, jsonOutput };
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  const { wavPath, topK, jsonOutput } = parseArgs();

  // 1. Read WAV file
  if (!fs.existsSync(wavPath)) {
    console.error(`File not found: ${wavPath}`);
    process.exit(1);
  }

  const wavBuffer = fs.readFileSync(wavPath);
  let audioData;
  try {
    audioData = wavDecoder.decode(wavBuffer);
  } catch (err) {
    console.error(`Failed to decode WAV file: ${err.message}`);
    process.exit(1);
  }

  // Extract mono channel
  const samples = audioData.channelData[0]; // Float32Array
  const sr = audioData.sampleRate;

  // 2. Compute mel-spectrogram
  const { data: spectrogram } = audioToSpectrogram(samples, sr);

  // 3. Run inference
  const { logits, embedding, probs } = await predict(spectrogram);

  // 4. Output
  const topIndices = probs
    .map((p, i) => ({ p, i }))
    .sort((a, b) => b.p - a.p)
    .slice(0, topK);

  if (jsonOutput) {
    const result = {
      emotion: EMOTIONS[topIndices[0].i],
      confidence: topIndices[0].p,
      probabilities: Object.fromEntries(EMOTIONS.map((e, i) => [e, probs[i]])),
      embedding: Array.from(embedding),
    };
    console.log(JSON.stringify(result, null, 2));
  } else {
    const winner = topIndices[0];
    const winnerEmoji = EMOTION_EMOJIS[winner.i];
    console.log(`\n  ${winnerEmoji}  ${EMOTIONS[winner.i].toUpperCase()}  (${(winner.p * 100).toFixed(1)}%)\n`);

    console.log("  Top emotions:");
    for (const { p, i } of topIndices) {
      const bar = "█".repeat(Math.round(p * 30));
      console.log(`    ${EMOTIONS[i].padEnd(10)} ${(p * 100).toFixed(1)}%  ${bar}`);
    }
    console.log("");
    console.log(`  Embedding dimension: ${embedding.length}`);
    console.log(`  Embedding range:     [${embedding.reduce((a, b) => Math.min(a, b), Infinity).toFixed(4)}, ${embedding.reduce((a, b) => Math.max(a, b), -Infinity).toFixed(4)}]`);
    console.log("");
  }
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
