/**
 * melspectrogram.ts — Pure TypeScript mel-spectrogram computation.
 *
 * Replicates audiovec's Python preprocessing pipeline (librosa defaults):
 *   22050 Hz → STFT (2048 FFT, 512 hop) → mel filterbank (128 bands, 8 kHz)
 *   → power-to-dB → normalize to [0, 1] → pad/trim to 228 frames
 */

// ── Constants ────────────────────────────────────────────────────────────────

const SAMPLE_RATE = 22050;
const N_MELS = 128;
const F_MAX = 8000;
const N_FFT = 2048;
const HOP_LENGTH = 512;
const MAX_PAD_LEN = 228;
const TOP_DB = 80.0;
const N_FREQS = N_FFT / 2 + 1; // 1025

// ── Mel filterbank (pre-computed) ────────────────────────────────────────────

function buildMelFilterbank(): Float64Array {
  const melMin = 2595 * Math.log10(1 + 0 / 700);
  const melMax = 2595 * Math.log10(1 + F_MAX / 700);
  const melPoints = new Float64Array(N_MELS + 2);

  for (let i = 0; i < N_MELS + 2; i++) {
    const mel = melMin + (i / (N_MELS + 1)) * (melMax - melMin);
    melPoints[i] = 700 * (Math.pow(10, mel / 2595) - 1);
  }

  const bins = new Float64Array(N_MELS + 2);
  for (let i = 0; i < N_MELS + 2; i++) {
    bins[i] = Math.floor(((N_FFT + 1) * melPoints[i]) / SAMPLE_RATE);
  }

  const fb = new Float64Array(N_MELS * N_FREQS);
  for (let m = 0; m < N_MELS; m++) {
    const bL = bins[m];
    const bC = bins[m + 1];
    const bR = bins[m + 2];
    for (let k = bL; k <= bR; k++) {
      const val = k < bC ? (k - bL) / (bC - bL) : (bR - k) / (bR - bC);
      fb[m * N_FREQS + k] = val;
    }
  }
  return fb;
}

let _melFilterbank: Float64Array | null = null;
function getMelFilterbank(): Float64Array {
  if (!_melFilterbank) _melFilterbank = buildMelFilterbank();
  return _melFilterbank;
}

// ── FFT (radix-2 Cooley-Tukey) ──────────────────────────────────────────────

function fft(real: Float64Array, imag: Float64Array): void {
  const n = real.length;
  if (n <= 1) return;

  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; (j & bit) !== 0; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      let tmp = real[i];
      real[i] = real[j];
      real[j] = tmp;
      tmp = imag[i];
      imag[i] = imag[j];
      imag[j] = tmp;
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = (-2 * Math.PI) / len;
    const wReal = Math.cos(angle);
    const wImag = Math.sin(angle);

    for (let i = 0; i < n; i += len) {
      let curReal = 1,
        curImag = 0;
      for (let j = 0; j < halfLen; j++) {
        const uReal = real[i + j];
        const uImag = imag[i + j];
        const vReal =
          real[i + j + halfLen] * curReal - imag[i + j + halfLen] * curImag;
        const vImag =
          real[i + j + halfLen] * curImag + imag[i + j + halfLen] * curReal;

        real[i + j] = uReal + vReal;
        imag[i + j] = uImag + vImag;
        real[i + j + halfLen] = uReal - vReal;
        imag[i + j + halfLen] = uImag - vImag;

        const newReal = curReal * wReal - curImag * wImag;
        curImag = curReal * wImag + curImag * wReal;
        curReal = newReal;
      }
    }
  }
}

function hannWindow(n: number): Float64Array {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
  }
  return w;
}

function frameMagnitude(
  frame: Float64Array,
  window: Float64Array,
  outReal: Float64Array,
  outImag: Float64Array
): Float64Array {
  for (let i = 0; i < N_FFT; i++) {
    outReal[i] = frame[i] * window[i];
    outImag[i] = 0;
  }
  fft(outReal, outImag);

  const mag = new Float64Array(N_FREQS);
  for (let i = 0; i < N_FREQS; i++) {
    mag[i] = Math.sqrt(outReal[i] * outReal[i] + outImag[i] * outImag[i]);
  }
  return mag;
}

// ── Resampling (linear interpolation) ────────────────────────────────────────

function resample(audio: Float64Array, fromSr: number): Float64Array {
  if (fromSr === SAMPLE_RATE) return audio;
  const ratio = SAMPLE_RATE / fromSr;
  const newLen = Math.ceil(audio.length * ratio);
  const resampled = new Float64Array(newLen);
  for (let i = 0; i < newLen; i++) {
    const pos = i / ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    if (idx >= audio.length - 1) {
      resampled[i] = audio[audio.length - 1];
    } else {
      resampled[i] = audio[idx] * (1 - frac) + audio[idx + 1] * frac;
    }
  }
  return resampled;
}

// ── Public API ───────────────────────────────────────────────────────────────

export interface SpectrogramResult {
  /** Flattened (N_MELS, MAX_PAD_LEN) spectrogram normalized to [0, 1]. */
  data: Float64Array;
  nMels: number;
  nFrames: number;
}

/**
 * Convert raw PCM samples to a normalized mel-spectrogram.
 *
 * @param samples - Raw PCM samples (mono, any bit depth, Float32Array or Float64Array)
 * @param sr - Original sample rate of `samples`
 * @returns Spectrogram result
 */
export function audioToSpectrogram(
  samples: Float32Array | Float64Array,
  sr: number
): SpectrogramResult {
  // Convert to Float64Array if needed
  const audio64 =
    samples instanceof Float64Array
      ? samples
      : Float64Array.from(samples);

  // 0. Resample to SAMPLE_RATE if needed
  const audio = resample(audio64, sr);

  // 1. STFT with center padding
  const window = hannWindow(N_FFT);
  const outReal = new Float64Array(N_FFT);
  const outImag = new Float64Array(N_FFT);
  const fb = getMelFilterbank();

  const padLen = N_FFT >>> 1;
  const paddedAudio = new Float64Array(audio.length + N_FFT);
  paddedAudio.set(audio, padLen);

  const nFrames = Math.max(
    1,
    Math.floor((paddedAudio.length - N_FFT) / HOP_LENGTH) + 1
  );

  const melSpec = new Float64Array(N_MELS * nFrames);

  for (let t = 0; t < nFrames; t++) {
    const start = t * HOP_LENGTH;
    const frame = paddedAudio.subarray(start, start + N_FFT);
    const mag = frameMagnitude(frame, window, outReal, outImag);

    for (let m = 0; m < N_MELS; m++) {
      let val = 0;
      const offset = m * N_FREQS;
      for (let k = 0; k < N_FREQS; k++) {
        const fbVal = fb[offset + k];
        if (fbVal > 0) {
          val += fbVal * (mag[k] * mag[k]);
        }
      }
      melSpec[m * nFrames + t] = val;
    }
  }

  // 3. Find global max
  let maxVal = 0;
  for (let i = 0; i < melSpec.length; i++) {
    if (melSpec[i] > maxVal) maxVal = melSpec[i];
  }
  if (maxVal === 0) maxVal = 1e-10;

  // 4. Power-to-dB + normalize + pad/trim to MAX_PAD_LEN
  const dB = new Float64Array(N_MELS * MAX_PAD_LEN);

  for (let m = 0; m < N_MELS; m++) {
    for (let t = 0; t < MAX_PAD_LEN; t++) {
      let val: number;
      if (t < nFrames) {
        const power = melSpec[m * nFrames + t] / maxVal;
        const clamped = Math.max(power, 1e-10);
        val = 10 * Math.log10(clamped);
        val = Math.max(val, -TOP_DB);
      } else {
        val = -TOP_DB;
      }
      dB[m * MAX_PAD_LEN + t] = (val + TOP_DB) / TOP_DB;
    }
  }

  return { data: dB, nMels: N_MELS, nFrames: MAX_PAD_LEN };
}
