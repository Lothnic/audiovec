/**
 * melspectrogram.js — Pure JS mel-spectrogram computation.
 *
 * Replicates the preprocessing pipeline from `audiovec.data.load_and_process_audio`
 * so that audiovec's ONNX model can be used directly from Node.js without
 * calling Python.
 *
 * Parameters (matching audiovec/config.py):
 *   sampleRate  = 22050
 *   nMels       = 128
 *   fMax        = 8000
 *   nFft        = 2048   (librosa default)
 *   hopLength   = 512    (librosa default)
 *   maxPadLen   = 228
 *   topDb       = 80
 */

// ── Constants ────────────────────────────────────────────────────────────────

const SAMPLE_RATE = 22050;
const N_MELS = 128;
const F_MAX = 8000;
const N_FFT = 2048;
const HOP_LENGTH = 512;
const MAX_PAD_LEN = 228;
const TOP_DB = 80.0;

// ── Pre-computed Mel filterbank ──────────────────────────────────────────────
//
// librosa.filters.mel(sr=22050, n_fft=2048, n_mels=128, fmax=8000)
// yields a matrix of shape (128, 1025), where 1025 = n_fft//2 + 1.
//
// We pre-compute it here so the JS code doesn't need to re-derive it.

const N_FREQS = N_FFT / 2 + 1; // 1025

/**
 * Build the mel filterbank matrix.
 * Returns Float64Array of length nMels * nFreqs (row-major).
 *
 * Algorithm: triangular filters evenly spaced on the mel scale.
 */
function buildMelFilterbank() {
  const melMin = 2595 * Math.log10(1 + 0 / 700);
  const melMax = 2595 * Math.log10(1 + F_MAX / 700);
  const melPoints = new Float64Array(N_MELS + 2);

  for (let i = 0; i < N_MELS + 2; i++) {
    const mel = melMin + (i / (N_MELS + 1)) * (melMax - melMin);
    melPoints[i] = 700 * (Math.pow(10, mel / 2595) - 1);
  }

  // Convert to FFT bin indices
  const bins = new Float64Array(N_MELS + 2);
  for (let i = 0; i < N_MELS + 2; i++) {
    bins[i] = Math.floor((N_FFT + 1) * melPoints[i] / SAMPLE_RATE);
  }

  const fb = new Float64Array(N_MELS * N_FREQS);
  for (let m = 0; m < N_MELS; m++) {
    const bL = bins[m];
    const bC = bins[m + 1];
    const bR = bins[m + 2];

    for (let k = bL; k <= bR; k++) {
      const val =
        k < bC
          ? (k - bL) / (bC - bL)
          : (bR - k) / (bR - bC);
      fb[m * N_FREQS + k] = val;
    }
  }

  return fb;
}

let _melFilterbank = null;
function getMelFilterbank() {
  if (!_melFilterbank) _melFilterbank = buildMelFilterbank();
  return _melFilterbank;
}

// ── FFT (radix-2 Cooley-Tukey) ──────────────────────────────────────────────

/**
 * In-place radix-2 FFT.
 * @param {Float64Array} real
 * @param {Float64Array} imag
 */
function fft(real, imag) {
  const n = real.length;
  if (n <= 1) return;

  // Bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      let tmp = real[i]; real[i] = real[j]; real[j] = tmp;
      tmp = imag[i]; imag[i] = imag[j]; imag[j] = tmp;
    }
  }

  // Butterfly
  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = -2 * Math.PI / len;
    const wReal = Math.cos(angle);
    const wImag = Math.sin(angle);

    for (let i = 0; i < n; i += len) {
      let curReal = 1, curImag = 0;
      for (let j = 0; j < halfLen; j++) {
        const uReal = real[i + j];
        const uImag = imag[i + j];
        const vReal = real[i + j + halfLen] * curReal - imag[i + j + halfLen] * curImag;
        const vImag = real[i + j + halfLen] * curImag + imag[i + j + halfLen] * curReal;

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

/**
 * Compute magnitude spectrum of a single frame (Hann-windowed).
 * @param {Float64Array} frame - length nFft
 * @param {Float64Array} window - Hann window, length nFft
 * @param {Float64Array} outReal
 * @param {Float64Array} outImag
 * @returns {Float64Array} magnitude spectrum, length nFreqs (1025)
 */
function frameMagnitude(frame, window, outReal, outImag) {
  for (let i = 0; i < N_FFT; i++) {
    outReal[i] = frame[i] * window[i];
    outImag[i] = 0;
  }
  fft(outReal, outImag);

  // Magnitude spectrum |FFT|, only positive frequencies
  const mag = new Float64Array(N_FREQS);
  for (let i = 0; i < N_FREQS; i++) {
    mag[i] = Math.sqrt(outReal[i] * outReal[i] + outImag[i] * outImag[i]);
  }
  return mag;
}

/**
 * Generate a Hann window of length n.
 */
function hannWindow(n) {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
  }
  return w;
}

// ── Public API ───────────────────────────────────────────────────────────────

/**
 * Convert raw audio samples to a normalised mel-spectrogram
 * matching audiovec's preprocessing pipeline.
 *
 * @param {Float64Array} samples - Raw PCM samples (mono, any bit-depth).
 * @param {number} sr - Sample rate of `samples`.
 * @returns {Float64Array} Spectrogram of shape (N_MELS, MAX_PAD_LEN)
 *                         normalized to [0, 1] with dB scaling.
 */
function audioToSpectrogram(samples, sr) {
  // 0. Resample to 22050 Hz if needed (simple linear interpolation)
  let audio = samples;
  if (sr !== SAMPLE_RATE) {
    const ratio = SAMPLE_RATE / sr;
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
    audio = resampled;
  }

  // 1. STFT  (with center padding, matching librosa's center=True default)
  const window = hannWindow(N_FFT);
  const outReal = new Float64Array(N_FFT);
  const outImag = new Float64Array(N_FFT);
  const fb = getMelFilterbank();

  // Pad signal with n_fft/2 zeros on each side (librosa center=True)
  const padLen = N_FFT >>> 1;
  const paddedAudio = new Float64Array(audio.length + N_FFT);
  paddedAudio.set(audio, padLen);

  const nFrames = Math.max(1, Math.floor((paddedAudio.length - N_FFT) / HOP_LENGTH) + 1);

  // (nMels × nFrames) power spectrogram
  const melSpec = new Float64Array(N_MELS * nFrames);

  for (let t = 0; t < nFrames; t++) {
    const start = t * HOP_LENGTH;
    const frame = paddedAudio.subarray(start, start + N_FFT);

    const mag = frameMagnitude(frame, window, outReal, outImag);

    // 2. Apply mel filterbank: melSpec[m, t] = sum_k fb[m,k] * mag[k]^2
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

  // 3. Power-to-dB: 10 * log10(melSpec / max)
  let maxVal = 0;
  for (let i = 0; i < melSpec.length; i++) {
    if (melSpec[i] > maxVal) maxVal = melSpec[i];
  }
  if (maxVal === 0) maxVal = 1e-10;

  const dB = new Float64Array(N_MELS * MAX_PAD_LEN);
  const padWidth = MAX_PAD_LEN - nFrames;

  // 4. Convert to dB, normalize to [0,1], pad/trim to MAX_PAD_LEN
  for (let m = 0; m < N_MELS; m++) {
    for (let t = 0; t < MAX_PAD_LEN; t++) {
      let val;
      if (t < nFrames) {
        const power = melSpec[m * nFrames + t] / maxVal;
        // clamp to avoid log(0)
        const clamped = Math.max(power, 1e-10);
        val = 10 * Math.log10(clamped);
        // clamp to [-topDb, 0]
        val = Math.max(val, -TOP_DB);
      } else {
        val = -TOP_DB;
      }
      // Normalize to [0, 1]
      dB[m * MAX_PAD_LEN + t] = (val + TOP_DB) / TOP_DB;
    }
  }

  return { data: dB, nMels: N_MELS, nFrames: MAX_PAD_LEN };
}

// ── Export ───────────────────────────────────────────────────────────────────

module.exports = { audioToSpectrogram, N_MELS, MAX_PAD_LEN, SAMPLE_RATE };
