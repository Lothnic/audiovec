/**
 * wav-encoder.ts — Encode raw PCM samples into a WAV blob.
 *
 * Takes a Float32Array of mono PCM samples and produces a standard
 * 16-bit PCM WAV file blob suitable for the prediction API.
 *
 * WAV format:
 *   RIFF header (12 B) + fmt chunk (24 B) + data chunk header (8 B) + PCM data
 *   = 44 bytes of headers total
 */

/** Number of bits per sample in the output WAV. */
const BITS_PER_SAMPLE = 16 as const;

/** Number of audio channels (mono). */
const NUM_CHANNELS = 1 as const;

/**
 * Encode raw PCM samples into a WAV blob.
 *
 * @param samples  - Interleaved Float32Array samples (values in [-1, 1]).
 * @param sr       - Sample rate of the audio.
 * @returns A Blob with MIME type audio/wav containing the WAV file.
 */
export function encodeWav(samples: Float32Array, sr: number): Blob {
  const numSamples = samples.length;
  const byteRate = sr * NUM_CHANNELS * (BITS_PER_SAMPLE / 8);
  const blockAlign = NUM_CHANNELS * (BITS_PER_SAMPLE / 8);
  const dataSize = numSamples * blockAlign;
  const bufferSize = 44 + dataSize;

  const buffer = new ArrayBuffer(bufferSize);
  const view = new DataView(buffer);

  // ── RIFF header ──────────────────────────────────────────────────────────
  writeString(view, 0, "RIFF");
  view.setUint32(4, bufferSize - 8, true);        // file size - 8
  writeString(view, 8, "WAVE");

  // ── fmt chunk ────────────────────────────────────────────────────────────
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);                    // chunk size
  view.setUint16(20, 1, true);                     // PCM = 1
  view.setUint16(22, NUM_CHANNELS, true);          // channels
  view.setUint32(24, sr, true);                    // sample rate
  view.setUint32(28, byteRate, true);              // byte rate
  view.setUint16(32, blockAlign, true);            // block align
  view.setUint16(34, BITS_PER_SAMPLE, true);       // bits per sample

  // ── data chunk ───────────────────────────────────────────────────────────
  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true);              // data size

  // ── PCM data (16-bit signed) ─────────────────────────────────────────────
  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    // Clamp to [-1, 1] and convert to 16-bit signed integer
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    const int16 = clamped < 0
      ? Math.round(clamped * 0x8000)
      : Math.round(clamped * 0x7FFF);
    view.setInt16(offset, int16, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

/** Helper: write a 4-character string at a byte offset in a DataView. */
function writeString(view: DataView, offset: number, str: string): void {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}
