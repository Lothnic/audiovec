/**
 * audio-utils.ts — Client-side audio format conversion.
 *
 * Uses the browser's native AudioContext.decodeAudioData() to decode
 * MP3/AAC/WebM files to raw PCM, then re-encodes them as WAV so the
 * server only needs to handle WAV (no ffmpeg dependency in production).
 *
 * Supported input formats depend on the browser:
 *   Chrome:  MP3, WAV, AAC, WebM (Opus/Vorbis), FLAC
 *   Firefox: MP3, WAV, Ogg (Vorbis), WebM (Opus)
 *   Safari:  MP3, WAV, AAC, MP4
 */

import { encodeWav } from "./wav-encoder";

/**
 * Decode any browser-decoded audio file to WAV.
 *
 * @param file  The uploaded audio file (MP3, AAC, WebM, etc.).
 * @returns A WAV File object ready to send to the prediction API.
 */
export async function decodeAudioFileToWav(file: File): Promise<File> {
  const arrayBuffer = await file.arrayBuffer();
  const audioCtx = new AudioContext();

  try {
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    // Get mono channel (first channel)
    const channelData = audioBuffer.getChannelData(0);
    const samples = new Float32Array(audioBuffer.length);
    samples.set(channelData);

    // Encode as 16-bit PCM WAV
    const wavBlob = encodeWav(samples, audioBuffer.sampleRate);

    // Derive name (replace extension with .wav)
    const name = file.name.replace(/\.[^.]+$/, ".wav");
    return new File([wavBlob], name, { type: "audio/wav" });
  } finally {
    audioCtx.close();
  }
}

/**
 * Check whether an audio file needs client-side conversion before upload.
 * WAV files can be sent as-is; everything else needs decoding.
 */
export function needsConversion(fileName: string): boolean {
  return !fileName.toLowerCase().endsWith(".wav");
}
