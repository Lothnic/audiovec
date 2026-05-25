/**
 * POST /api/predict
 *
 * Accepts a WAV or MP3 audio file via multipart form data and returns
 * emotion prediction results.
 *
 * Request:
 *   Content-Type: multipart/form-data
 *   Body:
 *     audio: File (WAV / MP3)
 *
 * Response (200):
 *   {
 *     emotion: string,
 *     confidence: number,
 *     probabilities: Record<string, number>,
 *     embedding: number[],
 *     duration: number
 *   }
 *
 * Response (400/500):
 *   { error: string }
 */

import { NextRequest, NextResponse } from "next/server";
import * as wavDecoder from "wav-decoder";
import { predict } from "@/lib/inference";

// ── Supported audio format ───────────────────────────────────────────────────
// WAV only. MP3 and other formats are converted to WAV client-side via
// the browser's native AudioContext.decodeAudioData().

const AUDIO_EXTENSION = ".wav";

function isWav(filename: string): boolean {
  return filename.toLowerCase().endsWith(AUDIO_EXTENSION);
}

// ── Route handler ────────────────────────────────────────────────────────────

export async function POST(request: NextRequest) {
  try {
    // 1. Parse multipart form
    const formData = await request.formData();
    const audioFile = formData.get("audio");

    if (!audioFile || !(audioFile instanceof File)) {
      return NextResponse.json(
        {
          error:
            "No audio file provided. Send a WAV or MP3 file as the 'audio' field.",
        },
        { status: 400 }
      );
    }

    // 2. Validate format
    if (!isWav(audioFile.name)) {
      return NextResponse.json(
        {
          error:
            "Only WAV files are accepted by the API. MP3 files are converted " +
            "to WAV client-side in the web interface.",
        },
        { status: 400 }
      );
    }

    // Check file size (20 MB limit)
    if (audioFile.size > 20 * 1024 * 1024) {
      return NextResponse.json(
        { error: "Audio file too large. Maximum size is 20 MB." },
        { status: 400 }
      );
    }

    // 3. Read the file buffer
    const buffer = Buffer.from(await audioFile.arrayBuffer());

    // 4. Decode WAV to PCM samples
    let audioData: wavDecoder.AudioData;
    try {
      audioData = await wavDecoder.decode(buffer);
    } catch {
      return NextResponse.json(
        {
          error:
            "Failed to decode WAV file. Ensure the file is a valid WAV format.",
        },
        { status: 400 }
      );
    }

    const samples = audioData.channelData[0];
    const durationSec = samples.length / audioData.sampleRate;

    if (durationSec < 0.1) {
      return NextResponse.json(
        { error: "Audio is too short (less than 0.1 seconds)." },
        { status: 400 }
      );
    }

    // 5. Run inference (spectrogram expects 22050 Hz)
    const result = await predict(samples, 22050, durationSec);

    // 6. Return result
    return NextResponse.json(result, { status: 200 });
  } catch (error) {
    console.error("Prediction error:", error);
    const message =
      error instanceof Error ? error.message : "Unknown error during prediction";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
