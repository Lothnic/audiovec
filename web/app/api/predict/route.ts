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
import { getModelVersion } from "@/lib/model-version";
import { createTrace, flushTraces } from "@/lib/langfuse";

// ── Supported audio format ───────────────────────────────────────────────────
// WAV only. MP3 and other formats are converted to WAV client-side via
// the browser's native AudioContext.decodeAudioData().

const AUDIO_EXTENSION = ".wav";

function isWav(filename: string): boolean {
  return filename.toLowerCase().endsWith(AUDIO_EXTENSION);
}

// ── Route handler ────────────────────────────────────────────────────────────

export async function POST(request: NextRequest) {
  const modelVersion = getModelVersion();
  const trace = createTrace("predict", [`model-version:${modelVersion}`]);

  try {
    // 1. Parse multipart form
    const formData = await request.formData();
    const audioFile = formData.get("audio");

    if (!audioFile || !(audioFile instanceof File)) {
      trace?.update({ output: "No audio file provided" });
      await flushTraces();
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
      trace?.update({ output: "Non-WAV file rejected" });
      await flushTraces();
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
      trace?.update({ output: "File too large" });
      await flushTraces();
      return NextResponse.json(
        { error: "Audio file too large. Maximum size is 20 MB." },
        { status: 400 }
      );
    }

    // 3. Read the file buffer
    const buffer = Buffer.from(await audioFile.arrayBuffer());

    // 4. Decode WAV to PCM samples
    const decodeSpan = trace?.span({
      name: "decode-wav",
      input: {
        fileName: audioFile.name,
        fileSizeBytes: audioFile.size,
      },
    });

    let audioData: wavDecoder.AudioData;
    try {
      audioData = await wavDecoder.decode(buffer);
    } catch {
      decodeSpan?.end({ output: "WAV decode failed" });
      trace?.update({ output: "WAV decode failed" });
      await flushTraces();
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

    decodeSpan?.end({
      output: {
        sampleRate: audioData.sampleRate,
        numChannels: audioData.channelData.length,
        numSamples: samples.length,
        durationSec,
      },
    });

    if (durationSec < 0.1) {
      trace?.update({ output: "Audio too short" });
      await flushTraces();
      return NextResponse.json(
        { error: "Audio is too short (less than 0.1 seconds)." },
        { status: 400 }
      );
    }

    // 5. Run inference (spectrogram expects 22050 Hz)
    const inferenceSpan = trace?.span({
      name: "onnx-inference",
      input: {
        durationSec,
        sampleRate: 22050,
      },
    });

    const result = await predict(samples, 22050, durationSec);

    inferenceSpan?.end({
      output: {
        emotion: result.emotion,
        confidence: result.confidence,
        probabilities: result.probabilities,
        embeddingNorm: Math.sqrt(
          result.embedding.reduce((s, v) => s + v * v, 0)
        ),
      },
    });

    // 6. Return result
    trace?.update({ output: result.emotion });
    await flushTraces();
    return NextResponse.json(result, { status: 200 });
  } catch (error) {
    console.error("Prediction error:", error);
    const message =
      error instanceof Error ? error.message : "Unknown error during prediction";

    trace?.update({ output: message });
    await flushTraces();
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
