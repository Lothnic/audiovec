/**
 * inference.test.ts — Unit tests for the ONNX inference wrapper.
 *
 * Covers:
 *   - Model resolution (env var, /tmp cache, download)
 *   - Download fallback (success, HTTP error, network error)
 *   - ONNX session singleton pattern
 *   - Prediction: softmax, emotion selection, spectrogram reshape
 *   - Error propagation
 *
 * NOTE: We use dynamic import() within beforeEach rather than top-level
 * imports because inference.ts has a module-level `_session` singleton.
 * vitest's mockReset (handled manually here) would clear the session's
 * internal mock without resetting _session itself — causing stale references
 * between tests. Dynamic re-imports give us a fresh module per test suite.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// ── Module mocks (hoisted before ALL imports) ────────────────────────────────
// These replace the modules before any code imports them.

vi.mock("fs", () => {
  const mockFs = {
    existsSync: vi.fn(),
    mkdirSync: vi.fn(),
    writeFileSync: vi.fn(),
  };
  return { ...mockFs, default: mockFs };
});

vi.mock("onnxruntime-node", () => ({
  InferenceSession: { create: vi.fn() },
  Tensor: vi.fn().mockImplementation(function(
    this: any,
    type: string,
    data: Float32Array,
    dims: number[]
  ) {
    return { type, data, dims };
  }),
}));

vi.mock("../melspectrogram", () => ({
  audioToSpectrogram: vi.fn(),
}));

// ── Imports (resolved via hoisted mocks) ─────────────────────────────────────
import fs from "fs";
import * as ort from "onnxruntime-node";
import { audioToSpectrogram } from "../melspectrogram";
import { EMOTIONS } from "../emotions";

// ── Helpers ──────────────────────────────────────────────────────────────────

const FAKE_SPECTROGRAM = new Float64Array(128 * 228).fill(0.5);

function makeMockSession(overrides?: {
  logits?: Float32Array;
  embedding?: Float32Array;
}) {
  const logits = overrides?.logits ?? new Float32Array(8).fill(0);
  const embedding = overrides?.embedding ?? new Float32Array(256).fill(0.05);
  return {
    run: vi.fn().mockResolvedValue({
      logits: { data: logits },
      embedding: { data: embedding },
    }),
    release: vi.fn(),
    startProfiling: vi.fn(),
    endProfiling: vi.fn(),
    inputNames: ["spectrogram"],
    outputNames: ["logits", "embedding"],
    inputMetadata: [{ name: "spectrogram" }] as any,
    outputMetadata: [{ name: "logits" }, { name: "embedding" }] as any,
  };
}

function makeFetchResponse(ok: boolean, data?: ArrayBuffer) {
  return {
    ok,
    status: ok ? 200 : 500,
    statusText: ok ? "OK" : "Internal Server Error",
    arrayBuffer: () => Promise.resolve(data ?? new ArrayBuffer(100)),
  };
}

// ── Shared test setup ────────────────────────────────────────────────────────


// ── Test suite ───────────────────────────────────────────────────────────────

// Since we need a fresh module per describe block (to reset _session),
// we declare suite-scoped variables and re-import in beforeEach.
let predict: typeof import("../inference")["predict"];

describe("predict() — model resolution", () => {
  const mockSession = makeMockSession();
  let originalModelPath: string | undefined;

  beforeEach(async () => {
    vi.clearAllMocks();
    vi.unstubAllGlobals();
    vi.resetModules();

    // Re-import to get a fresh module (resets _session = null)
    predict = (await import("../inference")).predict;

    originalModelPath = process.env.MODEL_PATH;
    vi.mocked(audioToSpectrogram).mockReturnValue({
      data: FAKE_SPECTROGRAM,
      nMels: 128,
      nFrames: 228,
    });
    // Note: fs, fetch, and InferenceSession.create are set per-test below
    // to avoid interfering with each other.
  });

  afterEach(() => {
    delete process.env.MODEL_PATH;
    if (originalModelPath !== undefined) {
      process.env.MODEL_PATH = originalModelPath;
    }
    vi.unstubAllGlobals();
  });

  // ── MODEL_PATH env var ─────────────────────────────────────────────────

  it("uses MODEL_PATH env var when set", async () => {
    process.env.MODEL_PATH = "/custom/path/model.onnx";
    vi.mocked(fs.existsSync).mockImplementation(
      (p: fs.PathLike) => p === "/custom/path/model.onnx"
    );
    vi.mocked(ort.InferenceSession.create).mockResolvedValue(mockSession);

    const samples = new Float32Array(16000);
    await predict(samples, 22050, 1.0);

    expect(ort.InferenceSession.create).toHaveBeenCalledWith(
      "/custom/path/model.onnx"
    );
  });

  // ── Download path ──────────────────────────────────────────────────────

  it("downloads model files when no local path exists", async () => {
    vi.mocked(fs.existsSync).mockReturnValue(false);
    const mockFetch = vi.fn().mockResolvedValue(makeFetchResponse(true));
    vi.stubGlobal("fetch", mockFetch);
    vi.mocked(ort.InferenceSession.create).mockResolvedValue(mockSession);

    const samples = new Float32Array(16000);
    await predict(samples, 22050, 1.0);

    // Should have created the cache directory
    expect(fs.mkdirSync).toHaveBeenCalledWith("/tmp/models", {
      recursive: true,
    });
    // Should have downloaded both files
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it("downloads both .onnx and .onnx.data files", async () => {
    vi.mocked(fs.existsSync).mockReturnValue(false);
    const mockFetch = vi.fn().mockResolvedValue(makeFetchResponse(true));
    vi.stubGlobal("fetch", mockFetch);
    vi.mocked(ort.InferenceSession.create).mockResolvedValue(mockSession);

    const samples = new Float32Array(16000);
    await predict(samples, 22050, 1.0);

    expect(mockFetch.mock.calls[0][0]).toContain("crnn-transformer.onnx");
    expect(mockFetch.mock.calls[1][0]).toContain("crnn-transformer.onnx.data");
  });

  // ── /tmp cache ─────────────────────────────────────────────────────────

  it("uses cached model in /tmp when available", async () => {
    vi.mocked(fs.existsSync).mockImplementation(
      (p: fs.PathLike) => p === "/tmp/models/crnn-transformer.onnx"
    );
    vi.mocked(ort.InferenceSession.create).mockResolvedValue(mockSession);

    const samples = new Float32Array(16000);
    await predict(samples, 22050, 1.0);

    // Should NOT try to download (no fetch stub → would throw if called)
    expect(ort.InferenceSession.create).toHaveBeenCalledWith(
      "/tmp/models/crnn-transformer.onnx"
    );
  });

  // ── Download failures ──────────────────────────────────────────────────

  it("throws when download fails with HTTP error", async () => {
    vi.mocked(fs.existsSync).mockReturnValue(false);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(makeFetchResponse(false))
    );

    const samples = new Float32Array(16000);
    await expect(predict(samples, 22050, 1.0)).rejects.toThrow(
      /Failed to download/
    );
  });

  it("throws when network fails", async () => {
    vi.mocked(fs.existsSync).mockReturnValue(false);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new Error("Network error"))
    );

    const samples = new Float32Array(16000);
    await expect(predict(samples, 22050, 1.0)).rejects.toThrow(
      "Network error"
    );
  });
});

// ── Session singleton ─────────────────────────────────────────────────────────

describe("predict() — session singleton", () => {
  const mockSession = makeMockSession();

  beforeEach(async () => {
    vi.clearAllMocks();
    vi.unstubAllGlobals();
    vi.resetModules();
    predict = (await import("../inference")).predict;

    vi.mocked(audioToSpectrogram).mockReturnValue({
      data: FAKE_SPECTROGRAM,
      nMels: 128,
      nFrames: 228,
    });
    vi.mocked(fs.existsSync).mockReturnValue(false);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(makeFetchResponse(true))
    );
    vi.mocked(ort.InferenceSession.create).mockResolvedValue(mockSession);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("creates ONNX session only once across multiple sequential calls", async () => {
    const samples = new Float32Array(16000);

    await predict(samples, 22050, 1.0);
    await predict(samples, 22050, 2.0);

    expect(ort.InferenceSession.create).toHaveBeenCalledTimes(1);
  });
});

// ── Prediction logic ─────────────────────────────────────────────────────────

describe("predict() — prediction logic", () => {
  const mockSession = makeMockSession();

  beforeEach(async () => {
    vi.clearAllMocks();
    vi.unstubAllGlobals();
    vi.resetModules();
    predict = (await import("../inference")).predict;

    vi.mocked(audioToSpectrogram).mockReturnValue({
      data: FAKE_SPECTROGRAM,
      nMels: 128,
      nFrames: 228,
    });
    vi.mocked(fs.existsSync).mockReturnValue(false);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(makeFetchResponse(true))
    );
    vi.mocked(ort.InferenceSession.create).mockResolvedValue(mockSession);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  // ── Result structure ───────────────────────────────────────────────────

  it("returns the correct result structure", async () => {
    const samples = new Float32Array(16000);
    const result = await predict(samples, 22050, 3.5);

    expect(result).toHaveProperty("emotion");
    expect(result).toHaveProperty("confidence");
    expect(result).toHaveProperty("probabilities");
    expect(result).toHaveProperty("embedding");
    expect(result).toHaveProperty("duration");
    expect(result).toHaveProperty("spectrogram");
  });

  it("includes duration in result", async () => {
    const samples = new Float32Array(16000);
    const result = await predict(samples, 22050, 2.5);
    expect(result.duration).toBe(2.5);
  });

  it("returns all 8 emotions in probabilities", async () => {
    const samples = new Float32Array(16000);
    const result = await predict(samples, 22050, 1.0);
    expect(Object.keys(result.probabilities).sort()).toEqual(
      [...EMOTIONS].sort()
    );
  });

  // ── Softmax & emotion selection ────────────────────────────────────────

  it("selects the emotion with highest probability", async () => {
    const logits = new Float32Array(8);
    logits[0] = 0.5;
    logits[1] = 0.3;
    logits[2] = 10.0; // ← highest (happy)
    logits[3] = 0.2;
    logits[4] = 0.1;
    logits[5] = 0.4;
    logits[6] = 0.0;
    logits[7] = 0.8;

    vi.mocked(ort.InferenceSession.create).mockResolvedValue(
      makeMockSession({
        logits,
        embedding: new Float32Array(256).fill(0.05),
      })
    );

    const samples = new Float32Array(16000);
    const result = await predict(samples, 22050, 1.0);

    expect(result.emotion).toBe("happy");
    expect(result.confidence).toBeGreaterThan(0.5);
  });

  it("probabilities sum to ~1.0", async () => {
    const logits = new Float32Array([1.2, 0.5, -0.3, 2.1, 0.0, -1.5, 0.8, 0.3]);

    vi.mocked(ort.InferenceSession.create).mockResolvedValue(
      makeMockSession({
        logits,
        embedding: new Float32Array(256).fill(0.05),
      })
    );

    const samples = new Float32Array(16000);
    const result = await predict(samples, 22050, 1.0);

    const sum = Object.values(result.probabilities).reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1.0, 5);
  });

  // ── Embedding ──────────────────────────────────────────────────────────

  it("returns 256-element embedding vector", async () => {
    const embedding = new Float32Array(256);
    for (let i = 0; i < 256; i++) {
      embedding[i] = Math.sin(i * 0.1);
    }

    vi.mocked(ort.InferenceSession.create).mockResolvedValue(
      makeMockSession({
        logits: new Float32Array(8).fill(0),
        embedding,
      })
    );

    const samples = new Float32Array(16000);
    const result = await predict(samples, 22050, 1.0);

    expect(result.embedding).toHaveLength(256);
    expect(result.embedding[0]).toBeCloseTo(0, 1);
  });

  // ── Spectrogram ────────────────────────────────────────────────────────

  it("returns 128x228 spectrogram matrix", async () => {
    const samples = new Float32Array(16000);
    const result = await predict(samples, 22050, 1.0);

    expect(result.spectrogram).toHaveLength(128);
    expect(result.spectrogram[0]).toHaveLength(228);
  });
});

// ── Audio input edge cases ────────────────────────────────────────────────────

describe("predict() — audio input edge cases", () => {
  const mockSession = makeMockSession();

  beforeEach(async () => {
    vi.clearAllMocks();
    vi.unstubAllGlobals();
    vi.resetModules();
    predict = (await import("../inference")).predict;

    vi.mocked(audioToSpectrogram).mockReturnValue({
      data: FAKE_SPECTROGRAM,
      nMels: 128,
      nFrames: 228,
    });
    vi.mocked(fs.existsSync).mockReturnValue(false);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(makeFetchResponse(true))
    );
    vi.mocked(ort.InferenceSession.create).mockResolvedValue(mockSession);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("handles Float32Array input", async () => {
    const samples = new Float32Array(8000);
    const result = await predict(samples, 22050, 0.5);
    expect(result.emotion).toBeDefined();
    expect(typeof result.confidence).toBe("number");
  });

  it("handles very short audio", async () => {
    vi.mocked(audioToSpectrogram).mockReturnValue({
      data: new Float64Array(128 * 228).fill(0.1),
      nMels: 128,
      nFrames: 228,
    });

    const samples = new Float32Array(100);
    const result = await predict(samples, 22050, 0.05);
    expect(result.emotion).toBeDefined();
  });

  it("handles silent audio (all zeros)", async () => {
    vi.mocked(audioToSpectrogram).mockReturnValue({
      data: new Float64Array(128 * 228).fill(0),
      nMels: 128,
      nFrames: 228,
    });

    const samples = new Float32Array(16000);
    const result = await predict(samples, 22050, 1.0);
    expect(result.emotion).toBeDefined();
    expect(EMOTIONS).toContain(result.emotion);
  });
});
