/**
 * langfuse.ts — Langfuse observability client.
 *
 * Provides a singleton Langfuse instance for tracing ONNX inference calls.
 * The client is initialized lazily and survives warm starts.
 *
 * Environment variables:
 *   LANGFUSE_SECRET_KEY   — required
 *   LANGFUSE_PUBLIC_KEY   — required
 *   LANGFUSE_BASE_URL     — optional (defaults to https://cloud.langfuse.com)
 *
 * If the required env vars are missing, the client acts as a no-op
 * so the app works without monitoring configured.
 */

import { Langfuse } from "langfuse";
import type { LangfuseTraceClient } from "langfuse";

// ── Singleton ────────────────────────────────────────────────────────────────

let _client: Langfuse | null = null;

function getClient(): Langfuse | null {
  if (_client) return _client;

  const secretKey = process.env.LANGFUSE_SECRET_KEY;
  const publicKey = process.env.LANGFUSE_PUBLIC_KEY;

  if (!secretKey || !publicKey) {
    // Monitoring not configured — no-op
    return null;
  }

  _client = new Langfuse({
    secretKey,
    publicKey,
    baseUrl: process.env.LANGFUSE_BASE_URL ?? "https://cloud.langfuse.com",
  });

  return _client;
}

// ── Trace helpers ────────────────────────────────────────────────────────────

/**
 * Create a new trace. Returns null if Langfuse is not configured.
 *
 * @param name     - Trace name (e.g. "predict")
 * @param tags     - Optional tags for filtering in the Langfuse UI.
 *                   Pass `["model-version:v1.2.3"]` to segment traces
 *                   by model deployment.
 */
export function createTrace(
  name: string,
  tags?: string[],
): LangfuseTraceClient | null {
  return getClient()?.trace({ name, tags }) ?? null;
}

/**
 * Flush all pending events without shutting down the client.
 * Must be called before the serverless function terminates, otherwise
 * buffered events may be lost.
 */
export async function flushTraces(): Promise<void> {
  if (_client) {
    await _client.flushAsync();
  }
}
