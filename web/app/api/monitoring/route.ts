/**
 * GET /api/monitoring
 *
 * Queries the Langfuse API for recent "predict" traces and aggregates
 * metrics per model version  (model-version:<tag>), returning:
 *
 *   - Overview          — total predictions, versions, avg latency, error rate
 *   - Versions          — per-version count, latency stats, confidence, error rate
 *   - Time series       — daily breakdown per version (last 7 days)
 *   - Recent errors     — last 20 failed traces with details
 *
 * This route is used by the /monitoring dashboard page.
 *
 * Environment variables required (besides LANGFUSE_*):
 *   LANGFUSE_SECRET_KEY  — required to authenticate with the Langfuse API
 *   LANGFUSE_PUBLIC_KEY  — required (used for Basic Auth)
 *   LANGFUSE_BASE_URL    — optional (defaults to https://cloud.langfuse.com)
 */

import { NextResponse } from "next/server";

// ── Types ────────────────────────────────────────────────────────────────────

interface VersionMetrics {
  count: number;
  avgLatencyMs: number;
  p50LatencyMs: number;
  p95LatencyMs: number;
  minLatencyMs: number;
  maxLatencyMs: number;
  errorCount: number;
  errorRate: number;
  avgConfidence: number;
  topEmotion: string;
}

interface TimeSeriesPoint {
  date: string;
  version: string;
  count: number;
  avgLatencyMs: number;
  avgConfidence: number;
  errorCount: number;
}

interface ErrorTrace {
  id: string;
  timestamp: string;
  modelVersion: string;
  error: string;
  durationMs: number;
}

interface MonitoringResponse {
  overview: {
    totalPredictions: number;
    totalVersions: number;
    avgLatencyMs: number;
    errorRate: number;
    avgConfidence: number;
    lastUpdated: string;
  };
  versions: Record<string, VersionMetrics>;
  timeSeries: TimeSeriesPoint[];
  recentErrors: ErrorTrace[];
}

interface TraceObservation {
  name?: string;
  startTime?: string;
  endTime?: string;
  output?: unknown;
  level?: string;
}

interface TraceItem {
  id: string;
  name?: string;
  timestamp?: string;
  tags?: string[];
  input?: unknown;
  output?: unknown;
  metadata?: unknown;
  observations?: TraceObservation[];
}

interface TracesResponse {
  data?: TraceItem[];
  meta?: { page?: number; limit?: number; totalItems?: number };
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Extract the model version from a trace's tags array.
 * Tags follow the convention "model-version:<version>".
 */
function extractModelVersion(tags?: string[]): string | null {
  if (!tags) return null;
  for (const tag of tags) {
    if (tag.startsWith("model-version:")) {
      return tag.slice("model-version:".length);
    }
  }
  return null;
}

/**
 * Compute the duration of a trace in milliseconds based on observations.
 * Falls back to the trace timestamp if no observations are available.
 */
function computeDurationMs(
  trace: TraceItem,
): number {
  const observations = trace.observations ?? [];
  let earliestStart: number | null = null;
  let latestEnd: number | null = null;

  for (const obs of observations) {
    if (obs.startTime) {
      const t = new Date(obs.startTime).getTime();
      if (earliestStart === null || t < earliestStart) earliestStart = t;
    }
    if (obs.endTime) {
      const t = new Date(obs.endTime).getTime();
      if (latestEnd === null || t > latestEnd) latestEnd = t;
    }
  }

  if (earliestStart !== null && latestEnd !== null) {
    return latestEnd - earliestStart;
  }

  // Fallback: treat trace timestamp as single point
  return 0;
}

/**
 * Extract confidence from the "onnx-inference" span output.
 */
function extractConfidence(trace: TraceItem): number | null {
  const observations = trace.observations ?? [];
  for (const obs of observations) {
    if (obs.name === "onnx-inference") {
      const output = obs.output as Record<string, unknown> | undefined;
      if (output && typeof output.confidence === "number") {
        return output.confidence;
      }
    }
  }
  return null;
}

/**
 * Determine whether a trace represents an error.
 * Checks: ERROR-level observations, or output containing an error message.
 */
function isErrorTrace(trace: TraceItem): boolean {
  // Check for ERROR-level observations
  const observations = trace.observations ?? [];
  for (const obs of observations) {
    if (obs.level === "ERROR") return true;
  }

  // Check if output is a string (error message) or an object with a "WAV decode failed" pattern
  const output = trace.output;
  if (typeof output === "string") return true;
  if (typeof output === "object" && output !== null) {
    const out = output as Record<string, unknown>;
    if (typeof out.error === "string") return true;
  }

  return false;
}

/**
 * Extract the error message from a failed trace.
 */
function extractError(trace: TraceItem): string {
  // Prefer ERROR-level observation output
  const observations = trace.observations ?? [];
  for (const obs of observations) {
    if (obs.level === "ERROR") {
      if (typeof obs.output === "string") return obs.output;
      if (typeof obs.output === "object" && obs.output !== null) {
        const o = obs.output as Record<string, unknown>;
        if (typeof o.error === "string") return o.error;
      }
    }
  }

  // Fall back to trace output
  if (typeof trace.output === "string") return trace.output;
  if (typeof trace.output === "object" && trace.output !== null) {
    const o = trace.output as Record<string, unknown>;
    if (typeof o.error === "string") return o.error;
  }

  return "Unknown error";
}

/**
 * Compute percentiles from a sorted array of numbers.
 */
function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const index = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, Math.min(index, sorted.length - 1))];
}

// ── Route handler ────────────────────────────────────────────────────────────

export async function GET() {
  // 1. Validate Langfuse configuration
  const secretKey = process.env.LANGFUSE_SECRET_KEY;
  const publicKey = process.env.LANGFUSE_PUBLIC_KEY;
  const baseUrl =
    process.env.LANGFUSE_BASE_URL ?? "https://cloud.langfuse.com";

  if (!secretKey || !publicKey) {
    return NextResponse.json(
      {
        configured: false as const,
        error:
          "Langfuse not configured. Set LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY.",
      },
      { status: 200 },
    );
  }

  // 2. Fetch recent traces from Langfuse API
  const auth = Buffer.from(`${publicKey}:${secretKey}`).toString("base64");

  // Fetch the last 7 days of traces, up to 500
  const fromTimestamp = new Date(
    Date.now() - 7 * 24 * 60 * 60 * 1000,
  ).toISOString();

  const params = new URLSearchParams({
    name: "predict",
    fromTimestamp,
    limit: "100",
    fields: "core,observations,io,metrics",
    orderBy: "timestamp.desc",
  });

  let traces: TraceItem[] = [];
  try {
    const response = await fetch(
      `${baseUrl}/api/public/traces?${params}`,
      {
        headers: {
          Authorization: `Basic ${auth}`,
          "Content-Type": "application/json",
        },
        next: { revalidate: 60 }, // cache for 60 seconds
      },
    );

    if (!response.ok) {
      const body = await response.text().catch(() => "");
      return NextResponse.json(
        {
          configured: true as const,
          error: `Langfuse API error: HTTP ${response.status} — ${body}`,
        },
        { status: 502 },
      );
    }

    const result: TracesResponse = await response.json();
    traces = result.data ?? [];
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Failed to fetch Langfuse traces";
    return NextResponse.json(
      { configured: true as const, error: message },
      { status: 502 },
    );
  }

  // 3. Group by model version
  const versionBuckets = new Map<
    string,
    { traces: TraceItem[]; latencies: number[] }
  >();
  const unknownTraces: TraceItem[] = [];

  for (const trace of traces) {
    const version = extractModelVersion(trace.tags);
    const key = version ?? "__unknown__";
    const bucket = versionBuckets.get(key) ?? { traces: [], latencies: [] };
    bucket.traces.push(trace);
    versionBuckets.set(key, bucket);
  }

  // 4. Build per-version metrics
  const versions: Record<string, VersionMetrics> = {};
  const allLatencies: number[] = [];
  let totalErrors = 0;
  let totalConfSum = 0;
  let totalConfCount = 0;
  const emotionFrequency: Record<string, number> = {};

  for (const [version, bucket] of versionBuckets) {
    const displayVersion =
      version === "__unknown__" ? "unknown" : version;
    const latencies: number[] = [];
    let errorCount = 0;
    let confSum = 0;
    let confCount = 0;

    for (const trace of bucket.traces) {
      const durMs = computeDurationMs(trace);
      latencies.push(durMs);
      allLatencies.push(durMs);

      if (isErrorTrace(trace)) {
        errorCount++;
        totalErrors++;
      }

      const conf = extractConfidence(trace);
      if (conf !== null) {
        confSum += conf;
        confCount++;
        totalConfSum += conf;
        totalConfCount++;
      }
    }

    const sorted = [...latencies].sort((a, b) => a - b);
    const count = bucket.traces.length;
    const avgLatencyMs =
      count > 0
        ? sorted.reduce((a, b) => a + b, 0) / count
        : 0;

    versions[displayVersion] = {
      count,
      avgLatencyMs: Math.round(avgLatencyMs * 10) / 10,
      p50LatencyMs: Math.round(percentile(sorted, 50) * 10) / 10,
      p95LatencyMs: Math.round(percentile(sorted, 95) * 10) / 10,
      minLatencyMs: count > 0 ? Math.round(sorted[0] * 10) / 10 : 0,
      maxLatencyMs:
        count > 0
          ? Math.round(sorted[sorted.length - 1] * 10) / 10
          : 0,
      errorCount,
      errorRate: count > 0 ? errorCount / count : 0,
      avgConfidence: confCount > 0 ? confSum / confCount : 0,
      topEmotion: getTopEmotion(bucket.traces),
    };
  }

  // 5. Build time series (daily buckets per version)
  const timeSeries = buildTimeSeries(traces);

  // 6. Collect recent errors
  const recentErrors: ErrorTrace[] = [];
  for (const trace of traces) {
    if (recentErrors.length >= 20) break;
    if (isErrorTrace(trace)) {
      recentErrors.push({
        id: trace.id,
        timestamp: trace.timestamp ?? "",
        modelVersion: extractModelVersion(trace.tags) ?? "unknown",
        error: extractError(trace),
        durationMs: computeDurationMs(trace),
      });
    }
  }

  const versionKeys = Object.keys(versions);
  const sortedAllLatencies = [...allLatencies].sort((a, b) => a - b);

  const response: MonitoringResponse = {
    overview: {
      totalPredictions: traces.length,
      totalVersions: versionKeys.length,
      avgLatencyMs:
        allLatencies.length > 0
          ? Math.round(
              (allLatencies.reduce((a, b) => a + b, 0) /
                allLatencies.length) *
                10,
            ) / 10
          : 0,
      errorRate:
        traces.length > 0 ? totalErrors / traces.length : 0,
      avgConfidence:
        totalConfCount > 0 ? totalConfSum / totalConfCount : 0,
      lastUpdated: new Date().toISOString(),
    },
    versions,
    timeSeries,
    recentErrors,
  };

  return NextResponse.json(response, { status: 200 });
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/**
 * Get the most frequent emotion from a set of traces.
 */
function getTopEmotion(traces: TraceItem[]): string {
  const freq: Record<string, number> = {};
  for (const trace of traces) {
    const observations = trace.observations ?? [];
    for (const obs of observations) {
      if (obs.name === "onnx-inference") {
        const output = obs.output as Record<string, unknown> | undefined;
        if (output && typeof output.emotion === "string") {
          freq[output.emotion] = (freq[output.emotion] ?? 0) + 1;
        }
      }
    }
  }

  let top = "";
  let topCount = 0;
  for (const [emotion, count] of Object.entries(freq)) {
    if (count > topCount) {
      top = emotion;
      topCount = count;
    }
  }
  return top || "—";
}

/**
 * Build daily time series data, grouped by model version.
 */
function buildTimeSeries(traces: TraceItem[]): TimeSeriesPoint[] {
  // Create date buckets for the last 7 days
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  // Map<dateStr, Map<version, { count, latencySum, confSum, errorCount }>>
  const daily = new Map<
    string,
    Map<
      string,
      {
        count: number;
        latencySum: number;
        confSum: number;
        confCount: number;
        errorCount: number;
      }
    >
  >();

  // Initialize empty buckets for the last 7 days
  for (let i = 6; i >= 0; i--) {
    const d = new Date(today);
    d.setDate(d.getDate() - i);
    const key = d.toISOString().slice(0, 10);
    daily.set(key, new Map());
  }

  // Fill from trace data
  for (const trace of traces) {
    const traceDate = trace.timestamp
      ? new Date(trace.timestamp).toISOString().slice(0, 10)
      : null;
    if (!traceDate) continue;

    const day = daily.get(traceDate);
    if (!day) continue; // outside the 7-day window

    const version = extractModelVersion(trace.tags) ?? "unknown";
    let bucket = day.get(version);
    if (!bucket) {
      bucket = {
        count: 0,
        latencySum: 0,
        confSum: 0,
        confCount: 0,
        errorCount: 0,
      };
      day.set(version, bucket);
    }

    bucket.count++;
    bucket.latencySum += computeDurationMs(trace);
    if (isErrorTrace(trace)) bucket.errorCount++;

    const conf = extractConfidence(trace);
    if (conf !== null) {
      bucket.confSum += conf;
      bucket.confCount++;
    }
  }

  // Flatten into sorted array
  const result: TimeSeriesPoint[] = [];
  for (const [date, versionsMap] of daily) {
    for (const [version, data] of versionsMap) {
      if (data.count === 0) continue;
      result.push({
        date,
        version,
        count: data.count,
        avgLatencyMs:
          data.count > 0
            ? Math.round((data.latencySum / data.count) * 10) / 10
            : 0,
        avgConfidence:
          data.confCount > 0
            ? data.confSum / data.confCount
            : 0,
        errorCount: data.errorCount,
      });
    }
  }

  // Sort by date, then version
  result.sort((a, b) => {
    if (a.date !== b.date) return a.date.localeCompare(b.date);
    return a.version.localeCompare(b.version);
  });

  return result;
}
