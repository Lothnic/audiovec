"use client";

/**
 * /monitoring — Langfuse tracing dashboard.
 *
 * Displays per-model-version metrics aggregated from the Langfuse API:
 * predictions count, latency, confidence, and error rates.
 *
 * Data is fetched from /api/monitoring which proxies Langfuse's /api/public/traces.
 */

import { useCallback, useEffect, useState } from "react";

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

// ── Helpers ──────────────────────────────────────────────────────────────────

function fmtMs(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function fmtPct(rate: number): string {
  return `${(rate * 100).toFixed(1)}%`;
}

function fmtConfidence(c: number): string {
  return `${(c * 100).toFixed(0)}%`;
}

function fmtTime(iso: string): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function fmtShortDate(iso: string): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

// ── SVG Bar Chart ────────────────────────────────────────────────────────────

function BarChart({
  data,
  valueKey,
  labelKey,
  barColor,
  height = 180,
  formatValue = (v: number) => v.toFixed(0),
}: {
  data: { label: string; value: number }[];
  valueKey?: string;
  labelKey?: string;
  barColor: string;
  height?: number;
  formatValue?: (v: number) => string;
}) {
  if (data.length === 0) return null;

  const maxValue = Math.max(...data.map((d) => d.value), 1);
  const barWidth = Math.max(24, Math.min(60, 480 / data.length));
  const gap = 16;
  const chartWidth = data.length * (barWidth + gap) + 40;

  return (
    <svg
      viewBox={`0 0 ${chartWidth} ${height + 30}`}
      className="w-full max-w-full"
      style={{ height }}
      preserveAspectRatio="xMidYMid meet"
    >
      {/* Y-axis grid */}
      {[0, 0.25, 0.5, 0.75, 1].map((f) => {
        const y = height - f * height;
        return (
          <g key={f}>
            <line
              x1={0}
              y1={y}
              x2={chartWidth}
              y2={y}
              stroke="#2A2F3A"
              strokeWidth={1}
            />
            <text
              x={chartWidth - 4}
              y={y + 4}
              fill="#6B7280"
              fontSize={10}
              textAnchor="end"
            >
              {formatValue(f * maxValue)}
            </text>
          </g>
        );
      })}

      {/* Bars */}
      {data.map((d, i) => {
        const barH = (d.value / maxValue) * (height - 4);
        const x = 20 + i * (barWidth + gap);
        const y = height - barH;
        return (
          <g key={d.label}>
            <rect
              x={x}
              y={y}
              width={barWidth}
              height={barH}
              fill={barColor}
              rx={4}
              opacity={0.85}
              className="transition-opacity hover:opacity-100"
            />
            <text
              x={x + barWidth / 2}
              y={height + 14}
              fill="#9CA3AF"
              fontSize={10}
              textAnchor="middle"
              className="truncate"
            >
              {d.label.length > 8 ? d.label.slice(0, 8) + "…" : d.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ── Stat Card ────────────────────────────────────────────────────────────────

function StatCard({
  label,
  value,
  subtitle,
  accent = false,
}: {
  label: string;
  value: string;
  subtitle?: string;
  accent?: boolean;
}) {
  return (
    <div
      className={`rounded-xl p-4 sm:p-5 border transition-shadow ${
        accent
          ? "border-accent-purple/30 bg-accent-purple/[0.06]"
          : "border-surface-border/40 bg-[#161B22]"
      }`}
    >
      <div className="text-xs font-medium text-text-muted uppercase tracking-wider">
        {label}
      </div>
      <div className="mt-1 text-2xl sm:text-3xl font-bold tabular-nums text-white">
        {value}
      </div>
      {subtitle && (
        <div className="mt-0.5 text-xs text-text-dim">{subtitle}</div>
      )}
    </div>
  );
}

// ── Version Row ──────────────────────────────────────────────────────────────

function VersionRow({
  version,
  metrics,
  maxCount,
}: {
  version: string;
  metrics: VersionMetrics;
  maxCount: number;
}) {
  const barW = maxCount > 0 ? (metrics.count / maxCount) * 100 : 0;

  return (
    <div className="grid grid-cols-[1fr_auto_auto_auto_auto_auto] gap-2 sm:gap-4 items-center py-2.5 border-b border-surface-border/20 last:border-0 text-sm">
      {/* Version name + bar */}
      <div className="min-w-0">
        <div className="font-medium text-white truncate">{version}</div>
        <div className="mt-1 h-1.5 rounded-full bg-surface-border/20 overflow-hidden max-w-[160px]">
          <div
            className="h-full rounded-full bg-accent-purple"
            style={{ width: `${barW}%` }}
          />
        </div>
      </div>

      {/* Count */}
      <div className="tabular-nums text-right text-text-muted w-10">
        {metrics.count}
      </div>

      {/* Latency */}
      <div className="tabular-nums text-right text-text-muted w-16">
        {fmtMs(metrics.avgLatencyMs)}
      </div>

      {/* P95 Latency */}
      <div className="tabular-nums text-right text-text-muted w-16 hidden sm:block">
        {fmtMs(metrics.p95LatencyMs)}
      </div>

      {/* Confidence */}
      <div
        className={`tabular-nums text-right w-12 ${
          metrics.avgConfidence > 0.7
            ? "text-emerald-400"
            : metrics.avgConfidence > 0.4
              ? "text-amber-400"
              : "text-red-400"
        }`}
      >
        {fmtConfidence(metrics.avgConfidence)}
      </div>

      {/* Error rate */}
      <div
        className={`tabular-nums text-right w-12 ${
          metrics.errorRate > 0.1
            ? "text-red-400"
            : metrics.errorRate > 0
              ? "text-amber-400"
              : "text-emerald-400"
        }`}
      >
        {fmtPct(metrics.errorRate)}
      </div>
    </div>
  );
}

// ── Time Series Chart ────────────────────────────────────────────────────────

function TimeSeriesChart({
  data,
  selectedMetric,
}: {
  data: TimeSeriesPoint[];
  selectedMetric: "count" | "avgLatencyMs";
}) {
  if (data.length === 0) return null;

  // Group by date, then by version
  const dateOrder = [...new Set(data.map((d) => d.date))].sort();
  const versions = [...new Set(data.map((d) => d.version))];
  const versionColors = [
    "#A78BFA",
    "#34D399",
    "#F87171",
    "#60A5FA",
    "#FBBF24",
    "#F472B6",
  ];

  const maxValue = Math.max(
    ...data.map((d) =>
      selectedMetric === "count" ? d.count : d.avgLatencyMs,
    ),
    1,
  );

  const chartH = 200;
  const chartW = Math.max(dateOrder.length * 60, 300);
  const padding = { top: 10, bottom: 30, left: 40, right: 16 };

  const getX = (i: number) =>
    padding.left +
    (i / Math.max(dateOrder.length - 1, 1)) * (chartW - padding.left - padding.right);

  return (
    <svg
      viewBox={`0 0 ${chartW} ${chartH + padding.bottom + padding.top}`}
      className="w-full max-w-full"
      style={{ height: chartH + padding.bottom + padding.top }}
      preserveAspectRatio="xMidYMid meet"
    >
      {/* Y-axis grid */}
      {[0, 0.25, 0.5, 0.75, 1].map((f) => {
        const y = padding.top + (1 - f) * chartH;
        return (
          <g key={f}>
            <line
              x1={padding.left}
              y1={y}
              x2={chartW - padding.right}
              y2={y}
              stroke="#2A2F3A"
              strokeWidth={1}
            />
            <text
              x={padding.left - 8}
              y={y + 4}
              fill="#6B7280"
              fontSize={10}
              textAnchor="end"
            >
              {selectedMetric === "count"
                ? Math.round(f * maxValue)
                : fmtMs(f * maxValue)}
            </text>
          </g>
        );
      })}

      {/* Lines per version */}
      {versions.map((version, vi) => {
        const color = versionColors[vi % versionColors.length];
        const points = dateOrder.map((date, i) => {
          const point = data.find(
            (d) => d.date === date && d.version === version,
          );
          const val =
            selectedMetric === "count"
              ? point?.count ?? 0
              : point?.avgLatencyMs ?? 0;
          const x = getX(i);
          const y = padding.top + (1 - val / maxValue) * chartH;
          return { x, y, exists: point !== undefined };
        });

        const linePath = points
          .filter((p) => p.exists)
          .map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.y}`)
          .join(" ");

        return (
          <g key={version}>
            {linePath && (
              <path
                d={linePath}
                fill="none"
                stroke={color}
                strokeWidth={2}
                strokeLinejoin="round"
                strokeLinecap="round"
              />
            )}
            {points
              .filter((p) => p.exists)
              .map((p, i) => (
                <circle
                  key={i}
                  cx={p.x}
                  cy={p.y}
                  r={3}
                  fill={color}
                />
              ))}
          </g>
        );
      })}

      {/* X-axis labels */}
      {dateOrder.map((date, i) => (
        <text
          key={date}
          x={getX(i)}
          y={chartH + padding.top + 16}
          fill="#6B7280"
          fontSize={10}
          textAnchor="middle"
        >
          {fmtShortDate(date)}
        </text>
      ))}
    </svg>
  );
}

// ── Main Page ────────────────────────────────────────────────────────────────

export default function MonitoringPage() {
  const [data, setData] = useState<MonitoringResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notConfigured, setNotConfigured] = useState(false);
  const [timeSeriesMetric, setTimeSeriesMetric] = useState<
    "count" | "avgLatencyMs"
  >("count");
  const [selectedVersion, setSelectedVersion] = useState<string | "__all__">("__all__");

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/monitoring");
      const json = await res.json();

      if (json.configured === false) {
        setNotConfigured(true);
        setData(null);
        return;
      }

      if (!res.ok) {
        setError(json.error ?? "Failed to load monitoring data");
        setData(null);
        return;
      }

      setData(json as MonitoringResponse);
      setNotConfigured(false);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load monitoring data",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    // Auto-refresh every 60 seconds
    const interval = setInterval(fetchData, 60_000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // ── Not configured state ─────────────────────────────────────────────────

  if (notConfigured) {
    return (
      <main className="min-h-screen max-w-2xl mx-auto px-4 py-12">
        <div className="text-center">
          <div className="text-5xl mb-4">📊</div>
          <h1 className="text-2xl font-bold text-white mb-2">
            Monitoring Dashboard
          </h1>
          <div className="glass-card p-6">
            <p className="text-text-muted mb-4">
              Langfuse is not configured. To enable monitoring, set the
              following environment variables:
            </p>
            <pre className="text-left text-sm bg-[#0B0E14] p-4 rounded-lg overflow-x-auto text-text-muted">
              {`LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_PUBLIC_KEY=pk-...
# Optional:
LANGFUSE_BASE_URL=https://cloud.langfuse.com`}
            </pre>
          </div>
        </div>
      </main>
    );
  }

  // ── Error state ──────────────────────────────────────────────────────────

  if (error) {
    return (
      <main className="min-h-screen max-w-2xl mx-auto px-4 py-12">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-white mb-2">Monitoring</h1>
          <div className="banner-error mb-4">{error}</div>
          <button onClick={fetchData} className="btn-download">
            Retry
          </button>
        </div>
      </main>
    );
  }

  // ── Loading state ────────────────────────────────────────────────────────

  if (loading && !data) {
    return (
      <main className="min-h-screen max-w-5xl mx-auto px-4 py-12">
        <div className="animate-pulse space-y-6">
          <div className="h-8 w-48 bg-surface-border/20 rounded-lg" />
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-24 bg-surface-border/20 rounded-xl" />
            ))}
          </div>
          <div className="h-64 bg-surface-border/20 rounded-xl" />
        </div>
        <p className="text-center text-text-dim text-sm mt-8">
          Loading monitoring data…
        </p>
      </main>
    );
  }

  if (!data) return null;

  const { overview, versions, timeSeries, recentErrors } = data;
  const versionEntries = Object.entries(versions).sort(
    ([, a], [, b]) => b.count - a.count,
  );
  const maxCount = Math.max(...versionEntries.map(([, v]) => v.count), 1);

  // ── Filter by version ────────────────────────────────────────────────
  const filteredEntries =
    selectedVersion === "__all__"
      ? versionEntries
      : versionEntries.filter(([v]) => v === selectedVersion);

  // Build sorted version list for bar charts
  const countBars = versionEntries.map(([v, m]) => ({
    label: v,
    value: m.count,
  }));

  const latencyBars = versionEntries.map(([v, m]) => ({
    label: v,
    value: m.avgLatencyMs,
  }));

  const confidenceBars = versionEntries.map(([v, m]) => ({
    label: v,
    value: Math.round(m.avgConfidence * 100),
  }));

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <main className="min-h-screen max-w-5xl mx-auto px-3 sm:px-5 py-6 sm:py-10">
      {/* Header */}
      <div className="flex items-center justify-between mb-6 sm:mb-8">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
            Monitoring
          </h1>
          <p className="text-xs text-text-muted mt-0.5">
            Langfuse trace metrics &middot;{" "}
            {overview.lastUpdated
              ? `Updated ${fmtTime(overview.lastUpdated)}`
              : ""}
          </p>
        </div>
        <button
          onClick={fetchData}
          disabled={loading}
          className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition-colors ${
            loading
              ? "border-surface-border/20 text-text-dim cursor-not-allowed"
              : "border-surface-border/40 text-text-muted hover:border-accent-purple/40 hover:text-accent-purple"
          }`}
        >
          {loading ? "Refreshing…" : "↻ Refresh"}
        </button>
      </div>

      {/* ── Overview Cards ──────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4 mb-6 sm:mb-8">
        <StatCard
          label="Predictions"
          value={overview.totalPredictions.toLocaleString()}
          subtitle={`${overview.totalVersions} version${
            overview.totalVersions !== 1 ? "s" : ""
          } deployed`}
          accent
        />
        <StatCard
          label="Avg Latency"
          value={fmtMs(overview.avgLatencyMs)}
        />
        <StatCard
          label="Confidence"
          value={fmtConfidence(overview.avgConfidence)}
          subtitle="across all versions"
        />
        <StatCard
          label="Error Rate"
          value={fmtPct(overview.errorRate)}
          subtitle={recentErrors.length > 0 ? `Last 24h` : "No errors"}
        />
      </div>

      {/* ── Per-Version Comparison ──────────────────────────────────────── */}
      <div className="glass-card p-4 sm:p-5 mb-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold text-white">
            Per-Version Metrics
          </h2>

          {/* Version filter */}
          <div className="flex flex-wrap gap-1.5">
            <button
              onClick={() => setSelectedVersion("__all__")}
              className={`px-2 py-1 text-[11px] font-medium rounded-md transition-colors ${
                selectedVersion === "__all__"
                  ? "bg-accent-purple/20 text-accent-purple"
                  : "text-text-dim hover:text-text-muted"
              }`}
            >
              All
            </button>
            {versionEntries.map(([v]) => (
              <button
                key={v}
                onClick={() => setSelectedVersion(v)}
                className={`px-2 py-1 text-[11px] font-medium rounded-md transition-colors ${
                  selectedVersion === v
                    ? "bg-accent-purple/20 text-accent-purple"
                    : "text-text-dim hover:text-text-muted"
                }`}
              >
                {v.length > 12 ? v.slice(0, 12) + "…" : v}
              </button>
            ))}
          </div>
        </div>

        {/* Table header */}
        <div className="grid grid-cols-[1fr_auto_auto_auto_auto_auto] gap-2 sm:gap-4 items-center pb-2 text-xs font-medium text-text-dim uppercase tracking-wider border-b border-surface-border/20">
          <div>Version</div>
          <div className="text-right w-10">Count</div>
          <div className="text-right w-16">Avg Lat</div>
          <div className="text-right w-16 hidden sm:block">P95 Lat</div>
          <div className="text-right w-12">Conf</div>
          <div className="text-right w-12">Errors</div>
        </div>

        {/* Version rows */}
        {filteredEntries.length === 0 ? (
          <div className="py-6 text-center text-text-dim text-sm">
            {selectedVersion === "__all__"
              ? "No prediction traces found in the last 7 days."
              : `No data for version "${selectedVersion}" in the last 7 days.`}
          </div>
        ) : (
          filteredEntries.map(([version, metrics]) => (
            <VersionRow
              key={version}
              version={version}
              metrics={metrics}
              maxCount={maxCount}
            />
          ))
        )}
      </div>

      {/* ── Bar Charts ──────────────────────────────────────────────────── */}
      {versionEntries.length > 1 && (
        <div className="grid sm:grid-cols-3 gap-3 sm:gap-4 mb-6">
          {/* Count */}
          <div className="glass-card p-4">
            <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">
              Predictions
            </h3>
            <BarChart
              data={countBars}
              barColor="#A78BFA"
              formatValue={(v) => v.toFixed(0)}
            />
          </div>

          {/* Latency */}
          <div className="glass-card p-4">
            <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">
              Avg Latency
            </h3>
            <BarChart
              data={latencyBars}
              barColor="#34D399"
              formatValue={(v) => fmtMs(v)}
            />
          </div>

          {/* Confidence */}
          <div className="glass-card p-4">
            <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">
              Avg Confidence
            </h3>
            <BarChart
              data={confidenceBars}
              barColor="#60A5FA"
              formatValue={(v) => `${v}%`}
            />
          </div>
        </div>
      )}

      {/* ── Time Series ─────────────────────────────────────────────────── */}
      {timeSeries.length > 0 && (
        <div className="glass-card p-4 sm:p-5 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-white">
              7-Day Trend
            </h2>
            <div className="flex gap-1">
              <button
                onClick={() => setTimeSeriesMetric("count")}
                className={`px-2 py-1 text-[11px] font-medium rounded-md transition-colors ${
                  timeSeriesMetric === "count"
                    ? "bg-accent-purple/20 text-accent-purple"
                    : "text-text-dim hover:text-text-muted"
                }`}
              >
                Count
              </button>
              <button
                onClick={() => setTimeSeriesMetric("avgLatencyMs")}
                className={`px-2 py-1 text-[11px] font-medium rounded-md transition-colors ${
                  timeSeriesMetric === "avgLatencyMs"
                    ? "bg-accent-purple/20 text-accent-purple"
                    : "text-text-dim hover:text-text-muted"
                }`}
              >
                Latency
              </button>
            </div>
          </div>
          <TimeSeriesChart
            data={timeSeries}
            selectedMetric={timeSeriesMetric}
          />
          {/* Legend */}
          <div className="flex flex-wrap gap-3 mt-3">
            {[...new Set(timeSeries.map((d) => d.version))].map(
              (version, i) => {
                const colors = [
                  "#A78BFA",
                  "#34D399",
                  "#F87171",
                  "#60A5FA",
                  "#FBBF24",
                  "#F472B6",
                ];
                return (
                  <div key={version} className="flex items-center gap-1.5">
                    <span
                      className="w-2.5 h-2.5 rounded-full"
                      style={{
                        backgroundColor:
                          colors[i % colors.length],
                      }}
                    />
                    <span className="text-[11px] text-text-muted">
                      {version}
                    </span>
                  </div>
                );
              },
            )}
          </div>
        </div>
      )}

      {/* ── Recent Errors ───────────────────────────────────────────────── */}
      {recentErrors.length > 0 && (
        <div className="glass-card p-4 sm:p-5 mb-6 border-red-500/20">
          <h2 className="text-sm font-semibold text-red-400 mb-3">
            ⚠ Recent Errors ({recentErrors.length})
          </h2>
          <div className="space-y-2">
            {recentErrors.slice(0, 10).map((err) => (
              <div
                key={err.id}
                className="flex items-start gap-3 p-2.5 rounded-lg bg-red-500/5 border border-red-500/10 text-sm"
              >
                <div className="min-w-0 flex-1">
                  <div className="text-red-300 font-medium truncate">
                    {err.error}
                  </div>
                  <div className="text-[11px] text-text-dim mt-0.5">
                    {err.modelVersion} &middot;{" "}
                    {fmtTime(err.timestamp)} &middot;{" "}
                    {fmtMs(err.durationMs)}
                  </div>
                </div>
                <div className="text-text-dim text-[11px] font-mono truncate max-w-[80px]">
                  {err.id.slice(0, 8)}…
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Footer ──────────────────────────────────────────────────────── */}
      <div className="text-xs text-text-dim border-t border-surface-border/30 pt-4 text-center">
        Data from Langfuse &middot; Last 7 days
      </div>
    </main>
  );
}
