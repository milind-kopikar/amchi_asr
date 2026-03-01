import Link from "next/link";
import samplesData from "../../../public/samples.json";

// ─── Types ──────────────────────────────────────────────────────────────────

type Sample = (typeof samplesData.samples)[number];

// ─── Histogram helpers ───────────────────────────────────────────────────────

/** Splits values into 10 equal bins: 0–10%, 10–20%, …, 90–100% */
function makeBins(values: number[]): { label: string; count: number }[] {
  return Array.from({ length: 10 }, (_, i) => {
    const lo = i * 10;
    const hi = lo + 10;
    return {
      label: `${lo}`,
      count: values.filter((v) => {
        const pct = v * 100;
        return i === 9 ? pct >= lo && pct <= 100 : pct >= lo && pct < hi;
      }).length,
    };
  });
}

function mean(values: number[]) {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

// ─── SVG Histogram ──────────────────────────────────────────────────────────

interface HistogramProps {
  values: number[];
  title: string;
  subtitle: string;
  barColor: string;
  meanColor: string;
  xAxisLabel: string;
}

function Histogram({
  values,
  title,
  subtitle,
  barColor,
  meanColor,
  xAxisLabel,
}: HistogramProps) {
  const bins = makeBins(values);
  const maxCount = Math.max(...bins.map((b) => b.count), 1);
  const avg = mean(values);

  // Canvas dimensions
  const W = 320,
    H = 200;
  const mt = 22,
    mr = 14,
    mb = 52,
    ml = 28;
  const cw = W - ml - mr;
  const ch = H - mt - mb;
  const barW = cw / bins.length;

  // Y-axis tick values
  const half = Math.round(maxCount / 2);
  const yTicks = [...new Set([0, half > 0 ? half : undefined, maxCount].filter(Boolean) as number[])];

  // Mean x position (0–1 maps to 0–cw over 10 bins)
  const meanX = avg * 10 * barW;

  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm flex flex-col gap-1">
      <p className="text-sm font-semibold text-gray-800">{title}</p>
      <p className="text-xs text-gray-400">{subtitle}</p>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full mt-1"
        aria-label={title}
      >
        <g transform={`translate(${ml},${mt})`}>
          {/* Y gridlines + labels */}
          {yTicks.map((v) => {
            const y = ch - (v / maxCount) * ch;
            return (
              <g key={v}>
                <line
                  x1={0}
                  y1={y}
                  x2={cw}
                  y2={y}
                  stroke="#f3f4f6"
                  strokeWidth={1}
                />
                <text
                  x={-4}
                  y={y + 3.5}
                  fontSize={9}
                  textAnchor="end"
                  fill="#9ca3af"
                >
                  {v}
                </text>
              </g>
            );
          })}

          {/* Bars */}
          {bins.map((b, i) => {
            const barH = (b.count / maxCount) * ch;
            return (
              <g key={i}>
                <rect
                  x={i * barW + 1.5}
                  y={ch - barH}
                  width={barW - 3}
                  height={Math.max(barH, 0)}
                  fill={barColor}
                  rx={2}
                  opacity={0.82}
                />
                {b.count > 0 && (
                  <text
                    x={i * barW + barW / 2}
                    y={ch - barH - 3}
                    fontSize={9}
                    textAnchor="middle"
                    fill="#374151"
                  >
                    {b.count}
                  </text>
                )}
              </g>
            );
          })}

          {/* X-axis bin labels */}
          {bins.map((b, i) => (
            <text
              key={i}
              x={i * barW + barW / 2}
              y={ch + 13}
              fontSize={8}
              textAnchor="middle"
              fill="#9ca3af"
            >
              {b.label}%
            </text>
          ))}

          {/* X-axis title */}
          <text
            x={cw / 2}
            y={ch + 32}
            fontSize={9}
            textAnchor="middle"
            fill="#6b7280"
          >
            {xAxisLabel} (%)
          </text>

          {/* Y-axis "count" label */}
          <text
            x={-(ch / 2)}
            y={-18}
            fontSize={9}
            textAnchor="middle"
            fill="#6b7280"
            transform="rotate(-90)"
          >
            # samples
          </text>

          {/* Axes */}
          <line x1={0} y1={ch} x2={cw} y2={ch} stroke="#e5e7eb" strokeWidth={1} />
          <line x1={0} y1={0} x2={0} y2={ch} stroke="#e5e7eb" strokeWidth={1} />

          {/* Mean dashed line */}
          <line
            x1={meanX}
            y1={0}
            x2={meanX}
            y2={ch}
            stroke={meanColor}
            strokeDasharray="4 3"
            strokeWidth={1.5}
          />
          <text
            x={Math.min(meanX + 4, cw - 36)}
            y={10}
            fontSize={8.5}
            fill={meanColor}
          >
            mean {Math.round(avg * 100)}%
          </text>
        </g>
      </svg>
    </div>
  );
}

// ─── Stat pill ───────────────────────────────────────────────────────────────

function StatPill({
  label,
  value,
  sub,
  color,
}: {
  label: string;
  value: string;
  sub: string;
  color: string;
}) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white px-4 py-3 text-center shadow-sm flex-1">
      <p className="text-xs text-gray-400 mb-0.5">{label}</p>
      <p className={`text-2xl font-bold ${color}`}>{value}</p>
      <p className="text-xs text-gray-400">{sub}</p>
    </div>
  );
}

// ─── Page ────────────────────────────────────────────────────────────────────

export default function ResultsPage() {
  const samples: Sample[] = samplesData.samples;
  const n = samples.length;

  const werBefore = samples.map((s) => s.wer_before);
  const werAfter = samples.map((s) => s.wer_after);
  const cerBefore = samples.map((s) => s.cer_before);
  const cerAfter = samples.map((s) => s.cer_after);

  const meanWerBefore = Math.round(mean(werBefore) * 100);
  const meanWerAfter = Math.round(mean(werAfter) * 100);
  const meanCerBefore = Math.round(mean(cerBefore) * 100);
  const meanCerAfter = Math.round(mean(cerAfter) * 100);

  return (
    <main className="min-h-screen bg-gray-50 py-6 px-4">
      <div className="max-w-3xl mx-auto space-y-6">

        {/* ── Nav ── */}
        <nav className="flex gap-1 border-b border-gray-200 pb-0">
          <Link
            href="/"
            className="px-3 py-1.5 text-sm font-medium text-gray-500 hover:text-gray-700 border-b-2 border-transparent -mb-px transition-colors"
          >
            Demo
          </Link>
          <span className="px-3 py-1.5 text-sm font-medium text-indigo-600 border-b-2 border-indigo-600 -mb-px">
            Results
          </span>
        </nav>

        {/* ── Header ── */}
        <header className="text-center space-y-1">
          <h1 className="text-2xl font-bold text-gray-900 tracking-tight">
            📊 ASR Evaluation Results
          </h1>
          <p className="text-sm text-gray-500">{samplesData.meta.model}</p>
          <p className="text-xs text-gray-400">
            {n} test samples — Amchi Konkani (GSB Konkani)
          </p>
        </header>

        {/* ── Summary stats ── */}
        <section>
          <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">
            Mean metrics across all {n} samples
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <StatPill
              label="WER — Fine-Tuned ASR"
              value={`${meanWerBefore}%`}
              sub="before post-processing"
              color="text-gray-700"
            />
            <StatPill
              label="WER — Post-Processed"
              value={`${meanWerAfter}%`}
              sub={`${meanWerAfter < meanWerBefore ? "↓ " + (meanWerBefore - meanWerAfter) + "pp improvement" : "no change"}`}
              color={meanWerAfter < meanWerBefore ? "text-emerald-600" : "text-gray-700"}
            />
            <StatPill
              label="CER — Fine-Tuned ASR"
              value={`${meanCerBefore}%`}
              sub="before post-processing"
              color="text-gray-700"
            />
            <StatPill
              label="CER — Post-Processed"
              value={`${meanCerAfter}%`}
              sub={`${meanCerAfter < meanCerBefore ? "↓ " + (meanCerBefore - meanCerAfter) + "pp improvement" : meanCerAfter > meanCerBefore ? "↑ " + (meanCerAfter - meanCerBefore) + "pp (char drift)" : "no change"}`}
              color={meanCerAfter < meanCerBefore ? "text-emerald-600" : meanCerAfter > meanCerBefore ? "text-amber-600" : "text-gray-700"}
            />
          </div>
        </section>

        {/* ── WER charts ── */}
        <section>
          <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">
            Word Error Rate (WER) distribution
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Histogram
              values={werBefore}
              title="Fine-Tuned ASR Output"
              subtitle="WER distribution before post-processing"
              barColor="#6366f1"
              meanColor="#4338ca"
              xAxisLabel="WER"
            />
            <Histogram
              values={werAfter}
              title="Post-Processed Output"
              subtitle="WER distribution after post-processing"
              barColor="#10b981"
              meanColor="#059669"
              xAxisLabel="WER"
            />
          </div>
        </section>

        {/* ── CER charts ── */}
        <section>
          <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">
            Character Error Rate (CER) distribution
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Histogram
              values={cerBefore}
              title="Fine-Tuned ASR Output"
              subtitle="CER distribution before post-processing"
              barColor="#6366f1"
              meanColor="#4338ca"
              xAxisLabel="CER"
            />
            <Histogram
              values={cerAfter}
              title="Post-Processed Output"
              subtitle="CER distribution after post-processing"
              barColor="#10b981"
              meanColor="#059669"
              xAxisLabel="CER"
            />
          </div>
        </section>

        {/* ── Footer ── */}
        <footer className="text-center text-xs text-gray-400 pt-4 pb-2 border-t border-gray-100">
          <p>{samplesData.meta.postprocessing}</p>
        </footer>
      </div>
    </main>
  );
}
