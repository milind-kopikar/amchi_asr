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

// ─── Wilcoxon Signed-Rank Test ───────────────────────────────────────────────
// Two-tailed, normal approximation with continuity correction + tie handling.
// Returns: W+ (sum of positive ranks), W- (sum of negative ranks),
//          n (non-zero pairs), z (test statistic), p (two-tailed), r (effect size).

interface WilcoxonResult {
  Wplus: number;
  Wminus: number;
  n: number;
  nZeros: number;
  z: number;
  p: number;
  r: number;
}

function erf(x: number): number {
  // Abramowitz & Stegun approximation (max error < 1.5e-7)
  const a = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429];
  const p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const t = 1 / (1 + p * Math.abs(x));
  let poly = 0, tp = t;
  for (const ai of a) { poly += ai * tp; tp *= t; }
  return sign * (1 - poly * Math.exp(-x * x));
}

function normalSurv(z: number): number {
  return 0.5 * (1 - erf(z / Math.sqrt(2)));
}

function wilcoxonSignedRank(before: number[], after: number[]): WilcoxonResult {
  const diffs = before.map((b, i) => b - after[i]);
  const nonZero = diffs.filter((d) => Math.abs(d) > 1e-9);
  const nZeros = diffs.length - nonZero.length;
  const n = nonZero.length;

  if (n === 0) return { Wplus: 0, Wminus: 0, n: 0, nZeros, z: 0, p: 1, r: 0 };

  // Sort by |diff|, preserving original index for rank assignment
  const items = nonZero.map((d, i) => ({ d, abs: Math.abs(d), orig: i }));
  items.sort((a, b) => a.abs - b.abs);

  const ranks = new Array<number>(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j < n - 1 && Math.abs(items[j + 1].abs - items[j].abs) < 1e-9) j++;
    const avgRank = (i + j) / 2 + 1; // 1-indexed average rank
    for (let k = i; k <= j; k++) ranks[items[k].orig] = avgRank;
    i = j + 1;
  }

  let Wplus = 0, Wminus = 0;
  for (let i = 0; i < n; i++) {
    if (nonZero[i] > 0) Wplus += ranks[i];
    else Wminus += ranks[i];
  }

  // Tie correction for variance
  const tieMap: Record<string, number> = {};
  for (const it of items) {
    const key = it.abs.toFixed(8);
    tieMap[key] = (tieMap[key] ?? 0) + 1;
  }
  const tieCorr = Object.values(tieMap).reduce((s, t) => s + (t ** 3 - t), 0) / 48;

  const mu = (n * (n + 1)) / 4;
  const sigma = Math.sqrt((n * (n + 1) * (2 * n + 1)) / 24 - tieCorr);

  const W = Math.min(Wplus, Wminus);
  const Wadj = W + (W < mu ? 0.5 : -0.5); // continuity correction
  const z = (Wadj - mu) / sigma;
  const p = 2 * normalSurv(Math.abs(z));
  const r = Math.abs(z) / Math.sqrt(n); // effect size

  return { Wplus, Wminus, n, nZeros, z, p, r };
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

  // Wilcoxon tests — computed at build time (server component)
  const werTest = wilcoxonSignedRank(werBefore, werAfter);
  const cerTest = wilcoxonSignedRank(cerBefore, cerAfter);

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

        {/* ── Statistical Analysis ── */}
        <section className="space-y-4">
          <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">
            Statistical analysis — Wilcoxon Signed-Rank Test
          </p>
          <p className="text-xs text-gray-500">
            Paired non-parametric test comparing each sample before and after
            post-processing. H₀: no change in median error rate. Two-tailed,
            normal approximation with continuity correction and tie handling.
            Effect size r = |z| / √n.
          </p>

          {/* Results table */}
          <div className="overflow-x-auto rounded-xl border border-gray-200 shadow-sm">
            <table className="w-full text-sm bg-white">
              <thead>
                <tr className="border-b border-gray-100 text-xs text-gray-500 uppercase tracking-wide">
                  <th className="text-left px-4 py-2.5">Metric</th>
                  <th className="text-right px-3 py-2.5">n pairs</th>
                  <th className="text-right px-3 py-2.5">W⁺ (↓ better)</th>
                  <th className="text-right px-3 py-2.5">W⁻ (↑ worse)</th>
                  <th className="text-right px-3 py-2.5">z</th>
                  <th className="text-right px-3 py-2.5">p (two-tailed)</th>
                  <th className="text-right px-3 py-2.5">r</th>
                  <th className="text-left px-4 py-2.5">Result</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {[
                  { label: "WER", test: werTest, direction: "WER improved in 3/37 samples" },
                  { label: "CER", test: cerTest, direction: "CER changed in 10/37 samples" },
                ].map(({ label, test, direction }) => {
                  const sig = test.p < 0.05;
                  const improved = test.Wplus > test.Wminus; // more rank weight on improvements
                  return (
                    <tr key={label} className="text-gray-700">
                      <td className="px-4 py-3 font-semibold">{label}</td>
                      <td className="text-right px-3 py-3 text-gray-500">
                        {test.n}
                        {test.nZeros > 0 && (
                          <span className="text-gray-400"> (+{test.nZeros} tied)</span>
                        )}
                      </td>
                      <td className="text-right px-3 py-3">{test.Wplus.toFixed(1)}</td>
                      <td className="text-right px-3 py-3">{test.Wminus.toFixed(1)}</td>
                      <td className="text-right px-3 py-3">{test.z.toFixed(2)}</td>
                      <td className="text-right px-3 py-3 font-mono">
                        <span className={sig ? (improved ? "text-emerald-700 font-semibold" : "text-amber-700 font-semibold") : "text-gray-500"}>
                          {test.p < 0.001 ? "< 0.001" : test.p.toFixed(3)}
                        </span>
                      </td>
                      <td className="text-right px-3 py-3 text-gray-500">{test.r.toFixed(2)}</td>
                      <td className="px-4 py-3">
                        {!sig ? (
                          <span className="inline-flex items-center gap-1 text-xs text-gray-500 bg-gray-50 border border-gray-200 px-2 py-0.5 rounded-full">
                            Not significant
                          </span>
                        ) : improved ? (
                          <span className="inline-flex items-center gap-1 text-xs text-emerald-700 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-full">
                            ✓ Significant improvement
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-xs text-amber-700 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded-full">
                            ⚠ Significant degradation
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Interpretation callout */}
          <div className="rounded-xl border border-indigo-100 bg-indigo-50 p-4 space-y-2 text-sm">
            <p className="font-semibold text-indigo-800">Interpretation</p>
            <ul className="space-y-1.5 text-indigo-700 text-xs leading-relaxed list-none">
              <li>
                <span className="font-medium">WER (p = {werTest.p.toFixed(3)}, not significant):</span>{" "}
                The safety valve reverted {werTest.nZeros}/{n} samples where Gemini would have
                worsened WER, leaving only {werTest.n} samples with any WER change. With n = {werTest.n}
                effective pairs, the test is underpowered — a minimum of ~6 pairs is needed to
                reach p &lt; 0.05. The safety valve successfully prevented WER regression but
                also limits detectable improvement.
              </li>
              <li>
                <span className="font-medium">CER (p = {cerTest.p.toFixed(3)}, significant):</span>{" "}
                W⁻ ({cerTest.Wminus.toFixed(0)}) ≫ W⁺ ({cerTest.Wplus.toFixed(0)}) — when
                post-processing does change text, it introduces character-level drift
                (substitutions and insertions) even when the overall word-level output is
                preserved by the safety valve. This is a known limitation of LLM-based
                correction on low-resource scripts.
              </li>
            </ul>
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
