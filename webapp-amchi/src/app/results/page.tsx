import Link from "next/link";

// ─── All data from 50-epoch Amchi Konkani run (2026-03-02) ─────────────────
// Source: results/experiments/20260302_031806/final_test_results.json
//         results/amchi_analysis/comparison_report.txt

const SPEAKERS = [
  {
    label: "Asha Heble",
    id: "ashaheble",
    n: 35,
    trainN: 156,
    mean: 0.5169,
    median: 0.500,
    std: 0.2042,
    min: 0.000,
    max: 1.000,
    cer: 0.2183,
    color: "#6366f1",
    colorLight: "#e0e7ff",
    colorBorder: "#a5b4fc",
  },
  {
    label: "Dipti Ajgaonkar",
    id: "dipti",
    n: 35,
    trainN: 3,
    mean: 0.6030,
    median: 0.625,
    std: 0.1927,
    min: 0.143,
    max: 1.000,
    cer: 0.2604,
    color: "#f97316",
    colorLight: "#ffedd5",
    colorBorder: "#fdba74",
  },
  {
    label: "Lali Momadi",
    id: "lalimomadi",
    n: 34,
    trainN: 125,
    mean: 0.5194,
    median: 0.500,
    std: 0.1975,
    min: 0.125,
    max: 1.000,
    cer: 0.2231,
    color: "#10b981",
    colorLight: "#d1fae5",
    colorBorder: "#6ee7b7",
  },
];

// Stacked histogram: c = [asha_count, dipti_count, lali_count] per WER bin
// Bins: [0,10%), [10,20%), ..., [90,100%]  (verified: total = 35+35+34 = 104)
const HIST = [
  { label: "0–10%",   c: [1, 0, 0] },
  { label: "10–20%",  c: [1, 1, 2] },
  { label: "20–30%",  c: [4, 1, 3] },
  { label: "30–40%",  c: [2, 2, 4] },
  { label: "40–50%",  c: [6, 4, 5] },
  { label: "50–60%",  c: [8, 8, 9] },
  { label: "60–70%",  c: [6, 10, 5] },
  { label: "70–80%",  c: [4, 3, 2] },
  { label: "80–90%",  c: [2, 4, 3] },
  { label: "90–100%", c: [1, 2, 1] },
];

const OVERALL_WER = 0.5467;
const OVERALL_STD = 0.2022;
const PILOT_WER = 0.351;

// ─── SVG Histogram ───────────────────────────────────────────────────────────
// Layout constants (SVG coordinate units)
const ML = 34;   // margin left (for y-axis labels)
const MT = 10;   // margin top
const MB = 50;   // margin bottom (for x-axis labels)
const CW = 476;  // chart area width
const CH = 168;  // chart area height
const SLOT = CW / HIST.length;        // px per bin = 47.6
const BAR_W = SLOT - 4;               // bar width with gap
const YMAX = 28;                      // y-axis max (actual max count = 25)
const YS = CH / YMAX;                 // px per count unit

function pct(n: number, d = 1) {
  return (n * 100).toFixed(d) + "%";
}

function StackedHistogram() {
  const svgW = CW + ML + 10;
  const svgH = CH + MT + MB;
  const yTicks = [0, 5, 10, 15, 20, 25];

  return (
    <svg
      viewBox={`0 0 ${svgW} ${svgH}`}
      className="w-full"
      role="img"
      aria-label="WER distribution histogram for 104 test samples, stacked by speaker"
    >
      {/* Gridlines + Y-axis labels */}
      {yTicks.map((cnt) => {
        const y = MT + CH - cnt * YS;
        return (
          <g key={cnt}>
            <line
              x1={ML} y1={y} x2={ML + CW} y2={y}
              stroke={cnt === 0 ? "#d1d5db" : "#f3f4f6"}
              strokeWidth="1"
            />
            <text x={ML - 4} y={y + 4} textAnchor="end" fontSize="10" fill="#9ca3af">
              {cnt}
            </text>
          </g>
        );
      })}

      {/* Stacked bars — asha at bottom, dipti in middle, lali on top */}
      {HIST.map((bin, bi) => {
        const x = ML + bi * SLOT;
        let stackY = MT + CH; // start at chart bottom, grow upward
        const total = bin.c.reduce((a, b) => a + b, 0);

        return (
          <g key={bi}>
            {bin.c.map((cnt, ci) => {
              const h = cnt * YS;
              stackY -= h;
              const rectY = stackY;
              return cnt > 0 ? (
                <g key={ci}>
                  <rect
                    x={x + 2} y={rectY} width={BAR_W} height={h}
                    fill={SPEAKERS[ci].color}
                    opacity="0.80"
                    rx="2"
                  />
                  {h > 13 && (
                    <text
                      x={x + 2 + BAR_W / 2} y={rectY + h / 2 + 4}
                      textAnchor="middle" fontSize="9" fill="white" fontWeight="600"
                    >
                      {cnt}
                    </text>
                  )}
                </g>
              ) : null;
            })}

            {/* Total count above each bar */}
            {total > 0 && (
              <text
                x={x + 2 + BAR_W / 2} y={MT + CH - total * YS - 3}
                textAnchor="middle" fontSize="9" fill="#6b7280"
              >
                {total}
              </text>
            )}

            {/* X-axis bin label */}
            <text
              x={x + 2 + BAR_W / 2} y={MT + CH + 14}
              textAnchor="middle" fontSize="9.5" fill="#6b7280"
            >
              {bin.label}
            </text>
          </g>
        );
      })}

      {/* Mean WER vertical dashed line */}
      {(() => {
        const meanX = ML + OVERALL_WER * CW;
        return (
          <>
            <line
              x1={meanX} y1={MT} x2={meanX} y2={MT + CH}
              stroke="#dc2626" strokeWidth="1.5" strokeDasharray="4,3"
            />
            <text x={meanX + 3} y={MT + 12} fontSize="9" fill="#dc2626">
              Mean {pct(OVERALL_WER)}
            </text>
          </>
        );
      })()}

      {/* Axis lines */}
      <line x1={ML} y1={MT} x2={ML} y2={MT + CH} stroke="#e5e7eb" strokeWidth="1" />
      <line x1={ML} y1={MT + CH} x2={ML + CW} y2={MT + CH} stroke="#e5e7eb" strokeWidth="1" />

      {/* Y-axis title */}
      <text
        transform={`rotate(-90) translate(${-(MT + CH / 2)}, 11)`}
        textAnchor="middle" fontSize="10" fill="#9ca3af"
      >
        Samples
      </text>

      {/* X-axis title */}
      <text
        x={ML + CW / 2} y={svgH - 4}
        textAnchor="middle" fontSize="10" fill="#9ca3af"
      >
        Word Error Rate (WER)
      </text>
    </svg>
  );
}

// ─── Per-speaker horizontal WER bars ─────────────────────────────────────────
function SpeakerBars() {
  const labelW = 128;
  const barAreaW = 330;
  const rowH = 44;
  const svgW = labelW + barAreaW + 55;
  const svgH = SPEAKERS.length * rowH + 30;

  return (
    <svg
      viewBox={`0 0 ${svgW} ${svgH}`}
      className="w-full"
      role="img"
      aria-label="Per-speaker mean WER comparison"
    >
      {/* Pilot baseline dashed line */}
      {(() => {
        const px = labelW + PILOT_WER * barAreaW;
        return (
          <>
            <line x1={px} y1={6} x2={px} y2={svgH - 14} stroke="#9ca3af" strokeWidth="1" strokeDasharray="3,2" />
            <text x={px - 2} y={14} textAnchor="end" fontSize="9" fill="#9ca3af">Pilot</text>
            <text x={px - 2} y={23} textAnchor="end" fontSize="9" fill="#9ca3af">35.1%</text>
          </>
        );
      })()}

      {/* Per-speaker rows */}
      {SPEAKERS.map((sp, i) => {
        const y = 16 + i * rowH;
        const meanBarW = sp.mean * barAreaW;
        const stdL = Math.max(0, sp.mean - sp.std) * barAreaW;
        const stdR = Math.min(1, sp.mean + sp.std) * barAreaW;

        return (
          <g key={sp.id}>
            {/* Track background */}
            <rect x={labelW} y={y + 10} width={barAreaW} height={18} fill="#f3f4f6" rx="3" />
            {/* ±1 std shading */}
            <rect x={labelW + stdL} y={y + 10} width={stdR - stdL} height={18} fill={sp.color} opacity="0.15" />
            {/* Mean bar */}
            <rect x={labelW} y={y + 10} width={meanBarW} height={18} fill={sp.color} opacity="0.75" rx="3" />
            {/* Median tick (white line) */}
            <line
              x1={labelW + sp.median * barAreaW} y1={y + 10}
              x2={labelW + sp.median * barAreaW} y2={y + 28}
              stroke="white" strokeWidth="2"
            />
            {/* Speaker label */}
            <text x={labelW - 8} y={y + 23} textAnchor="end" fontSize="11" fill="#374151" fontWeight="500">
              {sp.label}
            </text>
            {/* Mean WER value label */}
            <text x={labelW + meanBarW + 5} y={y + 23} fontSize="10.5" fill={sp.color} fontWeight="600">
              {pct(sp.mean)}
            </text>
          </g>
        );
      })}

      {/* X-axis ticks */}
      {[0, 0.25, 0.5, 0.75, 1.0].map((v) => {
        const x = labelW + v * barAreaW;
        return (
          <g key={v}>
            <line x1={x} y1={svgH - 16} x2={x} y2={svgH - 10} stroke="#d1d5db" strokeWidth="1" />
            <text x={x} y={svgH - 2} textAnchor="middle" fontSize="9" fill="#9ca3af">
              {Math.round(v * 100)}%
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ─── Error type horizontal stacked bar ───────────────────────────────────────
function ErrorTypeBar() {
  const segments = [
    { label: "Correct (59.3%)",       frac: 0.5929, color: "#d1fae5", text: "#065f46" },
    { label: "Substitutions (27.8%)", frac: 0.2782, color: "#fecaca", text: "#991b1b" },
    { label: "Insertions (9.4%)",     frac: 0.0941, color: "#fed7aa", text: "#92400e" },
    { label: "Deletions (3.5%)",      frac: 0.0349, color: "#e9d5ff", text: "#6b21a8" },
  ];
  const barH = 28;
  const barW = 460;
  let xPos = 0;

  return (
    <div>
      <svg viewBox={`0 0 ${barW} ${barH}`} className="w-full rounded-lg overflow-hidden" role="img" aria-label="Error type breakdown">
        {segments.map((seg) => {
          const w = seg.frac * barW;
          const rx = xPos;
          xPos += w;
          return (
            <g key={seg.label}>
              <rect x={rx} y={0} width={w} height={barH} fill={seg.color} />
              {w > 38 && (
                <text
                  x={rx + w / 2} y={barH / 2 + 4}
                  textAnchor="middle" fontSize="10" fill={seg.text} fontWeight="600"
                >
                  {pct(seg.frac, 0)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2">
        {segments.map((seg) => (
          <span key={seg.label} className="flex items-center gap-1.5 text-xs text-gray-600">
            <span
              className="inline-block w-3 h-3 rounded-sm flex-shrink-0"
              style={{ background: seg.color, outline: `1px solid ${seg.text}30` }}
            />
            {seg.label}
          </span>
        ))}
      </div>
    </div>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────
export default function ResultsPage() {
  return (
    <main className="min-h-screen bg-gray-50 py-6 px-4">
      <div className="max-w-2xl mx-auto space-y-6">

        {/* Nav */}
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

        {/* Header */}
        <header className="space-y-1">
          <h1 className="text-2xl font-bold text-gray-900 tracking-tight">
            Amchi Konkani ASR — Results
          </h1>
          <p className="text-sm text-gray-500">
            50-epoch fine-tune · IndicConformer (AI4Bharat Marathi base) · Story 5 test set · 2026-03-02
          </p>
        </header>

        {/* Summary stat cards */}
        <div className="grid grid-cols-3 gap-3">
          <div className="rounded-xl bg-white border border-gray-200 p-3 shadow-sm text-center">
            <p className="text-2xl font-bold text-gray-900">{pct(OVERALL_WER)}</p>
            <p className="text-xs text-gray-500 mt-0.5">Overall WER</p>
            <p className="text-xs text-gray-400">±{pct(OVERALL_STD)} std</p>
          </div>
          <div className="rounded-xl bg-white border border-gray-200 p-3 shadow-sm text-center">
            <p className="text-2xl font-bold text-gray-900">104</p>
            <p className="text-xs text-gray-500 mt-0.5">Test samples</p>
            <p className="text-xs text-gray-400">3 speakers, Story 5</p>
          </div>
          <div className="rounded-xl bg-white border border-gray-200 p-3 shadow-sm text-center">
            <p className="text-2xl font-bold text-indigo-600">ep 47</p>
            <p className="text-xs text-gray-500 mt-0.5">Best checkpoint</p>
            <p className="text-xs text-gray-400">val WER 53.2%</p>
          </div>
        </div>

        {/* Histogram */}
        <section className="rounded-xl bg-white border border-gray-200 p-4 shadow-sm space-y-3">
          <div>
            <h2 className="text-sm font-semibold text-gray-800">WER Distribution — 104 Test Samples</h2>
            <p className="text-xs text-gray-400 mt-0.5">
              Stacked by speaker. Red dashed line = mean WER (54.7%). White tick in bars = median.
            </p>
          </div>

          <StackedHistogram />

          {/* Legend */}
          <div className="flex flex-wrap gap-3">
            {SPEAKERS.map((sp) => (
              <span key={sp.id} className="flex items-center gap-1.5 text-xs text-gray-600">
                <span className="inline-block w-3 h-3 rounded-sm flex-shrink-0" style={{ background: sp.color }} />
                {sp.label}
              </span>
            ))}
          </div>

          <p className="text-xs text-gray-400 leading-relaxed">
            The distribution peaks in the 50–70% WER range (46 of 104 samples). Only 5 samples achieved
            WER below 20%; 4 samples exceeded 90%. Dipti Ajgaonkar contributes disproportionately to
            the 60–70% and 80–90% bins, reflecting her near-zero training representation.
          </p>
        </section>

        {/* Per-speaker breakdown */}
        <section className="rounded-xl bg-white border border-gray-200 p-4 shadow-sm space-y-4">
          <div>
            <h2 className="text-sm font-semibold text-gray-800">Per-Speaker Breakdown</h2>
            <p className="text-xs text-gray-400 mt-0.5">
              Bar = mean WER. Shading = ±1 std. White tick = median. Dashed = pilot baseline (35.1%).
            </p>
          </div>

          <SpeakerBars />

          {/* Stats table */}
          <div className="overflow-x-auto -mx-1">
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="border-b border-gray-100">
                  {["Speaker", "Test N", "Train N", "Mean WER", "Median", "±Std", "CER"].map((h, i) => (
                    <th
                      key={h}
                      className={`py-1.5 font-medium text-gray-500 ${i === 0 ? "text-left pr-2" : "text-right px-2"}`}
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {SPEAKERS.map((sp) => (
                  <tr key={sp.id} className="border-b border-gray-50">
                    <td className="py-2 pr-2">
                      <span className="flex items-center gap-1.5">
                        <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: sp.color }} />
                        <span className="font-medium text-gray-800">{sp.label}</span>
                      </span>
                    </td>
                    <td className="py-2 px-2 text-right text-gray-600">{sp.n}</td>
                    <td className="py-2 px-2 text-right">
                      <span className={sp.trainN <= 10 ? "font-semibold text-red-600" : "text-gray-600"}>
                        {sp.trainN}{sp.trainN <= 10 ? " ⚠" : ""}
                      </span>
                    </td>
                    <td className="py-2 px-2 text-right font-semibold" style={{ color: sp.color }}>
                      {pct(sp.mean)}
                    </td>
                    <td className="py-2 px-2 text-right text-gray-600">{pct(sp.median, 0)}</td>
                    <td className="py-2 px-2 text-right text-gray-500">{pct(sp.std)}</td>
                    <td className="py-2 pl-2 text-right text-gray-500">{pct(sp.cer)}</td>
                  </tr>
                ))}
                <tr className="bg-gray-50 font-medium">
                  <td className="py-2 pr-2 text-gray-700">Overall</td>
                  <td className="py-2 px-2 text-right text-gray-700">104</td>
                  <td className="py-2 px-2 text-right text-gray-500">511</td>
                  <td className="py-2 px-2 text-right font-bold text-gray-900">{pct(OVERALL_WER)}</td>
                  <td className="py-2 px-2 text-right text-gray-500">—</td>
                  <td className="py-2 px-2 text-right text-gray-500">{pct(OVERALL_STD)}</td>
                  <td className="py-2 pl-2 text-right text-gray-500">23.4%</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="rounded-lg bg-amber-50 border border-amber-100 px-3 py-2 text-xs text-gray-600">
            <span className="font-medium text-amber-700">Dipti Ajgaonkar — 3 training samples, 35 test samples.</span>
            {" "}She joined the project after Stories 1–4 were recorded and only contributed Story 5 content.
            The story-based train/test split (train = Stories 1, 2, 3, 7; test = Story 5) left her with
            near-zero training representation, explaining her higher WER.
          </div>
        </section>

        {/* Statistical analysis */}
        <section className="rounded-xl bg-white border border-gray-200 p-4 shadow-sm space-y-3">
          <h2 className="text-sm font-semibold text-gray-800">Statistical Analysis</h2>

          <div className="space-y-2">
            {/* Kruskal-Wallis */}
            <div className="rounded-lg bg-gray-50 border border-gray-100 p-3 text-xs space-y-1">
              <p className="font-semibold text-gray-700">Inter-speaker difference — Kruskal-Wallis test</p>
              <p className="text-gray-500">
                H = 3.94 &nbsp;|&nbsp; <span className="font-mono">p = 0.139</span> &nbsp;—&nbsp; not significant
              </p>
              <p className="text-gray-400">
                The 8.6 pp gap between Asha (51.7%) and Dipti (60.3%) does not reach statistical
                significance at n = 34–35 per speaker. Pairwise Mann-Whitney U tests also show no
                significant difference between any speaker pair (all p &gt; 0.07).
              </p>
            </div>

            {/* Wilcoxon vs pilot */}
            <div className="rounded-lg bg-red-50 border border-red-100 p-3 text-xs space-y-1">
              <p className="font-semibold text-red-800">Regression vs pilot — Wilcoxon signed-rank test</p>
              <p className="text-gray-700">
                38 audio IDs matched between the pilot run (35.1% WER) and this run (same speaker, Asha Heble).
                Mean WER change:{" "}
                <span className="font-semibold text-red-700">+14.4 pp</span> (35.1% → 49.5%).
                25/38 samples regressed, 7 improved, 6 unchanged.
              </p>
              <p className="text-gray-700">
                Wilcoxon W = 76.0 &nbsp;|&nbsp;{" "}
                <span className="font-mono font-semibold text-red-700">p = 0.000434</span>{" "}
                <span className="text-red-600">***</span>
              </p>
              <p className="text-gray-400">
                Even for Asha — the dominant training speaker — performance regressed significantly.
                This rules out test-set difficulty as the sole cause and confirms overfitting.
              </p>
            </div>

            {/* Sentence length */}
            <div className="rounded-lg bg-gray-50 border border-gray-100 p-3 text-xs space-y-1">
              <p className="font-semibold text-gray-700">Sentence length vs WER — Pearson correlation</p>
              <p className="text-gray-500">
                r = −0.069 &nbsp;|&nbsp; <span className="font-mono">p = 0.487</span> &nbsp;—&nbsp; not significant
              </p>
              <p className="text-gray-400">Sentence length has no meaningful effect on error rate.</p>
            </div>
          </div>
        </section>

        {/* Error type breakdown */}
        <section className="rounded-xl bg-white border border-gray-200 p-4 shadow-sm space-y-3">
          <div>
            <h2 className="text-sm font-semibold text-gray-800">Error Type Breakdown</h2>
            <p className="text-xs text-gray-400 mt-0.5">
              Fraction of all reference words across 104 test samples.
            </p>
          </div>
          <ErrorTypeBar />
          <p className="text-xs text-gray-400 leading-relaxed">
            Substitutions dominate (27.8%) — the model produces plausible Konkani phonemes but wrong
            tokens, typical of a Marathi base model adapting to a related dialect. Insertions (9.4%)
            reflect repeated or hallucinated tokens common in CTC decoding when the encoder overfits.
            Deletions are rare (3.5%).
          </p>
        </section>

        {/* Root cause + training context */}
        <section className="rounded-xl bg-white border border-gray-200 p-4 shadow-sm space-y-4">
          <h2 className="text-sm font-semibold text-gray-800">Root Cause Analysis</h2>

          <div className="space-y-3 text-xs">
            {[
              {
                n: "1",
                color: "bg-red-100 text-red-700",
                title: "Overfitting — 115M encoder params trained on 511 samples",
                body: "Val loss doubled from 49.5 (epoch 6) to 91.9 (epoch 49), a 1.86× increase. Train loss collapsed to 0.18. The encoder memorised training voices rather than learning generalisable Konkani acoustics. Wilcoxon confirms regression even for the dominant training speaker.",
              },
              {
                n: "2",
                color: "bg-orange-100 text-orange-700",
                title: "Speaker imbalance — Dipti had 3 train / 35 test samples",
                body: "The story-based split structurally excluded Dipti from training because she only recorded Story 5 content (the test set). The model was almost never trained on her voice.",
              },
              {
                n: "3",
                color: "bg-blue-100 text-blue-700",
                title: "Pilot comparison is not apples-to-apples",
                body: "The 35.1% pilot used the same file for validation and test (selection bias), evaluated only 1 speaker, and stopped at epoch 20 before significant overfitting. The 50-epoch run uses a proper held-out 3-speaker test set — a harder and fairer evaluation.",
              },
            ].map(({ n, color, title, body }) => (
              <div key={n} className="flex gap-2.5">
                <span className={`mt-0.5 w-5 h-5 rounded-full flex-shrink-0 flex items-center justify-center text-xs font-bold ${color}`}>
                  {n}
                </span>
                <div>
                  <p className="font-medium text-gray-800">{title}</p>
                  <p className="text-gray-500 mt-0.5 leading-relaxed">{body}</p>
                </div>
              </div>
            ))}
          </div>

          {/* Training curve summary */}
          <div className="grid grid-cols-2 gap-2 pt-1">
            {[
              { label: "Val loss (epoch 6 → 49)", value: "49.5 → 91.9  (+1.86×)" },
              { label: "Train loss (epoch 49)", value: "0.18 (collapsed)" },
              { label: "Best checkpoint", value: "Epoch 47, val WER 53.2%" },
              { label: "Trainable params", value: "115M encoder + 132K CTC" },
              { label: "Train / Dev / Test", value: "511 / 58 / 104 samples" },
              { label: "95% CI (bootstrap)", value: "[50.7%, 58.5%]" },
            ].map(({ label, value }) => (
              <div key={label} className="rounded-lg bg-gray-50 border border-gray-100 p-2.5 text-xs">
                <p className="text-gray-400">{label}</p>
                <p className="font-semibold text-gray-800 mt-0.5">{value}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Next experiments */}
        <section className="rounded-xl bg-indigo-50 border border-indigo-100 p-4 shadow-sm space-y-3">
          <h2 className="text-sm font-semibold text-indigo-800">Next Experiments — Running on RunPod</h2>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="rounded-lg bg-white border border-indigo-100 p-3 space-y-1.5">
              <p className="font-semibold text-indigo-700">Run C — Freeze Encoder</p>
              <p className="text-gray-500">
                Only the 132K CTC head is trained. 100 epochs, cosine LR with warm-up,
                same story-based split. Addresses overfitting.
              </p>
              <p className="text-indigo-500 font-medium">Target: WER &lt; 50%</p>
            </div>
            <div className="rounded-lg bg-white border border-indigo-100 p-3 space-y-1.5">
              <p className="font-semibold text-indigo-700">Run S — Speaker-Stratified</p>
              <p className="text-gray-500">
                Freeze encoder + resplit data 70/15/15 per speaker, giving Dipti ~27 train
                samples (was 3) and ensuring all speakers appear in dev.
              </p>
              <p className="text-indigo-500 font-medium">Target: Dipti WER &lt; 50%</p>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="text-center text-xs text-gray-400 pt-2 pb-4 space-y-0.5 border-t border-gray-100">
          <p>IndicConformer (AI4Bharat) · Fine-tuned on Amchi Konkani (GSB) · 50 epochs · Best ep 47</p>
          <p>indicconformer_stt_mr_hybrid_ctc_rnnt_large · Marathi SentencePiece tokenizer (256 tokens)</p>
        </footer>
      </div>
    </main>
  );
}
