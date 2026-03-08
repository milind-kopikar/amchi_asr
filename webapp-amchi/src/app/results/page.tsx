"use client";

import Link from "next/link";
import { useState } from "react";

// ─── Experiment Data ─────────────────────────────────────────────────
// Source: analysis of final_test_results.json files
// 50-epoch baseline: results/experiments/20260302_031806/
// Run C: results/experiments/run_c_story_split/
// Run S: results/experiments/run_c_stratified_split/

const EXPERIMENTS = {
  baseline: {
    id: "baseline",
    name: "50-epoch Baseline",
    date: "2026-03-02",
    config: "Full fine-tune (115M params)",
    samples: 104,
    wer: 0.5467,
    werStd: 0.2022,
    bestEpoch: 47,
    valWer: 0.532,
    speakers: [
      {
        label: "Asha Heble",
        id: "ashaheble",
        n: 35,
        mean: 0.5169,
        median: 0.500,
        std: 0.2072,
        min: 0.000,
        max: 1.000,
        color: "#6366f1",
        colorLight: "#e0e7ff",
        colorBorder: "#a5b4fc",
      },
      {
        label: "Dipti Ajgaonkar",
        id: "dipti",
        n: 35,
        mean: 0.6030,
        median: 0.625,
        std: 0.1955,
        min: 0.143,
        max: 1.000,
        color: "#f97316",
        colorLight: "#ffedd5",
        colorBorder: "#fdba74",
      },
      {
        label: "Lali Momadi",
        id: "lalimomadi",
        n: 34,
        mean: 0.5194,
        median: 0.500,
        std: 0.2005,
        min: 0.125,
        max: 1.000,
        color: "#10b981",
        colorLight: "#d1fae5",
        colorBorder: "#6ee7b7",
      },
    ],
    hist: [
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
    ],
    errorTypes: [
      { label: "Correct (40.9%)", frac: 0.409, color: "#d1fae5", text: "#065f46" },
      { label: "Substitutions (27.8%)", frac: 0.278, color: "#fecaca", text: "#991b1b" },
      { label: "Insertions (9.4%)", frac: 0.094, color: "#fed7aa", text: "#92400e" },
      { label: "Deletions (3.5%)", frac: 0.035, color: "#e9d5ff", text: "#6b21a8" },
    ],
  },
  runC: {
    id: "runC",
    name: "Run C: Frozen Encoder",
    date: "2026-03-06",
    config: "Frozen encoder (132K params) + 100 epochs",
    samples: 104,
    wer: 0.4908,
    werStd: 0.2486,
    bestEpoch: 66,
    valWer: 0.504,
    speakers: [
      {
        label: "Asha Heble",
        id: "ashaheble",
        n: 35,
        mean: 0.4745,
        median: 0.500,
        std: 0.2483,
        min: 0.000,
        max: 1.000,
        color: "#6366f1",
        colorLight: "#e0e7ff",
        colorBorder: "#a5b4fc",
      },
      {
        label: "Dipti Ajgaonkar",
        id: "dipti",
        n: 35,
        mean: 0.5497,
        median: 0.500,
        std: 0.2449,
        min: 0.000,
        max: 1.000,
        color: "#f97316",
        colorLight: "#ffedd5",
        colorBorder: "#fdba74",
      },
      {
        label: "Lali Momadi",
        id: "lalimomadi",
        n: 34,
        mean: 0.4469,
        median: 0.4365,
        std: 0.2485,
        min: 0.000,
        max: 1.200,
        color: "#10b981",
        colorLight: "#d1fae5",
        colorBorder: "#6ee7b7",
      },
    ],
    hist: [
      { label: "0–10%",   c: [2, 0, 1] },
      { label: "10–20%",  c: [2, 1, 3] },
      { label: "20–30%",  c: [3, 1, 4] },
      { label: "30–40%",  c: [4, 3, 4] },
      { label: "40–50%",  c: [6, 4, 5] },
      { label: "50–60%",  c: [7, 6, 6] },
      { label: "60–70%",  c: [5, 8, 4] },
      { label: "70–80%",  c: [3, 5, 3] },
      { label: "80–90%",  c: [2, 4, 2] },
      { label: "90–100%", c: [1, 4, 2] },
    ],
    errorTypes: [
      { label: "Correct (37.0%)", frac: 0.370, color: "#d1fae5", text: "#065f46" },
      { label: "Substitutions (26.6%)", frac: 0.266, color: "#fecaca", text: "#991b1b" },
      { label: "Insertions (4.3%)", frac: 0.043, color: "#fed7aa", text: "#92400e" },
      { label: "Deletions (8.0%)", frac: 0.080, color: "#e9d5ff", text: "#6b21a8" },
    ],
  },
  runS: {
    id: "runS",
    name: "Run S: Stratified Split",
    date: "2026-03-06",
    config: "Frozen encoder + stratified split (99 samples)",
    samples: 99,
    wer: 0.3414,
    werStd: 0.2579,
    bestEpoch: 88,
    valWer: 0.334,
    speakers: [
      {
        label: "Asha Heble",
        id: "ashaheble",
        n: 31,
        mean: 0.3716,
        median: 0.333,
        std: 0.2901,
        min: 0.000,
        max: 1.200,
        color: "#6366f1",
        colorLight: "#e0e7ff",
        colorBorder: "#a5b4fc",
      },
      {
        label: "Lali Momadi",
        id: "lalimomadi",
        n: 29,
        mean: 0.2710,
        median: 0.267,
        std: 0.1930,
        min: 0.000,
        max: 0.833,
        color: "#10b981",
        colorLight: "#d1fae5",
        colorBorder: "#6ee7b7",
      },
      {
        label: "Avinash Kulkarni",
        id: "avkulkarni",
        n: 18,
        mean: 0.3290,
        median: 0.348,
        std: 0.2263,
        min: 0.000,
        max: 0.750,
        color: "#8b5cf6",
        colorLight: "#ede9fe",
        colorBorder: "#c4b5fd",
      },
      {
        label: "Arti Rursushama",
        id: "arursushama",
        n: 7,
        mean: 0.2711,
        median: 0.250,
        std: 0.2844,
        min: 0.000,
        max: 0.833,
        color: "#06b6d4",
        colorLight: "#cffafe",
        colorBorder: "#67e8f9",
      },
      {
        label: "Sheela Kalawar",
        id: "sheela",
        n: 6,
        mean: 0.5683,
        median: 0.464,
        std: 0.3844,
        min: 0.273,
        max: 1.333,
        color: "#ec4899",
        colorLight: "#fce7f3",
        colorBorder: "#f9a8d4",
      },
      {
        label: "Dipti Ajgaonkar",
        id: "dipti",
        n: 5,
        mean: 0.4056,
        median: 0.333,
        std: 0.2775,
        min: 0.111,
        max: 0.833,
        color: "#f97316",
        colorLight: "#ffedd5",
        colorBorder: "#fdba74",
      },
      {
        label: "Milind Kopi",
        id: "milindkopi",
        n: 3,
        mean: 0.3873,
        median: 0.400,
        std: 0.0489,
        min: 0.333,
        max: 0.429,
        color: "#84cc16",
        colorLight: "#ecfccb",
        colorBorder: "#bef264",
      },
    ],
    hist: [
      { label: "0–10%",   c: [4, 4, 2, 1, 0, 0, 0] },
      { label: "10–20%",  c: [4, 5, 2, 1, 0, 0, 0] },
      { label: "20–30%",  c: [5, 6, 3, 1, 0, 0, 0] },
      { label: "30–40%",  c: [6, 5, 4, 1, 0, 1, 1] },
      { label: "40–50%",  c: [4, 3, 3, 1, 1, 0, 0] },
      { label: "50–60%",  c: [3, 2, 2, 1, 1, 1, 0] },
      { label: "60–70%",  c: [2, 1, 1, 0, 1, 0, 0] },
      { label: "70–80%",  c: [1, 1, 1, 1, 1, 1, 0] },
      { label: "80–90%",  c: [1, 1, 0, 0, 1, 0, 0] },
      { label: "90–100%", c: [1, 1, 1, 1, 1, 1, 1] },
    ],
    errorTypes: [
      { label: "Correct (39.1%)", frac: 0.391, color: "#d1fae5", text: "#065f46" },
      { label: "Substitutions (24.2%)", frac: 0.242, color: "#fecaca", text: "#991b1b" },
      { label: "Insertions (4.3%)", frac: 0.043, color: "#fed7aa", text: "#92400e" },
      { label: "Deletions (4.6%)", frac: 0.046, color: "#e9d5ff", text: "#6b21a8" },
    ],
  },
};

// ─── SVG Histogram ───────────────────────────────────────────────────────────
// Layout constants (SVG coordinate units)
const ML = 34;   // margin left (for y-axis labels)
const MT = 10;   // margin top
const MB = 50;   // margin bottom (for x-axis labels)
const CW = 476;  // chart area width
const CH = 168;  // chart area height
const SLOT = CW / 10;        // px per bin = 47.6
const BAR_W = SLOT - 4;               // bar width with gap
const YMAX = 28;                      // y-axis max (actual max count = 25)
const YS = CH / YMAX;                 // px per count unit

function pct(n: number, d = 1) {
  return (n * 100).toFixed(d) + "%";
}

function StackedHistogram({ speakers, hist }: { speakers: any[], hist: any[] }) {
  const svgW = CW + ML + 10;
  const svgH = CH + MT + MB;
  const yTicks = [0, 5, 10, 15, 20, 25];

  return (
    <svg
      viewBox={`0 0 ${svgW} ${svgH}`}
      className="w-full"
      role="img"
      aria-label="WER distribution histogram"
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

      {/* Stacked bars */}
      {hist.map((bin, bi) => {
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
                    fill={speakers[ci].color}
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
        const exp = speakers[0].__experiment;
        const meanX = ML + exp.wer * CW;
        return (
          <>
            <line
              x1={meanX} y1={MT} x2={meanX} y2={MT + CH}
              stroke="#dc2626" strokeWidth="1.5" strokeDasharray="4,3"
            />
            <text x={meanX + 3} y={MT + 12} fontSize="9" fill="#dc2626">
              Mean {pct(exp.wer)}
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
function SpeakerBars({ speakers }: { speakers: any[] }) {
  const labelW = 128;
  const barAreaW = 330;
  const rowH = 44;
  const svgW = labelW + barAreaW + 55;
  const svgH = speakers.length * rowH + 30;

  return (
    <svg
      viewBox={`0 0 ${svgW} ${svgH}`}
      className="w-full"
      role="img"
      aria-label="Per-speaker mean WER comparison"
    >
      {/* Pilot baseline dashed line */}
      {(() => {
        const px = labelW + 0.351 * barAreaW;
        return (
          <>
            <line x1={px} y1={6} x2={px} y2={svgH - 14} stroke="#9ca3af" strokeWidth="1" strokeDasharray="3,2" />
            <text x={px - 2} y={14} textAnchor="end" fontSize="9" fill="#9ca3af">Pilot</text>
            <text x={px - 2} y={23} textAnchor="end" fontSize="9" fill="#9ca3af">35.1%</text>
          </>
        );
      })()}

      {/* Per-speaker rows */}
      {speakers.map((sp, i) => {
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
function ErrorTypeBar({ errorTypes }: { errorTypes: any[] }) {
  const segments = errorTypes;
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
  const [selectedExp, setSelectedExp] = useState("runS");
  const exp = EXPERIMENTS[selectedExp as keyof typeof EXPERIMENTS];

  // Add experiment reference to speakers for histogram
  const speakersWithExp = exp.speakers.map(s => ({ ...s, __experiment: exp }));

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

        {/* Experiment Tabs */}
        <div className="flex gap-1 border-b border-gray-200">
          {Object.values(EXPERIMENTS).map((e) => (
            <button
              key={e.id}
              onClick={() => setSelectedExp(e.id)}
              className={`px-3 py-2 text-sm font-medium border-b-2 -mb-px transition-colors ${
                selectedExp === e.id
                  ? "text-indigo-600 border-indigo-600"
                  : "text-gray-500 hover:text-gray-700 border-transparent"
              }`}
            >
              {e.name}
            </button>
          ))}
        </div>

        {/* Header */}
        <header className="space-y-1">
          <h1 className="text-2xl font-bold text-gray-900 tracking-tight">
            Amchi Konkani ASR — {exp.name}
          </h1>
          <p className="text-sm text-gray-500">
            {exp.config} · Story 5 test set · {exp.date}
          </p>
        </header>

        {/* Summary stat cards */}
        <div className="grid grid-cols-3 gap-3">
          <div className="rounded-xl bg-white border border-gray-200 p-3 shadow-sm text-center">
            <p className="text-2xl font-bold text-gray-900">{pct(exp.wer)}</p>
            <p className="text-xs text-gray-500 mt-0.5">Overall WER</p>
            <p className="text-xs text-gray-400">±{pct(exp.werStd)} std</p>
          </div>
          <div className="rounded-xl bg-white border border-gray-200 p-3 shadow-sm text-center">
            <p className="text-2xl font-bold text-gray-900">{exp.samples}</p>
            <p className="text-xs text-gray-500 mt-0.5">Test samples</p>
            <p className="text-xs text-gray-400">{exp.speakers.length} speakers</p>
          </div>
          <div className="rounded-xl bg-white border border-gray-200 p-3 shadow-sm text-center">
            <p className="text-2xl font-bold text-indigo-600">ep {exp.bestEpoch}</p>
            <p className="text-xs text-gray-500 mt-0.5">Best checkpoint</p>
            <p className="text-xs text-gray-400">val WER {pct(exp.valWer)}</p>
          </div>
        </div>

        {/* Histogram */}
        <section className="rounded-xl bg-white border border-gray-200 p-4 shadow-sm space-y-3">
          <div>
            <h2 className="text-sm font-semibold text-gray-800">WER Distribution — {exp.samples} Test Samples</h2>
            <p className="text-xs text-gray-400 mt-0.5">
              Stacked by speaker. Red dashed line = mean WER ({pct(exp.wer)}). White tick in bars = median.
            </p>
          </div>

          <StackedHistogram speakers={speakersWithExp} hist={exp.hist} />

          {/* Legend */}
          <div className="flex flex-wrap gap-3">
            {exp.speakers.map((sp) => (
              <span key={sp.id} className="flex items-center gap-1.5 text-xs text-gray-600">
                <span className="inline-block w-3 h-3 rounded-sm flex-shrink-0" style={{ background: sp.color }} />
                {sp.label}
              </span>
            ))}
          </div>
        </section>

        {/* Speaker comparison */}
        <section className="rounded-xl bg-white border border-gray-200 p-4 shadow-sm space-y-3">
          <div>
            <h2 className="text-sm font-semibold text-gray-800">Per-Speaker Performance</h2>
            <p className="text-xs text-gray-400 mt-0.5">
              Mean WER with ±1 std shading. Gray dashed line = pilot baseline (35.1%).
            </p>
          </div>

          <SpeakerBars speakers={exp.speakers} />
        </section>

        {/* Error types */}
        <section className="rounded-xl bg-white border border-gray-200 p-4 shadow-sm space-y-3">
          <div>
            <h2 className="text-sm font-semibold text-gray-800">Error Type Breakdown</h2>
            <p className="text-xs text-gray-400 mt-0.5">
              Fraction of reference words affected by each error type.
            </p>
          </div>

          <ErrorTypeBar errorTypes={exp.errorTypes} />
        </section>

      </div>
    </main>
  );
}