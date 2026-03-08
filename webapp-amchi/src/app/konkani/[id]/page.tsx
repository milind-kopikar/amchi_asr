"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { EXPERIMENTS } from "@/lib/experiments";
import type { Experiment } from "@/lib/experiments";

// ─── Types ────────────────────────────────────────────────────────────────────

interface PerSample {
  id: number;
  reference: string;
  prediction: string;
  wer: number;
}

interface ResultsData {
  summary: { total_samples: number; mean_wer: number };
  per_sample: PerSample[];
}

// ─── Experiment descriptions ──────────────────────────────────────────────────

const DESCRIPTIONS: Record<
  string,
  { summary: string; hypothesis: string; outcome: string }
> = {
  baseline: {
    summary:
      "Full fine-tune of IndicConformer (all 115M parameters) for 50 epochs. " +
      "Train/test split is story-based: stories 1,2,3,7 for training, story 5 for testing (3 speakers).",
    hypothesis:
      "Fully fine-tuning the model on Amchi Konkani audio establishes a reliable baseline " +
      "and leverages all available capacity to adapt to the language.",
    outcome:
      "Test WER of 54.7% across 104 samples. Overfitting observed after epoch 6 " +
      "(val loss doubled by epoch 49). Sets the comparison point for subsequent runs.",
  },
  runC: {
    summary:
      "Encoder frozen — only the CTC decoder head (132K of 115M params) is trained for 100 epochs. " +
      "Same story-based train/test split as baseline, 3 speakers in test set.",
    hypothesis:
      "IndicConformer was pre-trained on large Indic speech including Marathi (closely related). " +
      "Freezing the encoder forces the model to adapt the decoder only, reducing overfitting risk " +
      "and training time while preserving strong acoustic representations.",
    outcome:
      "Test WER of 49.1% — 5.6pp improvement over baseline. Validates that frozen-encoder " +
      "training is a better fit for limited Amchi Konkani data.",
  },
  runS: {
    summary:
      "Frozen encoder + stratified split: test set is sampled proportionally from all 7 speakers " +
      "and all stories (99 test samples). Broadens speaker and domain coverage.",
    hypothesis:
      "The story-based split may overfit test evaluation to specific speakers. " +
      "A stratified split gives a more realistic estimate of generalisation " +
      "and exposes the model to a wider variety of speakers during training.",
    outcome:
      "Test WER of 34.1% — 15.7pp improvement over Run C and 20.6pp over baseline. " +
      "Best result so far. Confirms that speaker diversity and stratified evaluation " +
      "are key levers for Amchi Konkani ASR.",
  },
};

// ─── Results-tab path helper ──────────────────────────────────────────────────

function getResultsPath(id: string): string {
  const paths: Record<string, string> = {
    baseline: "/data/konkani_results_baseline.json",
    runC: "/data/konkani_results_runc.json",
    runS: "/data/konkani_results_runs.json",
  };
  return paths[id] ?? paths.runS;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function pct(n: number, d = 1) {
  return (n * 100).toFixed(d) + "%";
}

function werBadgeClass(wer: number) {
  if (wer === 0) return "bg-green-100 text-green-700";
  if (wer < 0.3) return "bg-blue-100 text-blue-700";
  if (wer < 0.6) return "bg-amber-100 text-amber-700";
  return "bg-red-100 text-red-600";
}

function modeBadgeClass(mode: string) {
  const colors: Record<string, string> = {
    FILL: "bg-blue-100 text-blue-800 border border-blue-200",
    RECONSTRUCT: "bg-orange-100 text-orange-800 border border-orange-200",
    PASSTHROUGH: "bg-purple-100 text-purple-800 border border-purple-200",
  };
  return colors[mode] ?? colors.FILL;
}

function computeWER(ref: string, hyp: string): number {
  const r = ref.split(/\s+/).filter(Boolean);
  const h = hyp.split(/\s+/).filter(Boolean);
  if (r.length === 0) return h.length === 0 ? 0 : 1;
  const dp = Array.from({ length: r.length + 1 }, (_, i) =>
    Array.from({ length: h.length + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0))
  );
  for (let i = 1; i <= r.length; i++) {
    for (let j = 1; j <= h.length; j++) {
      dp[i][j] =
        r[i - 1] === h[j - 1]
          ? dp[i - 1][j - 1]
          : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
    }
  }
  return dp[r.length][h.length] / r.length;
}

function RawAsrText({ text }: { text: string }) {
  const parts = text.split("⁇");
  return (
    <span className="devanagari text-lg leading-relaxed">
      {parts.map((part, i) => (
        <span key={i}>
          {part}
          {i < parts.length - 1 && (
            <span className="text-red-500 font-bold" title="Undecodable token">
              ⁇
            </span>
          )}
        </span>
      ))}
    </span>
  );
}

function Spinner() {
  return (
    <svg
      className="animate-spin h-4 w-4"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
    </svg>
  );
}

// ─── SVG components (Results tab) ─────────────────────────────────────────────

const ML = 34, MT = 10, MB = 50, CW = 476, CH = 168;
const SLOT = CW / 10;
const BAR_W = SLOT - 4;
const YMAX = 28;
const YS = CH / YMAX;

function StackedHistogram({ exp }: { exp: Experiment }) {
  const svgW = CW + ML + 10;
  const svgH = CH + MT + MB;
  const yTicks = [0, 5, 10, 15, 20, 25];
  const speakersWithExp = exp.speakers.map((s) => ({ ...s, __experiment: exp }));

  return (
    <svg viewBox={`0 0 ${svgW} ${svgH}`} className="w-full" role="img" aria-label="WER distribution histogram">
      {yTicks.map((cnt) => {
        const y = MT + CH - cnt * YS;
        return (
          <g key={cnt}>
            <line x1={ML} y1={y} x2={ML + CW} y2={y} stroke={cnt === 0 ? "#d1d5db" : "#f3f4f6"} strokeWidth="1" />
            <text x={ML - 4} y={y + 4} textAnchor="end" fontSize="10" fill="#9ca3af">{cnt}</text>
          </g>
        );
      })}

      {exp.hist.map((bin, bi) => {
        const x = ML + bi * SLOT;
        let stackY = MT + CH;
        const total = bin.c.reduce((a, b) => a + b, 0);
        return (
          <g key={bi}>
            {bin.c.map((cnt, ci) => {
              const h = cnt * YS;
              stackY -= h;
              const rectY = stackY;
              return cnt > 0 ? (
                <g key={ci}>
                  <rect x={x + 2} y={rectY} width={BAR_W} height={h} fill={speakersWithExp[ci].color} opacity="0.80" rx="2" />
                  {h > 13 && (
                    <text x={x + 2 + BAR_W / 2} y={rectY + h / 2 + 4} textAnchor="middle" fontSize="9" fill="white" fontWeight="600">{cnt}</text>
                  )}
                </g>
              ) : null;
            })}
            {total > 0 && (
              <text x={x + 2 + BAR_W / 2} y={MT + CH - total * YS - 3} textAnchor="middle" fontSize="9" fill="#6b7280">{total}</text>
            )}
            <text x={x + 2 + BAR_W / 2} y={MT + CH + 14} textAnchor="middle" fontSize="9.5" fill="#6b7280">{bin.label}</text>
          </g>
        );
      })}

      {(() => {
        const meanX = ML + exp.wer * CW;
        return (
          <>
            <line x1={meanX} y1={MT} x2={meanX} y2={MT + CH} stroke="#dc2626" strokeWidth="1.5" strokeDasharray="4,3" />
            <text x={meanX + 3} y={MT + 12} fontSize="9" fill="#dc2626">Mean {pct(exp.wer)}</text>
          </>
        );
      })()}

      <line x1={ML} y1={MT} x2={ML} y2={MT + CH} stroke="#e5e7eb" strokeWidth="1" />
      <line x1={ML} y1={MT + CH} x2={ML + CW} y2={MT + CH} stroke="#e5e7eb" strokeWidth="1" />
      <text transform={`rotate(-90) translate(${-(MT + CH / 2)}, 11)`} textAnchor="middle" fontSize="10" fill="#9ca3af">Samples</text>
      <text x={ML + CW / 2} y={svgH - 4} textAnchor="middle" fontSize="10" fill="#9ca3af">Word Error Rate (WER)</text>
    </svg>
  );
}

function SpeakerBars({ exp }: { exp: Experiment }) {
  const labelW = 128;
  const barAreaW = 330;
  const rowH = 44;
  const svgW = labelW + barAreaW + 55;
  const svgH = exp.speakers.length * rowH + 30;

  return (
    <svg viewBox={`0 0 ${svgW} ${svgH}`} className="w-full" role="img" aria-label="Per-speaker mean WER comparison">
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

      {exp.speakers.map((sp, i) => {
        const y = 16 + i * rowH;
        const meanBarW = sp.mean * barAreaW;
        const stdL = Math.max(0, sp.mean - sp.std) * barAreaW;
        const stdR = Math.min(1, sp.mean + sp.std) * barAreaW;
        return (
          <g key={sp.id}>
            <rect x={labelW} y={y + 10} width={barAreaW} height={18} fill="#f3f4f6" rx="3" />
            <rect x={labelW + stdL} y={y + 10} width={stdR - stdL} height={18} fill={sp.color} opacity="0.15" />
            <rect x={labelW} y={y + 10} width={meanBarW} height={18} fill={sp.color} opacity="0.75" rx="3" />
            <line x1={labelW + sp.median * barAreaW} y1={y + 10} x2={labelW + sp.median * barAreaW} y2={y + 28} stroke="white" strokeWidth="2" />
            <text x={labelW - 8} y={y + 23} textAnchor="end" fontSize="11" fill="#374151" fontWeight="500">{sp.label}</text>
            <text x={labelW + meanBarW + 5} y={y + 23} fontSize="10.5" fill={sp.color} fontWeight="600">{pct(sp.mean)}</text>
          </g>
        );
      })}

      {[0, 0.25, 0.5, 0.75, 1.0].map((v) => {
        const x = labelW + v * barAreaW;
        return (
          <g key={v}>
            <line x1={x} y1={svgH - 16} x2={x} y2={svgH - 10} stroke="#d1d5db" strokeWidth="1" />
            <text x={x} y={svgH - 2} textAnchor="middle" fontSize="9" fill="#9ca3af">{Math.round(v * 100)}%</text>
          </g>
        );
      })}
    </svg>
  );
}

function ErrorTypeBar({ exp }: { exp: Experiment }) {
  const barW = 460;
  const barH = 28;
  let xPos = 0;
  return (
    <div>
      <svg viewBox={`0 0 ${barW} ${barH}`} className="w-full rounded-lg overflow-hidden" role="img" aria-label="Error type breakdown">
        {exp.errorTypes.map((seg) => {
          const w = seg.frac * barW;
          const rx = xPos;
          xPos += w;
          return (
            <g key={seg.label}>
              <rect x={rx} y={0} width={w} height={barH} fill={seg.color} />
              {w > 38 && (
                <text x={rx + w / 2} y={barH / 2 + 4} textAnchor="middle" fontSize="10" fill={seg.text} fontWeight="600">
                  {pct(seg.frac, 0)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2">
        {exp.errorTypes.map((seg) => (
          <span key={seg.label} className="flex items-center gap-1.5 text-xs text-gray-600">
            <span className="inline-block w-3 h-3 rounded-sm flex-shrink-0" style={{ background: seg.color, outline: `1px solid ${seg.text}30` }} />
            {seg.label}
          </span>
        ))}
      </div>
    </div>
  );
}

// ─── Demo Tab ─────────────────────────────────────────────────────────────────

function DemoTab({ exp, resultsData }: { exp: Experiment; resultsData: ResultsData | null }) {
  const [selectedId, setSelectedId] = useState<number | null>(null);

  const [isTranscribing, setIsTranscribing] = useState(false);
  const [showRaw, setShowRaw] = useState(false);
  const [postProcessedText, setPostProcessedText] = useState<string | null>(null);
  const [postProcessedMode, setPostProcessedMode] = useState<string | null>(null);
  const [showPostProcessed, setShowPostProcessed] = useState(false);
  const [postWer, setPostWer] = useState<number | null>(null);

  const [isSpeaking, setIsSpeaking] = useState(false);
  const [ttsError, setTtsError] = useState<string | null>(null);
  const [englishText, setEnglishText] = useState<string | null>(null);

  const rawTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (resultsData && resultsData.per_sample.length > 0) {
      setSelectedId(resultsData.per_sample[0].id);
    }
  }, [resultsData]);

  useEffect(() => {
    return () => {
      if (rawTimer.current) clearTimeout(rawTimer.current);
    };
  }, []);

  const samples = resultsData?.per_sample ?? [];
  const selected = samples.find((s) => s.id === selectedId) ?? null;

  function resetResults() {
    setShowRaw(false);
    setPostProcessedText(null);
    setPostProcessedMode(null);
    setShowPostProcessed(false);
    setPostWer(null);
    setIsTranscribing(false);
    setIsSpeaking(false);
    setTtsError(null);
    setEnglishText(null);
    if (rawTimer.current) clearTimeout(rawTimer.current);
  }

  function handleSelectChange(id: number) {
    resetResults();
    setSelectedId(id);
  }

  async function runPostProcess(rawAsr: string, reference: string) {
    try {
      const res = await fetch("/api/postprocess", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: rawAsr }),
      });
      const json = await res.json();
      const corrected: string = json.corrected ?? rawAsr;
      const mode: string = json.mode ?? "PASSTHROUGH";
      setPostProcessedText(corrected);
      setPostProcessedMode(mode);
      setPostWer(computeWER(reference, corrected));
      setShowPostProcessed(true);
    } catch {
      setPostProcessedText(rawAsr);
      setPostProcessedMode("PASSTHROUGH");
      setPostWer(computeWER(reference, rawAsr));
      setShowPostProcessed(true);
    } finally {
      setIsTranscribing(false);
    }
  }

  function handleTranscribe() {
    if (!selected || isTranscribing) return;
    resetResults();
    setIsTranscribing(true);
    rawTimer.current = setTimeout(() => {
      setShowRaw(true);
      runPostProcess(selected.prediction, selected.reference);
    }, 1500);
  }

  async function handleSpeak() {
    if (!postProcessedText || isSpeaking) return;
    setIsSpeaking(true);
    setTtsError(null);
    setEnglishText(null);
    try {
      const res = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: postProcessedText }),
      });
      const json = await res.json();
      if (!res.ok) {
        const msg = json.detail ? `${json.error}: ${json.detail}` : json.error ?? `Error (${res.status})`;
        throw new Error(msg);
      }
      setEnglishText(json.englishText);
      const audio = new Audio(`data:audio/mp3;base64,${json.audioContent}`);
      audio.onended = () => setIsSpeaking(false);
      audio.onerror = () => { setIsSpeaking(false); setTtsError("Audio playback failed."); };
      await audio.play();
    } catch (err) {
      setTtsError(err instanceof Error ? err.message : "TTS error");
      setIsSpeaking(false);
    }
  }

  if (!resultsData) {
    return <p className="text-gray-400 text-sm animate-pulse">Loading samples…</p>;
  }

  const rawWer = selected ? selected.wer : null;
  const werImproved = postWer !== null && rawWer !== null && postWer < rawWer - 0.01;
  const audioSrc = selected ? `/api/audio/${selected.id}` : undefined;

  return (
    <div className="space-y-4">
      {/* Sample selector */}
      <div className="space-y-1">
        <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">
          Select a recording{" "}
          <span className="normal-case font-normal text-gray-400">
            ({samples.length} test samples, sorted best first)
          </span>
        </label>
        <select
          value={selectedId ?? ""}
          onChange={(e) => handleSelectChange(Number(e.target.value))}
          className="devanagari w-full rounded-xl border border-gray-200 bg-white px-3 py-2.5 text-sm text-gray-800 shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
        >
          {samples.map((s) => (
            <option key={s.id} value={s.id}>
              {s.reference} — #{s.id} (WER {Math.round(s.wer * 100)}%)
            </option>
          ))}
        </select>
      </div>

      {/* Reference box */}
      {selected && (
        <div className="rounded-xl border border-indigo-100 bg-indigo-50 p-3 shadow-sm space-y-1">
          <p className="text-xs font-medium text-indigo-500 uppercase tracking-wide">Reference sentence</p>
          <p className="devanagari text-lg text-gray-900 leading-relaxed">{selected.reference}</p>
        </div>
      )}

      {/* Audio player */}
      {selected && (
        <div className="rounded-xl border border-gray-200 bg-white p-3 shadow-sm">
          <audio key={audioSrc} controls src={audioSrc} className="w-full" preload="metadata" />
        </div>
      )}

      {/* Transcribe button */}
      <button
        onClick={handleTranscribe}
        disabled={!selected || isTranscribing}
        className={`w-full py-3.5 rounded-xl font-semibold text-base transition-all shadow-sm select-none ${
          isTranscribing
            ? "bg-indigo-400 text-white cursor-not-allowed"
            : "bg-indigo-600 hover:bg-indigo-700 active:scale-[0.98] text-white cursor-pointer"
        }`}
      >
        {isTranscribing ? (
          <span className="flex items-center justify-center gap-2">
            <Spinner />
            Transcribing…
          </span>
        ) : (
          "Transcribe"
        )}
      </button>

      {/* Progress bar */}
      {isTranscribing && (
        <div className="h-1 w-full bg-gray-100 rounded-full overflow-hidden -mt-2">
          <div className="h-full bg-indigo-400 rounded-full animate-progress" />
        </div>
      )}

      {/* Results */}
      {(showRaw || showPostProcessed) && selected && (
        <div className="space-y-4">
          <hr className="border-gray-200" />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Raw ASR card */}
            <div
              className={`rounded-xl border bg-white p-4 shadow-sm transition-all duration-500 ease-out ${
                showRaw ? "opacity-100 translate-y-0" : "opacity-0 translate-y-3 pointer-events-none"
              }`}
            >
              <div className="flex items-center gap-2 mb-3">
                <span className="text-lg leading-none">🔴</span>
                <span className="text-sm font-semibold text-gray-700">Fine-Tuned ASR Output</span>
              </div>
              <div className="min-h-[3rem] text-gray-800">
                <RawAsrText text={selected.prediction} />
              </div>
              <p className="mt-3 text-xs text-gray-400">
                WER: {rawWer !== null ? Math.round(rawWer * 100) + "%" : "—"}
              </p>
            </div>

            {/* Post-processed card */}
            <div
              className={`rounded-xl border bg-green-50 border-green-200 p-4 shadow-sm transition-all duration-500 ease-out ${
                showPostProcessed ? "opacity-100 translate-y-0" : "opacity-0 translate-y-3 pointer-events-none"
              }`}
            >
              <div className="flex items-center gap-2 mb-3">
                <span className="text-lg leading-none">✅</span>
                <span className="text-sm font-semibold text-gray-700">Post-Processed Output</span>
              </div>
              <div className="devanagari min-h-[3rem] text-gray-900 text-lg leading-relaxed">
                {postProcessedText}
              </div>
              <div className="mt-3 flex flex-wrap items-center gap-2">
                {postProcessedMode && (
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${modeBadgeClass(postProcessedMode)}`}>
                    {postProcessedMode}
                  </span>
                )}
                {postWer !== null && rawWer !== null && (
                  <span className={`text-xs font-medium ${werImproved ? "text-green-700" : "text-gray-500"}`}>
                    WER: {Math.round(rawWer * 100)}% → {Math.round(postWer * 100)}%{" "}
                    {werImproved ? "↓ improved" : "─ unchanged"}
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Speak in English */}
          {showPostProcessed && (
            <div className="space-y-2">
              <button
                onClick={handleSpeak}
                disabled={isSpeaking}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg border text-sm transition-colors select-none ${
                  isSpeaking
                    ? "border-indigo-200 bg-indigo-50 text-indigo-400 cursor-not-allowed"
                    : "border-gray-200 bg-white text-gray-700 hover:bg-gray-50 hover:border-gray-300 cursor-pointer"
                }`}
              >
                {isSpeaking ? (
                  <><Spinner /> Translating &amp; Speaking…</>
                ) : (
                  <>🔊 Speak in English</>
                )}
              </button>

              {englishText && (
                <div className="rounded-lg bg-amber-50 border border-amber-200 px-3 py-2 text-sm text-amber-900">
                  <span className="font-medium text-amber-700">English: </span>
                  {englishText}
                </div>
              )}

              {ttsError && <p className="text-xs text-red-500">{ttsError}</p>}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Results Tab ──────────────────────────────────────────────────────────────

function ResultsTab({ exp }: { exp: Experiment }) {
  return (
    <div className="space-y-6">
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
          <h2 className="text-sm font-semibold text-gray-800">
            WER Distribution — {exp.samples} Test Samples
          </h2>
          <p className="text-xs text-gray-400 mt-0.5">
            Stacked by speaker. Red dashed line = mean WER ({pct(exp.wer)}). White tick = median.
          </p>
        </div>
        <StackedHistogram exp={exp} />
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
        <SpeakerBars exp={exp} />
      </section>

      {/* Error types */}
      <section className="rounded-xl bg-white border border-gray-200 p-4 shadow-sm space-y-3">
        <div>
          <h2 className="text-sm font-semibold text-gray-800">Error Type Breakdown</h2>
          <p className="text-xs text-gray-400 mt-0.5">
            Fraction of reference words affected by each error type.
          </p>
        </div>
        <ErrorTypeBar exp={exp} />
      </section>

      {/* All-experiments comparison */}
      <section className="rounded-xl bg-white border border-gray-200 p-4 shadow-sm space-y-3">
        <h2 className="text-sm font-semibold text-gray-800">All Experiments</h2>
        <div className="space-y-2">
          {Object.values(EXPERIMENTS)
            .sort((a, b) => b.wer - a.wer)
            .map((e) => (
              <Link
                key={e.id}
                href={`/konkani/${e.id}`}
                className={`flex items-center justify-between rounded-lg border px-3 py-2 text-sm transition-colors ${
                  e.id === exp.id
                    ? "border-indigo-200 bg-indigo-50 text-indigo-800"
                    : "border-gray-200 bg-gray-50 text-gray-700 hover:bg-gray-100"
                }`}
              >
                <span className="font-medium">{e.name}</span>
                <span className="font-bold tabular-nums">{pct(e.wer)}</span>
              </Link>
            ))}
        </div>
      </section>
    </div>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function KonkaniExperimentPage() {
  const params = useParams();
  const rawId = Array.isArray(params.id) ? params.id[0] : params.id ?? "runS";

  const exp: Experiment = EXPERIMENTS[rawId] ?? EXPERIMENTS.runS;
  const desc = DESCRIPTIONS[rawId] ?? DESCRIPTIONS.runS;

  const [activeTab, setActiveTab] = useState<"demo" | "results">("demo");
  const [resultsData, setResultsData] = useState<ResultsData | null>(null);

  useEffect(() => {
    setResultsData(null);
    fetch(getResultsPath(rawId))
      .then((r) => r.json())
      .then((d: ResultsData) => setResultsData(d))
      .catch(() => setResultsData(null));
  }, [rawId]);

  return (
    <main className="min-h-screen bg-gray-50 py-6 px-4">
      <div className="max-w-2xl mx-auto space-y-5">

        {/* Back nav */}
        <Link href="/" className="inline-flex items-center gap-1.5 text-sm text-indigo-600 hover:text-indigo-800 transition-colors">
          <span aria-hidden="true">←</span> All experiments
        </Link>

        {/* Header */}
        <header className="space-y-1">
          <h1 className="text-2xl font-bold text-gray-900 tracking-tight">{exp.name}</h1>
          <p className="text-sm text-gray-500">{exp.config}</p>
        </header>

        {/* About this experiment */}
        <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm space-y-3 text-sm">
          <p className="text-gray-700 leading-relaxed">{desc.summary}</p>
          <div className="space-y-1">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Hypothesis</p>
            <p className="text-gray-600 leading-relaxed">{desc.hypothesis}</p>
          </div>
          <div className="space-y-1">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Outcome</p>
            <p className="text-gray-600 leading-relaxed">{desc.outcome}</p>
          </div>
        </div>

        {/* Tabs */}
        <nav className="flex gap-1 border-b border-gray-200 pb-0">
          {(["demo", "results"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-3 py-1.5 text-sm font-medium border-b-2 -mb-px transition-colors capitalize ${
                activeTab === tab
                  ? "text-indigo-600 border-indigo-600"
                  : "text-gray-500 hover:text-gray-700 border-transparent"
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </nav>

        {/* Tab content */}
        {activeTab === "demo" ? (
          <DemoTab exp={exp} resultsData={resultsData} />
        ) : (
          <ResultsTab exp={exp} />
        )}

        {/* Footer */}
        <footer className="text-center text-xs text-gray-400 pt-4 pb-2 space-y-0.5 border-t border-gray-100">
          <p>IndicConformer fine-tuned on Amchi Konkani (GSB / Chitrapur Saraswat)</p>
          <p>Post-processing: Gemini 2.5 Flash · TTS: Google Cloud (Indian English)</p>
        </footer>
      </div>
    </main>
  );
}
