"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useState, useEffect, useRef } from "react";
import {
  DEAF_SPEECH_EXPERIMENTS,
  getExperiment,
  getResultsPath,
} from "@/lib/deaf-speech-experiments";
import type { DeafSpeechSample } from "@/lib/deaf-speech-types";

// ─── Experiment descriptions ─────────────────────────────────────────────────

const DESCRIPTIONS: Record<
  string,
  { summary: string; hypothesis: string; outcome: string }
> = {
  dsc: {
    summary:
      "DS-C is the baseline. We fine-tune all 115 million parameters of IndicConformer on 124 recordings from a single deaf speaker (Marathi story 4 — daily activities) for 50 epochs. No encoder freezing, no data augmentation.",
    hypothesis:
      "Full fine-tuning should adapt the pre-trained Marathi acoustic representations to deaf speech characteristics — altered articulation, unusual prosody, and non-standard phoneme production. Starting with a single known speaker gives a clean, controlled baseline.",
    outcome:
      "Achieves 75.3% test WER. High, but expected for a first pass on such a challenging acoustic domain. Every subsequent experiment is compared against this number.",
  },
  dsa: {
    summary:
      "DS-A tests whether freezing the encoder helps. The conformer blocks are frozen and only the CTC decoder head is trained — 132K trainable parameters instead of 115M. Still the same single speaker and 124 samples, but trained for 100 epochs.",
    hypothesis:
      "With only 124 samples from one speaker, full fine-tuning risks catastrophic overfitting. Keeping the pre-trained Marathi acoustic representations frozen and only adapting the output layer might generalise better to the test set.",
    outcome:
      "Achieves 79.6% test WER — 4.3pp worse than DS-C. The hypothesis was wrong: deaf speech is acoustically distant enough from standard Marathi that the encoder itself needs to adapt. Unlike Amchi Konkani (where freezing helped), freezing hurts here.",
  },
  dsb: {
    summary:
      "DS-B tests whether more speakers and more data helps. We add 64 recordings from multiple additional deaf speakers across other story recordings, expanding the training set from 124 to 188 samples. Full fine-tune, 100 epochs.",
    hypothesis:
      "More diverse training data from additional speakers should help the model generalise to a broader range of deaf speech patterns, even if the sentence content differs from the test set.",
    outcome:
      "Achieves 93.1% test WER — 17.8pp worse than DS-C. The additional speakers introduced out-of-distribution acoustic patterns that confused the model. For deaf speech, speaker and domain consistency matters far more than raw data volume.",
  },
  dsd: {
    summary:
      "DS-D applies speed perturbation to the original 124 single-speaker recordings, generating 372 training clips: originals plus 0.9x and 1.1x speed variants. All data stays in-domain, same speaker. Full fine-tune, 100 epochs.",
    hypothesis:
      "Speed perturbation creates acoustic diversity (slightly faster and slower articulation) while keeping the same speaker characteristics and sentence content. This gives 3x training data with zero domain or speaker shift.",
    outcome:
      "Achieves 34.7% test WER — a 40.6pp improvement over DS-C. This is the breakthrough result. In-domain augmentation of a single speaker dramatically outperforms adding new speakers (DS-B) or freezing the encoder (DS-A). The model sees the same speaker at multiple tempos and generalises much better.",
  },
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

type PostMode = "RECONSTRUCT" | "FILL" | "PASSTHROUGH";

const MODE_COLORS: Record<PostMode, string> = {
  FILL: "bg-blue-100 text-blue-800 border border-blue-200",
  RECONSTRUCT: "bg-orange-100 text-orange-800 border border-orange-200",
  PASSTHROUGH: "bg-green-100 text-green-800 border border-green-200",
};

const MODE_LABELS: Record<PostMode, string> = {
  FILL: "Filled gaps",
  RECONSTRUCT: "Reconstructed",
  PASSTHROUGH: "No changes needed",
};

/** Word-level edit distance (Levenshtein) → WER */
function computeWER(reference: string, hypothesis: string): number {
  const normalize = (s: string) =>
    s
      .toLowerCase()
      .replace(/[।?!.,]/g, "")
      .trim()
      .split(/\s+/)
      .filter(Boolean);

  const ref = normalize(reference);
  const hyp = normalize(hypothesis);
  if (ref.length === 0) return hyp.length === 0 ? 0 : 1;

  const dp: number[][] = Array.from({ length: ref.length + 1 }, (_, i) =>
    Array.from({ length: hyp.length + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0))
  );

  for (let i = 1; i <= ref.length; i++) {
    for (let j = 1; j <= hyp.length; j++) {
      dp[i][j] =
        ref[i - 1] === hyp[j - 1]
          ? dp[i - 1][j - 1]
          : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
    }
  }

  return Math.min(dp[ref.length][hyp.length] / ref.length, 1);
}

function werColor(wer: number) {
  if (wer >= 0.8) return "text-red-600";
  if (wer >= 0.6) return "text-orange-500";
  if (wer >= 0.4) return "text-yellow-600";
  return "text-emerald-600";
}

function getAudioId(audioPath: string): string {
  return audioPath.split("/").pop()?.replace(".wav", "") ?? "";
}

function RawAsrText({ text }: { text: string }) {
  const parts = text.split("⁇");
  return (
    <span
      className="text-lg leading-relaxed"
      style={{ fontFamily: "var(--font-devanagari), Noto Sans Devanagari, sans-serif" }}
    >
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

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4 text-center shadow-sm">
      <p className="text-2xl font-bold text-gray-900">{value}</p>
      <p className="text-xs text-gray-500 mt-0.5">{label}</p>
      {sub && <p className="text-xs text-gray-400 mt-0.5">{sub}</p>}
    </div>
  );
}

function WerBadge({ wer }: { wer: number }) {
  const pct = wer * 100;
  const color =
    pct <= 10
      ? "bg-emerald-100 text-emerald-700"
      : pct <= 30
        ? "bg-green-100 text-green-700"
        : pct <= 60
          ? "bg-yellow-100 text-yellow-800"
          : pct <= 80
            ? "bg-orange-100 text-orange-800"
            : "bg-red-100 text-red-700";
  return (
    <span className={`text-xs font-mono px-1.5 py-0.5 rounded ${color}`}>
      {pct.toFixed(0)}%
    </span>
  );
}

function WerHistogram({ samples }: { samples: DeafSpeechSample[] }) {
  if (!samples.length) return null;

  const bins = Array.from({ length: 10 }, (_, i) => {
    const lo = i * 0.1, hi = (i + 1) * 0.1;
    return {
      label: `${i * 10}%`,
      count: samples.filter((s) => (i === 9 ? s.wer >= lo : s.wer >= lo && s.wer < hi)).length,
    };
  });

  const maxCount = Math.max(...bins.map((b) => b.count), 1);
  const mean = samples.reduce((a, s) => a + s.wer, 0) / samples.length;
  const W = 480, H = 180, mt = 20, mr = 10, mb = 44, ml = 28;
  const cw = W - ml - mr, ch = H - mt - mb, bw = cw / 10;
  const yTicks = [0, Math.round(maxCount / 2), maxCount].filter((v, i, a) => a.indexOf(v) === i);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" aria-label="WER distribution histogram">
      <g transform={`translate(${ml},${mt})`}>
        {yTicks.map((v) => {
          const y = ch - (v / maxCount) * ch;
          return (
            <g key={v}>
              <line x1={0} y1={y} x2={cw} y2={y} stroke="#f3f4f6" strokeWidth={1} />
              <text x={-4} y={y + 3.5} fontSize={9} textAnchor="end" fill="#9ca3af">{v}</text>
            </g>
          );
        })}
        {bins.map((b, i) => {
          const h = (b.count / maxCount) * ch;
          return (
            <g key={i}>
              <rect x={i * bw + 1.5} y={ch - h} width={bw - 3} height={Math.max(h, 0)} fill="#6366f1" rx={2} opacity={0.8} />
              {b.count > 0 && (
                <text x={i * bw + bw / 2} y={ch - h - 3} fontSize={8} textAnchor="middle" fill="#374151">{b.count}</text>
              )}
              <text x={i * bw + bw / 2} y={ch + 13} fontSize={7.5} textAnchor="middle" fill="#9ca3af">{b.label}</text>
            </g>
          );
        })}
        <line x1={0} y1={ch} x2={cw} y2={ch} stroke="#e5e7eb" strokeWidth={1} />
        <line x1={0} y1={0} x2={0} y2={ch} stroke="#e5e7eb" strokeWidth={1} />
        <line x1={mean * 10 * bw} y1={0} x2={mean * 10 * bw} y2={ch} stroke="#10b981" strokeDasharray="4 3" strokeWidth={1.5} />
        <text x={Math.min(mean * 10 * bw + 4, cw - 65)} y={12} fontSize={8.5} fill="#10b981">
          mean {(mean * 100).toFixed(1)}%
        </text>
        <text x={cw / 2} y={ch + 30} fontSize={9} textAnchor="middle" fill="#6b7280">Word Error Rate (WER)</text>
        <text x={-(ch / 2)} y={-18} fontSize={9} textAnchor="middle" fill="#6b7280" transform="rotate(-90)"># samples</text>
      </g>
    </svg>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

type SortOrder = "original" | "wer_asc" | "wer_desc";

export default function ExperimentPage() {
  const params = useParams();
  const id = typeof params.id === "string" ? params.id : "dsc";
  const exp = getExperiment(id);
  const desc = DESCRIPTIONS[exp.id] ?? DESCRIPTIONS.dsc;

  // ── Tab
  const [tab, setTab] = useState<"demo" | "results">("demo");

  // ── Shared data
  const [samples, setSamples] = useState<DeafSpeechSample[]>([]);
  const [loading, setLoading] = useState(true);

  // ── Demo interaction
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [showRaw, setShowRaw] = useState(false);
  const [showPostProcessed, setShowPostProcessed] = useState(false);
  const [postProcessedText, setPostProcessedText] = useState<string | null>(null);
  const [postProcessedMode, setPostProcessedMode] = useState<PostMode>("PASSTHROUGH");
  const [postProcessedWer, setPostProcessedWer] = useState<number | null>(null);
  const [isPostProcessing, setIsPostProcessing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [ttsError, setTtsError] = useState<string | null>(null);
  const [isSpeakingEnglish, setIsSpeakingEnglish] = useState(false);
  const [englishText, setEnglishText] = useState<string | null>(null);
  const [englishTtsError, setEnglishTtsError] = useState<string | null>(null);
  const [aboutExpanded, setAboutExpanded] = useState(false);

  // ── Results tab
  const [sortBy, setSortBy] = useState<SortOrder>("original");

  const rawTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const baseline = DEAF_SPEECH_EXPERIMENTS.dsc;
  const baselineDelta = exp.id !== "dsc" ? (exp.wer - baseline.wer) * 100 : null;
  const allExps = Object.values(DEAF_SPEECH_EXPERIMENTS).sort((a, b) => b.wer - a.wer);

  useEffect(() => {
    setLoading(true);
    setSamples([]);
    setSelectedIdx(0);
    resetDemo();
    fetch(getResultsPath(exp.id))
      .then((r) => r.json())
      .then((data) => setSamples(data.per_sample ?? []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [exp.id]);

  useEffect(() => {
    return () => {
      if (rawTimer.current) clearTimeout(rawTimer.current);
    };
  }, []);

  // Demo samples sorted best-first for the dropdown
  const demoSamples = [...samples].sort((a, b) => a.wer - b.wer);
  const selected = demoSamples[selectedIdx] ?? null;
  const audioId = selected ? getAudioId(selected.audio) : null;

  function resetDemo() {
    setShowRaw(false);
    setShowPostProcessed(false);
    setPostProcessedText(null);
    setPostProcessedWer(null);
    setIsPostProcessing(false);
    setIsTranscribing(false);
    setIsSpeaking(false);
    setTtsError(null);
    setIsSpeakingEnglish(false);
    setEnglishText(null);
    setEnglishTtsError(null);
    if (rawTimer.current) clearTimeout(rawTimer.current);
  }

  function handleSelectChange(idx: number) {
    resetDemo();
    setSelectedIdx(idx);
  }

  function handleTranscribe() {
    if (!selected || isTranscribing) return;
    resetDemo();
    setIsTranscribing(true);

    rawTimer.current = setTimeout(() => {
      setShowRaw(true);
      // Kick off Gemini post-processing while the raw card is visible
      runPostProcess(selected.prediction, selected.reference);
    }, 1500);
  }

  async function runPostProcess(prediction: string, reference: string) {
    setIsPostProcessing(true);
    try {
      const res = await fetch("/api/postprocess", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prediction }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error ?? "Post-processing failed");

      const text = json.result ?? prediction;
      setPostProcessedText(text);
      setPostProcessedMode((json.mode as PostMode) ?? "PASSTHROUGH");
      setPostProcessedWer(computeWER(reference, text));
    } catch {
      // Fallback: strip ⁇ tokens
      const cleaned = prediction.replace(/⁇/g, "").replace(/\s+/g, " ").trim();
      setPostProcessedText(cleaned);
      setPostProcessedMode("PASSTHROUGH");
      setPostProcessedWer(computeWER(reference, cleaned));
    } finally {
      setIsPostProcessing(false);
      setShowPostProcessed(true);
      setIsTranscribing(false);
    }
  }

  async function handleSpeak() {
    if (!selected || isSpeaking) return;
    setIsSpeaking(true);
    setTtsError(null);

    // Speak the post-processed output (or raw if post-processing hasn't run)
    const textToSpeak = (postProcessedText ?? selected.prediction)
      .replace(/⁇/g, "")
      .replace(/।/g, "")
      .replace(/\?/g, "")
      .trim();

    try {
      const res = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textToSpeak }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error ?? `Error ${res.status}`);

      const audio = new Audio(`data:audio/mp3;base64,${json.audioContent}`);
      audio.onended = () => setIsSpeaking(false);
      audio.onerror = () => {
        setIsSpeaking(false);
        setTtsError("Audio playback failed.");
      };
      await audio.play();
    } catch (err) {
      setTtsError(err instanceof Error ? err.message : "TTS error");
      setIsSpeaking(false);
    }
  }

  async function handleSpeakEnglish() {
    if (!postProcessedText || isSpeakingEnglish) return;
    setIsSpeakingEnglish(true);
    setEnglishTtsError(null);
    setEnglishText(null);

    const textToTranslate = postProcessedText
      .replace(/⁇/g, "")
      .replace(/।/g, "")
      .trim();

    try {
      const res = await fetch("/api/tts-english", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textToTranslate }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error ?? `Error ${res.status}`);

      setEnglishText(json.englishText);
      const audio = new Audio(`data:audio/mp3;base64,${json.audioContent}`);
      audio.onended = () => setIsSpeakingEnglish(false);
      audio.onerror = () => {
        setIsSpeakingEnglish(false);
        setEnglishTtsError("Audio playback failed.");
      };
      await audio.play();
    } catch (err) {
      setEnglishTtsError(err instanceof Error ? err.message : "Translation/TTS error");
      setIsSpeakingEnglish(false);
    }
  }

  const sortedSamples = [...samples].sort((a, b) => {
    if (sortBy === "wer_asc") return a.wer - b.wer;
    if (sortBy === "wer_desc") return b.wer - a.wer;
    return 0;
  });

  const rawWerImproved =
    postProcessedWer !== null && selected !== null && postProcessedWer < selected.wer - 0.01;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* ── Header ── */}
      <header className="border-b border-gray-200 bg-white">
        <div className="max-w-2xl mx-auto px-4 pt-4 pb-0">
          <Link href="/" className="text-xs text-gray-400 hover:text-gray-600 transition-colors">
            ← All experiments
          </Link>

          <div className="flex items-start justify-between mt-2 pb-3">
            <div>
              <h1 className="text-xl font-bold text-gray-900">{exp.name}</h1>
              <p className="text-xs text-gray-500 mt-0.5">{exp.config}</p>
            </div>
            <div className="text-right shrink-0 ml-4">
              <p className={`text-3xl font-bold ${werColor(exp.wer)}`}>
                {(exp.wer * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-gray-400">Test WER</p>
            </div>
          </div>

          {/* Tabs */}
          <div className="flex gap-0 border-t border-gray-100">
            {(["demo", "results"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors capitalize ${
                  tab === t
                    ? "border-purple-600 text-purple-700"
                    : "border-transparent text-gray-500 hover:text-gray-700"
                }`}
              >
                {t === "demo" ? "Demo" : "Results"}
              </button>
            ))}
          </div>
        </div>
      </header>

      <main className="max-w-2xl mx-auto px-4 py-6 space-y-5">

        {/* ══════════════════ DEMO TAB ══════════════════ */}
        {tab === "demo" && (
          <>
            {/* About — collapsible */}
            <section className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
              <button
                onClick={() => setAboutExpanded(!aboutExpanded)}
                className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-gray-50 transition-colors"
              >
                <span className="text-sm font-semibold text-gray-800">
                  About this experiment
                </span>
                <span className="text-gray-400 text-lg leading-none">
                  {aboutExpanded ? "−" : "+"}
                </span>
              </button>
              {aboutExpanded && (
                <div className="px-4 pb-4 space-y-3 border-t border-gray-100">
                  <p className="text-sm text-gray-700 pt-3">{desc.summary}</p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div className="bg-blue-50 rounded-lg p-3 border border-blue-100">
                      <p className="text-xs font-semibold text-blue-700 mb-1">Hypothesis</p>
                      <p className="text-xs text-blue-900 leading-relaxed">{desc.hypothesis}</p>
                    </div>
                    <div
                      className={`rounded-lg p-3 border ${
                        exp.id === "dsd"
                          ? "bg-emerald-50 border-emerald-100"
                          : "bg-amber-50 border-amber-100"
                      }`}
                    >
                      <p
                        className={`text-xs font-semibold mb-1 ${
                          exp.id === "dsd" ? "text-emerald-700" : "text-amber-700"
                        }`}
                      >
                        Outcome
                      </p>
                      <p
                        className={`text-xs leading-relaxed ${
                          exp.id === "dsd" ? "text-emerald-900" : "text-amber-900"
                        }`}
                      >
                        {desc.outcome}
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </section>

            {/* Sample selector + interactive demo */}
            {loading ? (
              <div className="text-center text-sm text-gray-400 py-8 animate-pulse">
                Loading samples…
              </div>
            ) : demoSamples.length > 0 ? (
              <>
                {/* Dropdown */}
                <div className="space-y-1">
                  <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                    Select a recording{" "}
                    <span className="normal-case font-normal text-gray-400">
                      ({demoSamples.length} samples, sorted best first)
                    </span>
                  </label>
                  <select
                    value={selectedIdx}
                    onChange={(e) => handleSelectChange(Number(e.target.value))}
                    className="w-full rounded-xl border border-gray-200 bg-white px-3 py-2.5 text-sm text-gray-800 shadow-sm focus:outline-none focus:ring-2 focus:ring-purple-400"
                    style={{ fontFamily: "var(--font-devanagari), Noto Sans Devanagari, sans-serif" }}
                  >
                    {demoSamples.map((s, i) => (
                      <option key={i} value={i}>
                        {s.reference} — WER {Math.round(s.wer * 100)}%
                      </option>
                    ))}
                  </select>
                </div>

                {/* Reference sentence */}
                {selected && (
                  <div className="rounded-xl border border-indigo-100 bg-indigo-50 p-3 shadow-sm space-y-1">
                    <p className="text-xs font-medium text-indigo-500 uppercase tracking-wide">
                      Reference sentence
                    </p>
                    <p
                      className="text-lg text-gray-900 leading-relaxed"
                      style={{ fontFamily: "var(--font-devanagari), Noto Sans Devanagari, sans-serif" }}
                    >
                      {selected.reference}
                    </p>
                  </div>
                )}

                {/* Audio player */}
                {audioId && (
                  <div className="rounded-xl border border-gray-200 bg-white p-3 shadow-sm">
                    <audio
                      key={audioId}
                      controls
                      src={`/api/audio/${audioId}`}
                      className="w-full"
                      preload="metadata"
                    />
                  </div>
                )}

                {/* Transcribe button */}
                <button
                  onClick={handleTranscribe}
                  disabled={!selected || isTranscribing}
                  className={`w-full py-3.5 rounded-xl font-semibold text-base transition-all shadow-sm select-none ${
                    isTranscribing
                      ? "bg-purple-400 text-white cursor-not-allowed"
                      : "bg-purple-600 hover:bg-purple-700 active:scale-[0.98] text-white cursor-pointer"
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
                    <div className="h-full bg-purple-400 rounded-full animate-progress" />
                  </div>
                )}

                {/* ── Two-card results ── */}
                {(showRaw || showPostProcessed) && selected && (
                  <div className="space-y-4">
                    <hr className="border-gray-200" />

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {/* Raw ASR card */}
                      <div
                        className={`rounded-xl border bg-white p-4 shadow-sm transition-all duration-500 ease-out ${
                          showRaw
                            ? "opacity-100 translate-y-0"
                            : "opacity-0 translate-y-3 pointer-events-none"
                        }`}
                      >
                        <div className="flex items-center gap-2 mb-3">
                          <span className="text-lg leading-none">🔴</span>
                          <span className="text-sm font-semibold text-gray-700">
                            Fine-Tuned ASR Output
                          </span>
                        </div>
                        <div className="min-h-[3rem] text-gray-800">
                          <RawAsrText text={selected.prediction} />
                        </div>
                        <p className="mt-3 text-xs text-gray-400">
                          WER: {Math.round(selected.wer * 100)}%
                        </p>
                      </div>

                      {/* Post-processed card */}
                      <div
                        className={`rounded-xl border bg-green-50 border-green-200 p-4 shadow-sm transition-all duration-500 ease-out ${
                          showPostProcessed
                            ? "opacity-100 translate-y-0"
                            : "opacity-0 translate-y-3 pointer-events-none"
                        }`}
                      >
                        <div className="flex items-center gap-2 mb-3">
                          <span className="text-lg leading-none">✅</span>
                          <span className="text-sm font-semibold text-gray-700">
                            Post-Processed Output
                          </span>
                        </div>

                        {isPostProcessing ? (
                          <div className="min-h-[3rem] flex items-center gap-2 text-sm text-gray-400">
                            <Spinner />
                            Post-processing with Gemini…
                          </div>
                        ) : (
                          <div
                            className="min-h-[3rem] text-gray-900 text-lg leading-relaxed"
                            style={{ fontFamily: "var(--font-devanagari), Noto Sans Devanagari, sans-serif" }}
                          >
                            {postProcessedText}
                          </div>
                        )}

                        {!isPostProcessing && postProcessedWer !== null && (
                          <div className="mt-3 flex flex-wrap items-center gap-2">
                            <span
                              className={`text-xs px-2 py-0.5 rounded-full font-medium ${MODE_COLORS[postProcessedMode]}`}
                            >
                              {MODE_LABELS[postProcessedMode]}
                            </span>
                            <span
                              className={`text-xs font-medium ${
                                rawWerImproved ? "text-green-700" : "text-gray-500"
                              }`}
                            >
                              WER: {Math.round(selected.wer * 100)}% →{" "}
                              {Math.round(postProcessedWer * 100)}%{" "}
                              {rawWerImproved ? "↓ improved" : "─ unchanged"}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* TTS buttons */}
                    {showPostProcessed && !isPostProcessing && (
                      <div className="space-y-2">
                        <div className="flex flex-wrap gap-2">
                          {/* Marathi TTS */}
                          <button
                            onClick={handleSpeak}
                            disabled={isSpeaking || isSpeakingEnglish}
                            className={`flex items-center gap-2 px-4 py-2 rounded-lg border text-sm transition-colors select-none ${
                              isSpeaking
                                ? "border-purple-200 bg-purple-50 text-purple-400 cursor-not-allowed"
                                : isSpeakingEnglish
                                  ? "border-gray-100 bg-gray-50 text-gray-300 cursor-not-allowed"
                                  : "border-gray-200 bg-white text-gray-700 hover:bg-gray-50 hover:border-gray-300 cursor-pointer"
                            }`}
                          >
                            {isSpeaking ? (
                              <><Spinner /> Speaking Marathi…</>
                            ) : (
                              <>🔊 Speak in Marathi</>
                            )}
                          </button>

                          {/* English translation + TTS */}
                          <button
                            onClick={handleSpeakEnglish}
                            disabled={isSpeakingEnglish || isSpeaking}
                            className={`flex items-center gap-2 px-4 py-2 rounded-lg border text-sm transition-colors select-none ${
                              isSpeakingEnglish
                                ? "border-amber-200 bg-amber-50 text-amber-400 cursor-not-allowed"
                                : isSpeaking
                                  ? "border-gray-100 bg-gray-50 text-gray-300 cursor-not-allowed"
                                  : "border-amber-200 bg-amber-50 text-amber-700 hover:bg-amber-100 hover:border-amber-300 cursor-pointer"
                            }`}
                          >
                            {isSpeakingEnglish ? (
                              <><Spinner /> Translating &amp; Speaking…</>
                            ) : (
                              <>🌐 Speak in English</>
                            )}
                          </button>
                        </div>

                        {/* English translation display */}
                        {englishText && (
                          <div className="rounded-lg bg-amber-50 border border-amber-200 px-3 py-2 text-sm text-amber-900">
                            <span className="font-medium text-amber-700">English: </span>
                            {englishText}
                          </div>
                        )}

                        {ttsError && <p className="text-xs text-red-500">{ttsError}</p>}
                        {englishTtsError && <p className="text-xs text-red-500">{englishTtsError}</p>}
                      </div>
                    )}
                  </div>
                )}
              </>
            ) : (
              <div className="text-center text-sm text-gray-400 py-8">
                No samples available for this experiment.
              </div>
            )}
          </>
        )}

        {/* ══════════════════ RESULTS TAB ══════════════════ */}
        {tab === "results" && (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <StatCard
                label="Test WER"
                value={`${(exp.wer * 100).toFixed(1)}%`}
                sub="lower is better"
              />
              <StatCard
                label="Val WER"
                value={`${(exp.valWer * 100).toFixed(1)}%`}
                sub={`at epoch ${exp.bestEpoch}`}
              />
              <StatCard label="Test Samples" value={exp.samples} sub="unique recordings" />
              <StatCard
                label={baselineDelta !== null ? "vs Baseline" : "Reference"}
                value={
                  baselineDelta !== null
                    ? `${baselineDelta > 0 ? "+" : ""}${baselineDelta.toFixed(1)}pp`
                    : "—"
                }
                sub={
                  baselineDelta !== null
                    ? baselineDelta > 0
                      ? "worse"
                      : "better"
                    : "this is the baseline"
                }
              />
            </div>

            {loading ? (
              <div className="bg-white rounded-xl border border-gray-200 p-8 text-center text-sm text-gray-400 shadow-sm">
                Loading sample data…
              </div>
            ) : samples.length > 0 ? (
              <section className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
                <h2 className="text-sm font-semibold text-gray-800 mb-4">
                  WER Distribution ({samples.length} samples)
                </h2>
                <WerHistogram samples={samples} />
              </section>
            ) : null}

            {samples.length > 0 && (
              <section className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
                <h2 className="text-sm font-semibold text-gray-800 mb-4">
                  Best &amp; Worst Samples
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs font-semibold text-emerald-700 mb-2">Top 5 — lowest WER</p>
                    <div className="space-y-2">
                      {[...samples].sort((a, b) => a.wer - b.wer).slice(0, 5).map((s, i) => (
                        <div key={i} className="text-xs bg-emerald-50 rounded-lg p-2.5 border border-emerald-100 space-y-1">
                          <p className="text-gray-800 font-medium" style={{ fontFamily: "var(--font-devanagari), Noto Sans Devanagari, sans-serif" }}>{s.reference}</p>
                          <p className="text-gray-500" style={{ fontFamily: "var(--font-devanagari), Noto Sans Devanagari, sans-serif" }}>{s.prediction}</p>
                          <WerBadge wer={s.wer} />
                        </div>
                      ))}
                    </div>
                  </div>
                  <div>
                    <p className="text-xs font-semibold text-red-700 mb-2">Bottom 5 — highest WER</p>
                    <div className="space-y-2">
                      {[...samples].sort((a, b) => b.wer - a.wer).slice(0, 5).map((s, i) => (
                        <div key={i} className="text-xs bg-red-50 rounded-lg p-2.5 border border-red-100 space-y-1">
                          <p className="text-gray-800 font-medium" style={{ fontFamily: "var(--font-devanagari), Noto Sans Devanagari, sans-serif" }}>{s.reference}</p>
                          <p className="text-gray-500" style={{ fontFamily: "var(--font-devanagari), Noto Sans Devanagari, sans-serif" }}>{s.prediction}</p>
                          <WerBadge wer={s.wer} />
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </section>
            )}

            {samples.length > 0 && (
              <section className="bg-white rounded-xl border border-gray-200 shadow-sm">
                <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
                  <h2 className="text-sm font-semibold text-gray-800">
                    All samples
                    <span className="text-gray-400 font-normal ml-1">({samples.length})</span>
                  </h2>
                  <select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value as SortOrder)}
                    className="text-xs border border-gray-200 rounded px-2 py-1 text-gray-600 bg-white"
                  >
                    <option value="original">Original order</option>
                    <option value="wer_asc">Best first (low WER)</option>
                    <option value="wer_desc">Worst first (high WER)</option>
                  </select>
                </div>
                <div className="divide-y divide-gray-100 max-h-[560px] overflow-y-auto">
                  {sortedSamples.map((s, i) => (
                    <div key={i} className="px-5 py-3 grid grid-cols-[1fr_1fr_auto] gap-4 items-start text-sm hover:bg-gray-50 transition-colors">
                      <div>
                        <p className="text-xs text-gray-400 mb-0.5">Reference</p>
                        <p className="text-gray-800 font-medium" style={{ fontFamily: "var(--font-devanagari), Noto Sans Devanagari, sans-serif" }}>{s.reference}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-400 mb-0.5">ASR Prediction</p>
                        <p className="text-gray-600" style={{ fontFamily: "var(--font-devanagari), Noto Sans Devanagari, sans-serif" }}>{s.prediction}</p>
                      </div>
                      <div className="pt-5"><WerBadge wer={s.wer} /></div>
                    </div>
                  ))}
                </div>
              </section>
            )}

            <section className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
              <h2 className="text-sm font-semibold text-gray-800 mb-3">All experiments</h2>
              <div className="space-y-1.5">
                {allExps.map((e) => (
                  <div
                    key={e.id}
                    className={`flex items-center justify-between px-3 py-2 rounded-lg ${
                      e.id === exp.id ? "bg-purple-50 border border-purple-200" : "hover:bg-gray-50"
                    }`}
                  >
                    <Link
                      href={`/deaf-speech/${e.id}`}
                      className={`flex-1 text-sm font-medium ${
                        e.id === exp.id ? "text-purple-700" : "text-gray-600 hover:text-gray-900"
                      }`}
                    >
                      {e.name}
                      {e.id === exp.id && (
                        <span className="text-xs font-normal ml-1 text-purple-500">(current)</span>
                      )}
                    </Link>
                    <span className={`text-sm font-bold ${werColor(e.wer)}`}>
                      {(e.wer * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </section>
          </>
        )}
      </main>
    </div>
  );
}
