"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────

type Mode = "FILL" | "RECONSTRUCT" | "FILL_REVERTED" | "SKIP";

interface Sample {
  id: number;
  audio_url: string;
  reference: string;
  raw_asr: string;
  corrected: string;
  mode: Mode;
  wer_before: number;
  wer_after: number;
  is_good_demo: boolean;
}

interface SamplesData {
  meta: {
    total_samples: number;
    good_demo_samples: number;
    story: string;
    model: string;
    postprocessing: string;
    mode_legend: Record<Mode, string>;
  };
  samples: Sample[];
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function formatWer(wer: number) {
  return Math.round(wer * 100) + "%";
}

function modeBadge(mode: Mode) {
  const config: Record<Mode, { label: string; className: string }> = {
    FILL: {
      label: "Anchor-guided correction",
      className: "bg-blue-100 text-blue-800 border border-blue-200",
    },
    RECONSTRUCT: {
      label: "Full reconstruction",
      className: "bg-orange-100 text-orange-800 border border-orange-200",
    },
    FILL_REVERTED: {
      label: "Post-processing skipped",
      className: "bg-gray-100 text-gray-600 border border-gray-200",
    },
    SKIP: {
      label: "Perfect transcription",
      className: "bg-green-100 text-green-800 border border-green-200",
    },
  };
  return config[mode] ?? config.SKIP;
}

function RawAsrText({ text }: { text: string }) {
  const parts = text.split("⁇");
  return (
    <span className="devanagari text-lg leading-relaxed">
      {parts.map((part, i) => (
        <span key={i}>
          {part}
          {i < parts.length - 1 && (
            <span
              className="text-red-500 font-bold"
              title="Undecodable token — NeMo garble marker"
            >
              ⁇
            </span>
          )}
        </span>
      ))}
    </span>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function Home() {
  const [data, setData] = useState<SamplesData | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);

  // Reveal state — raw at 1.5 s, corrected at 2.0 s
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [showRaw, setShowRaw] = useState(false);
  const [showCorrected, setShowCorrected] = useState(false);

  // TTS state
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [ttsError, setTtsError] = useState<string | null>(null);

  const rawTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const correctedTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Load samples.json once on mount
  useEffect(() => {
    fetch("/samples.json")
      .then((r) => r.json())
      .then((d: SamplesData) => {
        setData(d);
        const first = d.samples.find((s) => s.wer_after <= 0.75);
        if (first) setSelectedId(first.id);
      });
  }, []);

  useEffect(() => {
    return () => {
      if (rawTimer.current) clearTimeout(rawTimer.current);
      if (correctedTimer.current) clearTimeout(correctedTimer.current);
    };
  }, []);

  if (!data) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <p className="text-gray-400 text-sm animate-pulse">Loading…</p>
      </div>
    );
  }

  const visibleSamples = data.samples.filter((s) => s.wer_after <= 0.75);

  const selected = data.samples.find((s) => s.id === selectedId) ?? null;

  function resetResults() {
    setShowRaw(false);
    setShowCorrected(false);
    setIsTranscribing(false);
    setIsSpeaking(false);
    setTtsError(null);
    if (rawTimer.current) clearTimeout(rawTimer.current);
    if (correctedTimer.current) clearTimeout(correctedTimer.current);
  }

  function handleSelectChange(id: number) {
    resetResults();
    setSelectedId(id);
  }

  function handleTranscribe() {
    if (!selected || isTranscribing) return;
    resetResults();
    setIsTranscribing(true);

    // Raw ASR appears at 1.5 s
    rawTimer.current = setTimeout(() => {
      setShowRaw(true);
    }, 1500);

    // AI-Corrected appears at 2.0 s; spinner stops
    correctedTimer.current = setTimeout(() => {
      setShowCorrected(true);
      setIsTranscribing(false);
    }, 2000);
  }

  async function handleSpeak() {
    if (!selected || isSpeaking) return;
    setIsSpeaking(true);
    setTtsError(null);
    try {
      const res = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: selected.corrected, gender: "MALE" }),
      });
      const json = await res.json();
      if (!res.ok) {
        const msg = json.detail
          ? `${json.error}: ${json.detail}`
          : json.error ?? `TTS failed (${res.status})`;
        throw new Error(msg);
      }
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

  const badge = selected ? modeBadge(selected.mode) : null;
  const werImproved =
    selected !== null && selected.wer_after < selected.wer_before - 0.01;

  // Route audio through /api/audio/[id] to avoid CORS issues
  const audioSrc = selected ? `/api/audio/${selected.id}` : undefined;

  return (
    <main className="min-h-screen bg-gray-50 py-6 px-4">
      <div className="max-w-2xl mx-auto space-y-5">

        {/* ── Nav ── */}
        <nav className="flex gap-1 border-b border-gray-200 pb-0">
          <span className="px-3 py-1.5 text-sm font-medium text-indigo-600 border-b-2 border-indigo-600 -mb-px">
            Demo
          </span>
          <Link
            href="/results"
            className="px-3 py-1.5 text-sm font-medium text-gray-500 hover:text-gray-700 border-b-2 border-transparent -mb-px transition-colors"
          >
            Results
          </Link>
        </nav>

        {/* ── Header ── */}
        <header className="text-center space-y-1 pb-1">
          <h1 className="text-2xl font-bold text-gray-900 tracking-tight">
            🎙️ Deaf Speech Transcriber
          </h1>
          <p className="devanagari text-sm text-gray-500">{data.meta.story}</p>
          <p className="text-xs text-gray-400">
            Fine-tuned IndicConformer + Gemini post-processing
          </p>
        </header>

        {/* ── Sample selector ── */}
        <div className="space-y-1">
          <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">
            Select a recording{" "}
            <span className="normal-case font-normal text-gray-400">
              ({visibleSamples.length} samples with post-processed WER ≤ 75%)
            </span>
          </label>
          <select
            value={selectedId ?? ""}
            onChange={(e) => handleSelectChange(Number(e.target.value))}
            className="devanagari w-full rounded-xl border border-gray-200 bg-white px-3 py-2.5 text-sm text-gray-800 shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
          >
            {visibleSamples.map((s) => (
              <option key={s.id} value={s.id}>
                {s.reference} — #{s.id}
              </option>
            ))}
          </select>
        </div>

        {/* ── Audio player ── */}
        {selected && (
          <div className="rounded-xl border border-gray-200 bg-white p-3 shadow-sm">
            <audio
              key={audioSrc}
              controls
              src={audioSrc}
              className="w-full"
              preload="metadata"
            />
          </div>
        )}

        {/* ── Transcribe button ── */}
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

        {/* ── Thin progress bar ── */}
        {isTranscribing && (
          <div className="h-1 w-full bg-gray-100 rounded-full overflow-hidden -mt-2">
            <div className="h-full bg-indigo-400 rounded-full animate-progress" />
          </div>
        )}

        {/* ── Results ── */}
        {(showRaw || showCorrected) && selected && (
          <div className="space-y-4">
            <hr className="border-gray-200" />

            {/* Side-by-side on md+, stacked on mobile */}
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
                  <RawAsrText text={selected.raw_asr} />
                </div>
                <p className="mt-3 text-xs text-gray-400">
                  WER: {formatWer(selected.wer_before)}
                </p>
              </div>

              {/* AI-Corrected card */}
              <div
                className={`rounded-xl border bg-green-50 border-green-200 p-4 shadow-sm transition-all duration-500 ease-out ${
                  showCorrected
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
                <div className="devanagari min-h-[3rem] text-gray-900 text-lg leading-relaxed">
                  {selected.corrected}
                </div>

                {/* Mode badge + WER change */}
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  {badge && (
                    <span
                      className={`text-xs px-2 py-0.5 rounded-full font-medium ${badge.className}`}
                    >
                      {selected.mode}: {badge.label}
                    </span>
                  )}
                  <span
                    className={`text-xs font-medium ${
                      werImproved ? "text-green-700" : "text-gray-500"
                    }`}
                  >
                    WER: {formatWer(selected.wer_before)} →{" "}
                    {formatWer(selected.wer_after)}{" "}
                    {werImproved ? "↓ improved" : "─ unchanged"}
                  </span>
                </div>
              </div>
            </div>

            {/* Reference */}
            {showCorrected && (
              <div
                className={`flex items-start gap-2 text-sm text-gray-500 px-1 transition-all duration-500 ${
                  showCorrected ? "opacity-100" : "opacity-0"
                }`}
              >
                <span>📖</span>
                <span>
                  <span className="font-medium text-gray-600">Reference: </span>
                  <span className="devanagari">{selected.reference}</span>
                </span>
              </div>
            )}

            {/* TTS — Speak corrected text */}
            {showCorrected && (
              <div className="space-y-1">
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
                    <>
                      <Spinner />
                      Speaking…
                    </>
                  ) : (
                    <>🔊 Speak Corrected Text</>
                  )}
                </button>
                {ttsError && (
                  <p className="text-xs text-red-500">{ttsError}</p>
                )}
              </div>
            )}
          </div>
        )}

        {/* ── Footer ── */}
        <footer className="text-center text-xs text-gray-400 pt-4 pb-2 space-y-0.5 border-t border-gray-100">
          <p>{data.meta.model}</p>
          <p>{data.meta.postprocessing}</p>
        </footer>
      </div>
    </main>
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
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8v8H4z"
      />
    </svg>
  );
}
