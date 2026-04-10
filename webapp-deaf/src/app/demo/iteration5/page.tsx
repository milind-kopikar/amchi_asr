"use client";

import Link from "next/link";
import { useState, useRef } from "react";

// ─── Simulated data ─────────────────────────────────────────────────────────
// Fixed sentence Taranath says: "एक लिटर दूध द्या."
// Raw ASR output for DS-D (~34.7% WER): "लिटर" gets garbled
const SIMULATED_RAW = "एक ⁇ दूध द्या";
const CORRECTED_TEXT = "एक लिटर दूध द्या";

function Spinner() {
  return (
    <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
    </svg>
  );
}

type Stage = "idle" | "recording" | "ready" | "processing" | "done" | "error";

export default function Iteration5Page() {
  const [stage, setStage] = useState<Stage>("idle");
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isSpeakingEnglish, setIsSpeakingEnglish] = useState(false);
  const [englishText, setEnglishText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const recordTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  function handleRecord() {
    setStage("recording");
    setError(null);
    setEnglishText(null);
    recordTimer.current = setTimeout(() => {
      setStage("ready");
    }, 2500);
  }

  async function handleTranscribe() {
    if (stage !== "ready") return;
    setStage("processing");
    setError(null);

    // Simulate ASR latency (1.5s), then immediately call TTS with corrected text
    await new Promise((r) => setTimeout(r, 1500));

    await speakMarathi();
    setStage("done");
  }

  async function speakMarathi() {
    setIsSpeaking(true);
    try {
      const res = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: CORRECTED_TEXT }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error ?? "TTS error");
      const audio = new Audio(`data:audio/mp3;base64,${json.audioContent}`);
      audio.onended = () => setIsSpeaking(false);
      audio.onerror = () => { setIsSpeaking(false); setError("Audio playback failed."); };
      await audio.play();
    } catch (err) {
      setError(err instanceof Error ? err.message : "TTS error");
      setIsSpeaking(false);
    }
  }

  async function speakEnglish() {
    if (isSpeakingEnglish) return;
    setIsSpeakingEnglish(true);
    setEnglishText(null);
    try {
      const res = await fetch("/api/tts-english", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: CORRECTED_TEXT }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error ?? "TTS error");
      setEnglishText(json.englishText ?? null);
      const audio = new Audio(`data:audio/mp3;base64,${json.audioContent}`);
      audio.onended = () => setIsSpeakingEnglish(false);
      audio.onerror = () => setIsSpeakingEnglish(false);
      await audio.play();
    } catch (err) {
      setError(err instanceof Error ? err.message : "English TTS error");
      setIsSpeakingEnglish(false);
    }
  }

  function reset() {
    if (recordTimer.current) clearTimeout(recordTimer.current);
    setStage("idle");
    setIsSpeaking(false);
    setIsSpeakingEnglish(false);
    setEnglishText(null);
    setError(null);
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="border-b border-gray-200 bg-white">
        <div className="max-w-lg mx-auto px-4 py-5 flex items-center gap-3">
          <Link href="/" className="text-gray-400 hover:text-gray-600 text-sm">← Back</Link>
          <div>
            <h1 className="text-lg font-bold text-gray-900">Iteration 5 — Direct Speech</h1>
            <p className="text-xs text-gray-500">Record → speak immediately (no transcript shown)</p>
          </div>
        </div>
      </header>

      <main className="max-w-lg mx-auto px-4 py-8 space-y-6">
        {/* Context card */}
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
          <p className="text-sm text-amber-800 font-medium mb-1">Design iteration context</p>
          <p className="text-xs text-amber-700">
            In this first app design, audio goes straight to TTS output as soon as transcription completes.
            The user never sees the transcript — the app speaks for them immediately.
            Taranath found this uncomfortable because errors were spoken aloud without his approval.
          </p>
        </div>

        {/* Simulated sentence */}
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <p className="text-xs text-gray-400 uppercase tracking-widest mb-2">Taranath is saying</p>
          <p className="text-2xl font-medium text-gray-900" style={{ fontFamily: "Noto Sans Devanagari, sans-serif" }}>
            एक लिटर दूध द्या।
          </p>
          <p className="text-sm text-gray-500 mt-1 italic">"Give me one litre of milk."</p>
        </div>

        {/* Controls */}
        <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-5">
          {/* Record button */}
          {stage === "idle" && (
            <button
              onClick={handleRecord}
              className="w-full flex items-center justify-center gap-3 bg-red-600 hover:bg-red-700 text-white font-semibold py-4 px-6 rounded-xl transition-colors"
            >
              <span className="text-xl">🎙</span> Record
            </button>
          )}

          {stage === "recording" && (
            <div className="w-full flex items-center justify-center gap-3 bg-red-100 border-2 border-red-400 text-red-700 font-semibold py-4 px-6 rounded-xl">
              <span className="relative flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-500 opacity-75" />
                <span className="relative inline-flex rounded-full h-3 w-3 bg-red-600" />
              </span>
              Recording… (auto-stops in 2.5s)
            </div>
          )}

          {/* Transcribe button */}
          {stage === "ready" && (
            <button
              onClick={handleTranscribe}
              className="w-full flex items-center justify-center gap-3 bg-purple-600 hover:bg-purple-700 text-white font-semibold py-4 px-6 rounded-xl transition-colors"
            >
              Transcribe &amp; Speak
            </button>
          )}

          {stage === "processing" && (
            <div className="w-full flex items-center justify-center gap-3 bg-purple-50 border border-purple-200 text-purple-700 font-semibold py-4 px-6 rounded-xl">
              <Spinner />
              {isSpeaking ? "Speaking…" : "Transcribing…"}
            </div>
          )}

          {/* Done state */}
          {stage === "done" && (
            <div className="space-y-4">
              <div className="bg-green-50 border border-green-200 rounded-xl p-4 text-center">
                <p className="text-sm text-green-700 font-medium">
                  {isSpeaking ? "🔊 Speaking in Marathi…" : "✓ Spoken in Marathi"}
                </p>
                <p className="text-xs text-green-600 mt-1">
                  No transcript was shown — the app spoke directly for Taranath.
                </p>
              </div>

              <button
                onClick={speakEnglish}
                disabled={isSpeakingEnglish}
                className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-60 text-white font-medium py-3 px-5 rounded-xl transition-colors"
              >
                {isSpeakingEnglish ? <><Spinner /> Speaking in English…</> : "🔊 Speak in English"}
              </button>

              {englishText && (
                <div className="bg-blue-50 border border-blue-100 rounded-lg p-3">
                  <p className="text-xs text-blue-500 mb-1">English translation</p>
                  <p className="text-sm text-blue-900">{englishText}</p>
                </div>
              )}

              {error && <p className="text-xs text-red-600 text-center">{error}</p>}

              <button onClick={reset} className="w-full text-sm text-gray-500 hover:text-gray-700 py-2">
                ↺ Try again
              </button>
            </div>
          )}

          {stage === "error" && (
            <div className="space-y-3">
              <p className="text-sm text-red-600 text-center">{error}</p>
              <button onClick={reset} className="w-full text-sm text-gray-500 hover:text-gray-700 py-2">↺ Try again</button>
            </div>
          )}
        </div>

        {/* What Taranath learned */}
        <div className="bg-gray-100 rounded-xl p-4">
          <p className="text-xs font-semibold text-gray-500 mb-1">What Taranath said about this</p>
          <p className="text-sm text-gray-700 italic">
            "I want to see what was transcribed before it speaks. What if it says something wrong to the vendor?"
          </p>
        </div>
      </main>
    </div>
  );
}
