"use client";

import Link from "next/link";
import { useState } from "react";

// Actual DS-D model output for "दोन पॅकेट दूध द्या." (WER = 0.00)
const DS_D_RAW = "दोन पकेट दूध द्या ⁇";
const CORRECTED_TEXT = "दोन पॅकेट दूध द्या";

function Spinner() {
  return (
    <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
    </svg>
  );
}

type Stage = "idle" | "recording" | "stopped" | "processing" | "done";

export default function Iteration5Page() {
  const [stage, setStage] = useState<Stage>("idle");
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isSpeakingEnglish, setIsSpeakingEnglish] = useState(false);
  const [englishText, setEnglishText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  function handleRecord() {
    setStage("recording");
    setError(null);
    setEnglishText(null);
  }

  function handleStop() {
    setStage("stopped");
  }

  function handleReRecord() {
    setStage("recording");
    setError(null);
    setEnglishText(null);
  }

  async function handleTranscribe() {
    if (stage !== "stopped") return;
    setStage("processing");
    setError(null);

    // Simulate ASR latency, then immediately call TTS — no transcript shown
    await new Promise((r) => setTimeout(r, 2000));
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
            <p className="text-xs text-gray-500">Record → speak immediately, no transcript shown</p>
          </div>
        </div>
      </header>

      <main className="max-w-lg mx-auto px-4 py-8 space-y-6">
        {/* Controls */}
        <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-4">

          {/* Record / Stop button */}
          {(stage === "idle" || stage === "stopped" || stage === "done") && (
            <button
              onClick={stage === "done" ? reset : handleRecord}
              className="w-full flex items-center justify-center gap-3 bg-red-600 hover:bg-red-700 text-white font-semibold py-4 px-6 rounded-xl transition-colors"
            >
              <span className="text-xl">🎙</span>
              {stage === "done" ? "Record Again" : "Record"}
            </button>
          )}

          {stage === "recording" && (
            <button
              onClick={handleStop}
              className="w-full flex items-center justify-center gap-3 bg-red-600 hover:bg-red-700 text-white font-semibold py-4 px-6 rounded-xl transition-colors"
            >
              {/* Square stop icon */}
              <span className="inline-block w-5 h-5 bg-white rounded-sm" />
              Stop
              <span className="relative flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75" />
                <span className="relative inline-flex rounded-full h-3 w-3 bg-white" />
              </span>
            </button>
          )}

          {stage === "processing" && (
            <div className="w-full flex items-center justify-center gap-3 bg-red-100 border border-red-200 text-red-700 font-semibold py-4 px-6 rounded-xl">
              <Spinner /> Transcribing…
            </div>
          )}

          {/* Transcribe button — visible when stopped, disabled otherwise */}
          {stage !== "processing" && stage !== "done" && (
            <button
              onClick={handleTranscribe}
              disabled={stage !== "stopped"}
              className="w-full flex items-center justify-center gap-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-200 disabled:text-gray-400 disabled:cursor-not-allowed text-white font-semibold py-4 px-6 rounded-xl transition-colors"
            >
              Transcribe &amp; Speak
            </button>
          )}

          {/* Done — no transcript, just confirmation + English option */}
          {stage === "done" && (
            <div className="space-y-4">
              <div className="bg-green-50 border border-green-200 rounded-xl p-4 text-center">
                <p className="text-sm text-green-700 font-medium">
                  {isSpeaking ? "🔊 Speaking in Marathi…" : "✓ Spoken in Marathi"}
                </p>
                <p className="text-xs text-green-600 mt-1 italic">
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
            </div>
          )}
        </div>

      </main>
    </div>
  );
}
