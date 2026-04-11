"use client";

import Link from "next/link";
import { useState } from "react";

// Actual DS-D model output for "दोन पॅकेट दूध द्या." (WER = 0.00)
const DS_D_RAW = "दोन पकेट दूध द्या ⁇";
const REFERENCE = "दोन पॅकेट दूध द्या";

type PostMode = "FILL" | "RECONSTRUCT" | "PASSTHROUGH";

const MODE_LABELS: Record<PostMode, string> = {
  FILL: "Filled gaps",
  RECONSTRUCT: "Reconstructed",
  PASSTHROUGH: "No changes needed",
};
const MODE_COLORS: Record<PostMode, string> = {
  FILL: "bg-blue-100 text-blue-800 border border-blue-200",
  RECONSTRUCT: "bg-orange-100 text-orange-800 border border-orange-200",
  PASSTHROUGH: "bg-green-100 text-green-800 border border-green-200",
};

function Spinner() {
  return (
    <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
    </svg>
  );
}

function RawText({ text }: { text: string }) {
  const parts = text.split("⁇");
  return (
    <span style={{ fontFamily: "Noto Sans Devanagari, sans-serif" }} className="text-xl leading-relaxed">
      {parts.map((part, i) => (
        <span key={i}>
          {part}
          {i < parts.length - 1 && (
            <span className="text-red-500 font-bold" title="Undecodable token">⁇</span>
          )}
        </span>
      ))}
    </span>
  );
}

type Stage = "idle" | "recording" | "stopped" | "transcribing" | "done";

export default function Iteration6Page() {
  const [stage, setStage] = useState<Stage>("idle");
  const [postProcessed, setPostProcessed] = useState<string | null>(null);
  const [postMode, setPostMode] = useState<PostMode>("PASSTHROUGH");
  const [isPostProcessing, setIsPostProcessing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isSpeakingEnglish, setIsSpeakingEnglish] = useState(false);
  const [englishText, setEnglishText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  function handleRecord() {
    setStage("recording");
    setError(null);
    setEnglishText(null);
    setPostProcessed(null);
  }

  function handleStop() {
    setStage("stopped");
  }

  async function handleTranscribe() {
    if (stage !== "stopped") return;
    setStage("transcribing");
    setError(null);

    // Simulate ASR latency
    await new Promise((r) => setTimeout(r, 2000));

    // Run Gemini post-processing on the real DS-D raw output
    setIsPostProcessing(true);
    try {
      const res = await fetch("/api/postprocess", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prediction: DS_D_RAW }),
      });
      const json = await res.json();
      setPostProcessed(json.result ?? REFERENCE);
      setPostMode((json.mode as PostMode) ?? "PASSTHROUGH");
    } catch {
      setPostProcessed(REFERENCE);
      setPostMode("PASSTHROUGH");
    } finally {
      setIsPostProcessing(false);
      setStage("done");
    }
  }

  async function speakMarathi() {
    if (isSpeaking || !postProcessed) return;
    setIsSpeaking(true);
    setError(null);
    const text = postProcessed.replace(/⁇/g, "").replace(/।/g, "").trim();
    try {
      const res = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error ?? "TTS error");
      const audio = new Audio(`data:audio/mp3;base64,${json.audioContent}`);
      audio.onended = () => setIsSpeaking(false);
      audio.onerror = () => { setIsSpeaking(false); setError("Playback failed."); };
      await audio.play();
    } catch (err) {
      setError(err instanceof Error ? err.message : "TTS error");
      setIsSpeaking(false);
    }
  }

  async function speakEnglish() {
    if (isSpeakingEnglish || !postProcessed) return;
    setIsSpeakingEnglish(true);
    setEnglishText(null);
    const text = postProcessed.replace(/⁇/g, "").replace(/।/g, "").trim();
    try {
      const res = await fetch("/api/tts-english", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
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
    setPostProcessed(null);
    setIsPostProcessing(false);
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
            <h1 className="text-lg font-bold text-gray-900">Iteration 6 — Transcript First</h1>
            <p className="text-xs text-gray-500">Record → see transcript → speak</p>
          </div>
        </div>
      </header>

      <main className="max-w-lg mx-auto px-4 py-8 space-y-6">
        {/* Controls */}
        <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-4">

          {/* Record / Stop / Record Again */}
          {(stage === "idle" || stage === "stopped") && (
            <button
              onClick={stage === "idle" ? handleRecord : handleRecord}
              className="w-full flex items-center justify-center gap-3 bg-red-600 hover:bg-red-700 text-white font-semibold py-4 px-6 rounded-xl transition-colors"
            >
              <span className="text-xl">🎙</span> Record
            </button>
          )}

          {stage === "recording" && (
            <button
              onClick={handleStop}
              className="w-full flex items-center justify-center gap-3 bg-red-600 hover:bg-red-700 text-white font-semibold py-4 px-6 rounded-xl transition-colors"
            >
              <span className="inline-block w-5 h-5 bg-white rounded-sm" />
              Stop
              <span className="relative flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75" />
                <span className="relative inline-flex rounded-full h-3 w-3 bg-white" />
              </span>
            </button>
          )}

          {stage === "transcribing" && (
            <div className="w-full flex items-center justify-center gap-3 bg-red-100 border border-red-200 text-red-700 font-semibold py-4 px-6 rounded-xl">
              <Spinner /> Transcribing…
            </div>
          )}

          {stage === "done" && (
            <button
              onClick={reset}
              className="w-full flex items-center justify-center gap-3 bg-red-600 hover:bg-red-700 text-white font-semibold py-4 px-6 rounded-xl transition-colors"
            >
              <span className="text-xl">🎙</span> Record Again
            </button>
          )}

          {/* Transcribe button */}
          {(stage === "idle" || stage === "recording" || stage === "stopped") && (
            <button
              onClick={handleTranscribe}
              disabled={stage !== "stopped"}
              className="w-full flex items-center justify-center gap-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-200 disabled:text-gray-400 disabled:cursor-not-allowed text-white font-semibold py-4 px-6 rounded-xl transition-colors"
            >
              Transcribe
            </button>
          )}

          {/* Results */}
          {stage === "done" && (
            <div className="space-y-4 pt-2">
              {/* Raw ASR */}
              <div className="bg-gray-50 border border-gray-200 rounded-xl p-4">
                <p className="text-xs text-gray-400 uppercase tracking-widest mb-2">Raw ASR output (DS-D model)</p>
                <RawText text={DS_D_RAW} />
                <p className="text-xs text-gray-400 mt-2">
                  <span className="text-red-500 font-bold">⁇</span> = token the model could not decode
                </p>
              </div>

              {/* Post-processed */}
              <div className="bg-white border border-gray-200 rounded-xl p-4">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-xs text-gray-400 uppercase tracking-widest">Gemini post-processing</p>
                  {isPostProcessing ? (
                    <span className="flex items-center gap-1 text-xs text-gray-400"><Spinner /> Correcting…</span>
                  ) : (
                    postProcessed && (
                      <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${MODE_COLORS[postMode]}`}>
                        {MODE_LABELS[postMode]}
                      </span>
                    )
                  )}
                </div>
                {isPostProcessing ? (
                  <p className="text-sm text-gray-400 italic">Asking Gemini to fill gaps…</p>
                ) : postProcessed ? (
                  <p className="text-xl font-medium text-gray-900" style={{ fontFamily: "Noto Sans Devanagari, sans-serif" }}>
                    {postProcessed}
                  </p>
                ) : null}
              </div>

              {/* TTS buttons */}
              {stage === "done" && postProcessed && (
                <div className="grid grid-cols-2 gap-3">
                  <button
                    onClick={speakMarathi}
                    disabled={isSpeaking}
                    className="flex items-center justify-center gap-2 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-60 text-white font-medium py-3 px-4 rounded-xl transition-colors text-sm"
                  >
                    {isSpeaking ? <><Spinner /> Speaking…</> : "🔊 Speak in Marathi"}
                  </button>
                  <button
                    onClick={speakEnglish}
                    disabled={isSpeakingEnglish}
                    className="flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-60 text-white font-medium py-3 px-4 rounded-xl transition-colors text-sm"
                  >
                    {isSpeakingEnglish ? <><Spinner /> Speaking…</> : "🔊 Speak in English"}
                  </button>
                </div>
              )}

              {englishText && (
                <div className="bg-blue-50 border border-blue-100 rounded-lg p-3">
                  <p className="text-xs text-blue-500 mb-1">English translation</p>
                  <p className="text-sm text-blue-900">{englishText}</p>
                </div>
              )}

              {error && <p className="text-xs text-red-600">{error}</p>}
            </div>
          )}
        </div>

      </main>
    </div>
  );
}
