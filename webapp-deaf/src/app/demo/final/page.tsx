"use client";

import Link from "next/link";
import { useState, useRef } from "react";

// ─── Simulated data ─────────────────────────────────────────────────────────
const SIMULATED_RAW = "एक ⁇ दूध द्या";

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
    <span style={{ fontFamily: "Noto Sans Devanagari, sans-serif" }} className="text-lg leading-relaxed">
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

type Stage = "idle" | "recording" | "ready" | "transcribing" | "raw" | "done";

export default function FinalPage() {
  const [stage, setStage] = useState<Stage>("idle");
  const [postProcessed, setPostProcessed] = useState<string>("");
  const [postMode, setPostMode] = useState<PostMode>("FILL");
  const [isPostProcessing, setIsPostProcessing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isSpeakingEnglish, setIsSpeakingEnglish] = useState(false);
  const [englishText, setEnglishText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const recordTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  function handleRecord() {
    reset();
    setStage("recording");
    recordTimer.current = setTimeout(() => setStage("ready"), 2500);
  }

  async function handleTranscribe() {
    if (stage !== "ready") return;
    setStage("transcribing");
    setError(null);

    await new Promise((r) => setTimeout(r, 1500));
    setStage("raw");

    setIsPostProcessing(true);
    try {
      const res = await fetch("/api/postprocess", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prediction: SIMULATED_RAW }),
      });
      const json = await res.json();
      setPostProcessed(json.result ?? "एक लिटर दूध द्या");
      setPostMode((json.mode as PostMode) ?? "FILL");
    } catch {
      setPostProcessed("एक लिटर दूध द्या");
      setPostMode("FILL");
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
    if (recordTimer.current) clearTimeout(recordTimer.current);
    setStage("idle");
    setPostProcessed("");
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
          <div className="flex items-center gap-2">
            <div>
              <h1 className="text-lg font-bold text-gray-900">Final Product</h1>
              <p className="text-xs text-gray-500">Record → edit transcript → speak</p>
            </div>
            <span className="text-xs bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded-full font-medium">BEST ⭐</span>
          </div>
        </div>
      </header>

      <main className="max-w-lg mx-auto px-4 py-8 space-y-6">
        {/* Context card */}
        <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-4">
          <p className="text-sm text-emerald-800 font-medium mb-1">Final product</p>
          <p className="text-xs text-emerald-700">
            The transcript is now editable before speaking. Taranath can fix any errors (at ~35% WER, roughly
            1 in 3 words may be wrong) before the app speaks on his behalf. This is the version he uses today.
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

          {stage === "ready" && (
            <button
              onClick={handleTranscribe}
              className="w-full flex items-center justify-center gap-3 bg-purple-600 hover:bg-purple-700 text-white font-semibold py-4 px-6 rounded-xl transition-colors"
            >
              Transcribe
            </button>
          )}

          {stage === "transcribing" && (
            <div className="w-full flex items-center justify-center gap-3 bg-purple-50 border border-purple-200 text-purple-700 font-semibold py-4 px-6 rounded-xl">
              <Spinner /> Transcribing…
            </div>
          )}

          {/* Raw + editable post-processed */}
          {(stage === "raw" || stage === "done") && (
            <div className="space-y-4">
              {/* Raw ASR */}
              <div className="bg-gray-50 border border-gray-200 rounded-xl p-4">
                <p className="text-xs text-gray-400 uppercase tracking-widest mb-2">Raw ASR output</p>
                <RawText text={SIMULATED_RAW} />
                <p className="text-xs text-gray-400 mt-2">
                  <span className="text-red-500 font-bold">⁇</span> = word the model could not decode
                </p>
              </div>

              {/* Post-processed — editable */}
              <div className="bg-white border-2 border-purple-200 rounded-xl p-4">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-xs text-gray-400 uppercase tracking-widest">Corrected transcript</p>
                  {isPostProcessing ? (
                    <span className="flex items-center gap-1 text-xs text-gray-400"><Spinner /> Correcting…</span>
                  ) : (
                    postMode && (
                      <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${MODE_COLORS[postMode]}`}>
                        {MODE_LABELS[postMode]}
                      </span>
                    )
                  )}
                </div>

                {isPostProcessing ? (
                  <p className="text-sm text-gray-400 italic py-2">Asking Gemini to fill gaps…</p>
                ) : (
                  <>
                    <textarea
                      value={postProcessed}
                      onChange={(e) => setPostProcessed(e.target.value)}
                      className="w-full text-xl font-medium text-gray-900 bg-transparent resize-none focus:outline-none focus:ring-2 focus:ring-purple-300 rounded-lg p-1 -ml-1"
                      style={{ fontFamily: "Noto Sans Devanagari, sans-serif" }}
                      rows={2}
                      disabled={stage !== "done"}
                    />
                    {stage === "done" && (
                      <p className="text-xs text-purple-500 mt-1">
                        ✎ You can edit this before speaking
                      </p>
                    )}
                  </>
                )}
              </div>

              {/* TTS buttons */}
              {stage === "done" && postProcessed && (
                <div className="grid grid-cols-2 gap-3">
                  <button
                    onClick={speakMarathi}
                    disabled={isSpeaking || isPostProcessing}
                    className="flex items-center justify-center gap-2 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-60 text-white font-medium py-3 px-4 rounded-xl transition-colors text-sm"
                  >
                    {isSpeaking ? <><Spinner /> Speaking…</> : "🔊 Speak in Marathi"}
                  </button>
                  <button
                    onClick={speakEnglish}
                    disabled={isSpeakingEnglish || isPostProcessing}
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

              <button onClick={reset} className="w-full text-sm text-gray-400 hover:text-gray-600 py-2">
                ↺ Try again
              </button>
            </div>
          )}
        </div>

        {/* Sarojini quote */}
        <div className="bg-gray-100 rounded-xl p-4">
          <p className="text-xs font-semibold text-gray-500 mb-1">Sarojini Shenoy, Taranath's wife</p>
          <p className="text-sm text-gray-700 italic">
            "The goal is not just deaf speech recognition — it is the restoration of dignity. Seeing Taranath
            hold up his phone and play it out to a vendor instead of being laughed at, is what this project is actually about."
          </p>
        </div>
      </main>
    </div>
  );
}
