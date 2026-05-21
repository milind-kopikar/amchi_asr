"use client";

/**
 * Deaf Speech ASR demo page — LIVE recording variant.
 *
 * Replaces the previous hardcoded transcription flow with:
 *   1. ``useMediaRecorder`` hook captures audio from the mic
 *   2. ``audioBlobToBase64Wav`` converts the browser-native blob to 16 kHz mono WAV
 *   3. POST /api/transcribe forwards to the RunPod DS-D endpoint
 *   4. Display raw ASR + corrected text + post-processing mode
 *   5. TTS playback of the corrected text (Marathi + English via existing routes)
 *
 * Falls back to displaying an error if any step fails (mic permission,
 * conversion, RunPod outage).
 */

import Link from "next/link";
import { useState } from "react";

import { useMediaRecorder } from "@/lib/useMediaRecorder";
import { audioBlobToBase64Wav } from "@/lib/wav-encoder";

type PostMode = "FILL" | "RECONSTRUCT" | "PASSTHROUGH" | "SKIPPED" | "SKIP" | "PP_ERROR" | "UNKNOWN";

const MODE_LABELS: Record<string, string> = {
  FILL: "Filled gaps",
  RECONSTRUCT: "Reconstructed",
  PASSTHROUGH: "No changes needed",
  SKIPPED: "Post-processing skipped",
  SKIP: "Empty input",
  PP_ERROR: "Post-processing failed (raw shown)",
  UNKNOWN: "Mode unknown",
};

const MODE_COLORS: Record<string, string> = {
  FILL: "bg-blue-100 text-blue-800 border border-blue-200",
  RECONSTRUCT: "bg-orange-100 text-orange-800 border border-orange-200",
  PASSTHROUGH: "bg-green-100 text-green-800 border border-green-200",
  SKIPPED: "bg-gray-100 text-gray-700 border border-gray-200",
  SKIP: "bg-gray-100 text-gray-700 border border-gray-200",
  PP_ERROR: "bg-amber-100 text-amber-800 border border-amber-200",
  UNKNOWN: "bg-gray-100 text-gray-700 border border-gray-200",
};

interface TranscribeResponse {
  raw: string;
  corrected: string;
  mode: PostMode;
  latency_ms: { asr: number; postprocess: number; total: number };
}

function Spinner() {
  return (
    <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
    </svg>
  );
}

function RawText({ text }: { text: string }) {
  if (!text) return <span className="text-gray-400 italic">(no output)</span>;
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

export default function FinalPage() {
  const recorder = useMediaRecorder();
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [transcribed, setTranscribed] = useState<TranscribeResponse | null>(null);
  const [postProcessed, setPostProcessed] = useState<string>("");  // editable copy
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isSpeakingEnglish, setIsSpeakingEnglish] = useState(false);
  const [englishText, setEnglishText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleTranscribe() {
    if (!recorder.audioBlob) {
      setError("No recording available — please record first.");
      return;
    }
    setIsTranscribing(true);
    setError(null);
    setEnglishText(null);

    try {
      const audioBase64 = await audioBlobToBase64Wav(recorder.audioBlob);
      const res = await fetch("/api/transcribe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ audio_base64: audioBase64 }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error ?? `Transcribe HTTP ${res.status}`);
      const result = json as TranscribeResponse;
      setTranscribed(result);
      setPostProcessed(result.corrected || result.raw);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Transcribe failed");
    } finally {
      setIsTranscribing(false);
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
    setError(null);
    const text = postProcessed.replace(/⁇/g, "").replace(/।/g, "").trim();
    try {
      const res = await fetch("/api/tts-english", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),  // skipTranslation omitted → Gemini translates
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error ?? "TTS-English error");
      if (json.translation) setEnglishText(json.translation);
      const audio = new Audio(`data:audio/mp3;base64,${json.audioContent}`);
      audio.onended = () => setIsSpeakingEnglish(false);
      audio.onerror = () => setIsSpeakingEnglish(false);
      await audio.play();
    } catch (err) {
      setError(err instanceof Error ? err.message : "English TTS error");
      setIsSpeakingEnglish(false);
    }
  }

  function resetAll() {
    recorder.reset();
    setTranscribed(null);
    setPostProcessed("");
    setIsSpeaking(false);
    setIsSpeakingEnglish(false);
    setEnglishText(null);
    setError(null);
  }

  // Derived UI state
  const showTranscribe = recorder.status === "ready" && !transcribed && !isTranscribing;
  const showResults = !!transcribed;

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="border-b border-gray-200 bg-white">
        <div className="max-w-lg mx-auto px-4 py-5 flex items-center gap-3">
          <Link href="/" className="text-gray-400 hover:text-gray-600 text-sm">← Back</Link>
          <div className="flex items-center gap-2">
            <div>
              <h1 className="text-lg font-bold text-gray-900">Final Product</h1>
              <p className="text-xs text-gray-500">Record → transcribe (live ASR) → speak</p>
            </div>
            <span className="text-xs bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded-full font-medium shrink-0">
              BEST ⭐
            </span>
          </div>
        </div>
      </header>

      <main className="max-w-lg mx-auto px-4 py-8 space-y-6">
        <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-4">
          {/* Idle / Ready (no recording yet, or recording done but not transcribed) */}
          {(recorder.status === "idle" || recorder.status === "error") && !transcribed && (
            <button
              onClick={recorder.start}
              className="w-full flex items-center justify-center gap-3 bg-red-600 hover:bg-red-700 text-white font-semibold py-4 px-6 rounded-xl transition-colors"
            >
              <span className="text-xl">🎙</span> Record
            </button>
          )}

          {recorder.status === "requesting" && (
            <div className="w-full flex items-center justify-center gap-3 bg-yellow-50 border border-yellow-200 text-yellow-800 font-semibold py-4 px-6 rounded-xl">
              <Spinner /> Requesting microphone…
            </div>
          )}

          {recorder.status === "recording" && (
            <button
              onClick={recorder.stop}
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

          {recorder.status === "stopping" && (
            <div className="w-full flex items-center justify-center gap-3 bg-gray-100 text-gray-700 font-semibold py-4 px-6 rounded-xl">
              <Spinner /> Finishing recording…
            </div>
          )}

          {showTranscribe && (
            <button
              onClick={handleTranscribe}
              className="w-full flex items-center justify-center gap-3 bg-purple-600 hover:bg-purple-700 text-white font-semibold py-4 px-6 rounded-xl transition-colors"
            >
              Transcribe
            </button>
          )}

          {isTranscribing && (
            <div className="w-full flex items-center justify-center gap-3 bg-purple-100 border border-purple-200 text-purple-700 font-semibold py-4 px-6 rounded-xl">
              <Spinner /> Transcribing…
            </div>
          )}

          {showResults && (
            <button
              onClick={resetAll}
              className="w-full flex items-center justify-center gap-3 bg-red-600 hover:bg-red-700 text-white font-semibold py-4 px-6 rounded-xl transition-colors"
            >
              <span className="text-xl">🎙</span> Record Again
            </button>
          )}

          {/* Recording metadata once available */}
          {recorder.audioBlob && !transcribed && (
            <div className="text-xs text-gray-500 text-center">
              Captured {(recorder.audioBlob.size / 1024).toFixed(1)} KB &middot; {recorder.audioBlob.type}
            </div>
          )}

          {recorder.error && (
            <p className="text-xs text-red-600">{recorder.error}</p>
          )}

          {/* Results */}
          {showResults && transcribed && (
            <div className="space-y-4 pt-2">
              <div className="bg-gray-50 border border-gray-200 rounded-xl p-4">
                <p className="text-xs text-gray-400 uppercase tracking-widest mb-2">Raw ASR output (DS-D model)</p>
                <RawText text={transcribed.raw} />
                <p className="text-xs text-gray-400 mt-2">
                  <span className="text-red-500 font-bold">⁇</span> = token the model could not decode
                </p>
              </div>

              <div className="bg-white border-2 border-purple-200 rounded-xl p-4">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-xs text-gray-400 uppercase tracking-widest">Corrected transcript</p>
                  <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${MODE_COLORS[transcribed.mode] ?? MODE_COLORS.UNKNOWN}`}>
                    {MODE_LABELS[transcribed.mode] ?? transcribed.mode}
                  </span>
                </div>
                <textarea
                  value={postProcessed}
                  onChange={(e) => setPostProcessed(e.target.value)}
                  className="w-full text-xl font-medium text-gray-900 bg-transparent resize-none focus:outline-none focus:ring-2 focus:ring-purple-300 rounded-lg p-1 -ml-1"
                  style={{ fontFamily: "Noto Sans Devanagari, sans-serif" }}
                  rows={2}
                />
                <p className="text-xs text-purple-500 mt-1">✎ Edit before speaking if needed</p>
                <p className="text-xs text-gray-400 mt-1">
                  Latency: ASR {transcribed.latency_ms.asr}ms · Post-proc {transcribed.latency_ms.postprocess}ms
                </p>
              </div>

              {postProcessed && (
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
