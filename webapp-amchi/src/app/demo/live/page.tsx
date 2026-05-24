"use client";

/**
 * Amchi Konkani ASR demo page — LIVE recording variant.
 *
 *   1. useMediaRecorder captures audio from the mic
 *   2. audioBlobToBase64Wav converts the browser blob to 16 kHz mono WAV
 *   3. POST /api/transcribe → RunPod Run S → {raw, corrected, mode, latency_ms}
 *   4. POST /api/feedback/init creates a feedback row + uploads the WAV
 *      to R2. The returned id is used for all subsequent updates and
 *      events on this recording.
 *   5. The user can edit either box, thumbs-up/down either box, and pick
 *      which box to TTS. Every interaction is recorded:
 *        - per-row state via /api/feedback/update (last value wins)
 *        - per-click event via /api/events (every click captured)
 *   6. After at least one transcription, a survey link appears at the
 *      bottom that opens /demo/live/survey.
 *
 * No live Amchi Konkani TTS (Google Cloud doesn't support it). TTS
 * goes via /api/tts which Gemini-translates to English then synthesises.
 * Same /api/tts endpoint is called for both Raw and Corrected — only
 * the `text` field changes.
 */

import Link from "next/link";
import { useEffect, useState } from "react";

import { useMediaRecorder } from "@/lib/useMediaRecorder";
import { audioBlobToBase64Wav } from "@/lib/wav-encoder";
import { getSessionId } from "@/lib/session";
import {
  initFeedback,
  updateFeedback,
  recordEvent,
  type ThumbValue,
} from "@/lib/feedback-client";

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

const AMCHI_TTS_LANGUAGE = "gemini-amchi-en"; // marker for the tts_language column

function Spinner() {
  return (
    <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
    </svg>
  );
}

function cleanForTts(text: string): string {
  // Strip NeMo's undecodable marker and stray danda before sending to Gemini.
  return text.replace(/⁇/g, "").replace(/।/g, "").trim();
}

/**
 * Thumbs button group. Returns null (no rating), "up", or "down".
 * Clicking the active value clears it; clicking the other toggles.
 */
function Thumbs({
  value,
  disabled,
  onChange,
}: {
  value: ThumbValue;
  disabled?: boolean;
  onChange: (next: ThumbValue) => void;
}) {
  const baseBtn =
    "flex items-center justify-center w-9 h-9 rounded-full text-lg transition-colors disabled:opacity-40";
  return (
    <div className="flex items-center gap-2" role="group" aria-label="rate transcription">
      <button
        type="button"
        disabled={disabled}
        aria-pressed={value === "up"}
        aria-label="thumbs up"
        onClick={() => onChange(value === "up" ? null : "up")}
        className={`${baseBtn} ${
          value === "up"
            ? "bg-emerald-100 text-emerald-700 ring-2 ring-emerald-300"
            : "bg-gray-100 hover:bg-gray-200 text-gray-600"
        }`}
      >
        👍
      </button>
      <button
        type="button"
        disabled={disabled}
        aria-pressed={value === "down"}
        aria-label="thumbs down"
        onClick={() => onChange(value === "down" ? null : "down")}
        className={`${baseBtn} ${
          value === "down"
            ? "bg-rose-100 text-rose-700 ring-2 ring-rose-300"
            : "bg-gray-100 hover:bg-gray-200 text-gray-600"
        }`}
      >
        👎
      </button>
    </div>
  );
}

export default function LivePage() {
  const recorder = useMediaRecorder();
  const [sessionId, setSessionId] = useState<string>("");
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [transcribed, setTranscribed] = useState<TranscribeResponse | null>(null);
  const [feedbackId, setFeedbackId] = useState<string | null>(null);

  // Editable text for each card — initialised after transcribe.
  const [editedRaw, setEditedRaw] = useState<string>("");
  const [postProcessed, setPostProcessed] = useState<string>("");

  // Thumbs ratings for each card.
  const [thumbRaw, setThumbRaw] = useState<ThumbValue>(null);
  const [thumbCorrected, setThumbCorrected] = useState<ThumbValue>(null);

  // TTS playback — which side is currently speaking (null = idle).
  const [speakingFor, setSpeakingFor] = useState<"raw" | "corrected" | null>(null);

  const [englishText, setEnglishText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Show the survey link once at least one transcription has completed
  // this session.
  const [hasEverTranscribed, setHasEverTranscribed] = useState(false);

  // Resolve the session id once on mount (sessionStorage access must
  // happen client-side).
  useEffect(() => {
    setSessionId(getSessionId());
  }, []);

  // ────────────────────────────────────────────────────────────────────────
  // Recording → transcribing → init feedback
  // ────────────────────────────────────────────────────────────────────────

  function handleRecordClick() {
    // Fire-and-forget product event. No await — Record should feel instant.
    if (sessionId) {
      recordEvent({ session_id: sessionId, event_type: "record_click" });
    }
    recorder.start();
  }

  async function handleTranscribe() {
    if (!recorder.audioBlob) {
      setError("No recording available — please record first.");
      return;
    }
    setIsTranscribing(true);
    setError(null);
    setEnglishText(null);

    if (sessionId) {
      recordEvent({ session_id: sessionId, event_type: "transcribe_click" });
    }

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
      setEditedRaw(result.raw);
      setPostProcessed(result.corrected || result.raw);
      setHasEverTranscribed(true);

      // Best-effort init of the feedback row. If this fails we still let
      // the user see + interact with the transcription, but the
      // subsequent thumbs/edit/TTS clicks will be no-ops (gated below
      // by `feedbackId !== null`).
      try {
        const init = await initFeedback({
          audio_base64: audioBase64,
          raw: result.raw,
          corrected: result.corrected,
          mode: result.mode,
          latency_ms: result.latency_ms,
          session_id: sessionId,
          user_agent: typeof navigator !== "undefined" ? navigator.userAgent : undefined,
        });
        setFeedbackId(init.id);
      } catch (initErr) {
        // Don't surface to the user — feedback logging is best-effort.
        console.warn("feedback init failed:", initErr);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Transcribe failed");
    } finally {
      setIsTranscribing(false);
    }
  }

  // ────────────────────────────────────────────────────────────────────────
  // Per-card user interactions (edit, thumbs, TTS)
  // ────────────────────────────────────────────────────────────────────────

  function handleEditedRawBlur() {
    if (!feedbackId || !transcribed) return;
    // Only persist if the user actually changed the text.
    if (editedRaw === transcribed.raw) return;
    updateFeedback({ id: feedbackId, edited_raw: editedRaw });
    recordEvent({
      session_id: sessionId,
      feedback_id: feedbackId,
      event_type: "edit_blur",
      event_target: "raw",
      event_value: String(editedRaw.length),
    });
  }

  function handleEditedCorrectedBlur() {
    if (!feedbackId || !transcribed) return;
    if (postProcessed === transcribed.corrected) return;
    updateFeedback({ id: feedbackId, edited_corrected: postProcessed });
    recordEvent({
      session_id: sessionId,
      feedback_id: feedbackId,
      event_type: "edit_blur",
      event_target: "corrected",
      event_value: String(postProcessed.length),
    });
  }

  function handleThumb(target: "raw" | "corrected", next: ThumbValue) {
    if (target === "raw") setThumbRaw(next);
    else setThumbCorrected(next);

    if (!feedbackId) return;
    const field = target === "raw" ? "thumb_raw" : "thumb_corrected";
    updateFeedback({ id: feedbackId, [field]: next });
    recordEvent({
      session_id: sessionId,
      feedback_id: feedbackId,
      event_type: "thumb_click",
      event_target: target,
      event_value: next ?? "clear",
    });
  }

  async function speakFor(which: "raw" | "corrected") {
    if (speakingFor) return;
    const sourceText = which === "raw" ? editedRaw : postProcessed;
    const text = cleanForTts(sourceText);
    if (!text) return;

    setSpeakingFor(which);
    setError(null);

    if (feedbackId) {
      updateFeedback({
        id: feedbackId,
        tts_choice: which,
        tts_language: AMCHI_TTS_LANGUAGE,
      });
      recordEvent({
        session_id: sessionId,
        feedback_id: feedbackId,
        event_type: "tts_click",
        event_target: which,
        event_value: "english",
      });
    }

    try {
      const res = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error ?? "TTS error");
      if (json.translation) setEnglishText(json.translation);
      const audio = new Audio(`data:audio/mp3;base64,${json.audioContent}`);
      audio.onended = () => setSpeakingFor(null);
      audio.onerror = () => {
        setSpeakingFor(null);
        setError("Playback failed.");
      };
      await audio.play();
    } catch (err) {
      setError(err instanceof Error ? err.message : "TTS error");
      setSpeakingFor(null);
    }
  }

  function handleSurveyLinkClick() {
    if (sessionId) {
      recordEvent({ session_id: sessionId, event_type: "survey_link_click" });
    }
  }

  function resetAll() {
    recorder.reset();
    setTranscribed(null);
    setFeedbackId(null);
    setEditedRaw("");
    setPostProcessed("");
    setThumbRaw(null);
    setThumbCorrected(null);
    setSpeakingFor(null);
    setEnglishText(null);
    setError(null);
  }

  const showTranscribe = recorder.status === "ready" && !transcribed && !isTranscribing;
  const showResults = !!transcribed;

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="border-b border-gray-200 bg-white">
        <div className="max-w-lg mx-auto px-4 py-5 flex items-center gap-3">
          <Link href="/" className="text-gray-400 hover:text-gray-600 text-sm">← Back</Link>
          <div>
            <h1 className="text-lg font-bold text-gray-900">Amchi Konkani — Live Demo</h1>
            <p className="text-xs text-gray-500">Record → ASR (Run S) → corrected transcript → English</p>
          </div>
        </div>
      </header>

      <main className="max-w-lg mx-auto px-4 py-8 space-y-6">
        <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-4">
          {(recorder.status === "idle" || recorder.status === "error") && !transcribed && (
            <button
              onClick={handleRecordClick}
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

          {recorder.audioBlob && !transcribed && (
            <div className="text-xs text-gray-500 text-center">
              Captured {(recorder.audioBlob.size / 1024).toFixed(1)} KB &middot; {recorder.audioBlob.type}
            </div>
          )}

          {recorder.error && (
            <p className="text-xs text-red-600">{recorder.error}</p>
          )}

          {showResults && transcribed && (
            <div className="space-y-4 pt-2">
              {/* ────── Raw ASR card ────── */}
              <div className="bg-gray-50 border border-gray-200 rounded-xl p-4">
                <p className="text-xs text-gray-400 uppercase tracking-widest mb-2">
                  Raw ASR output (Run S model)
                </p>
                <textarea
                  value={editedRaw}
                  onChange={(e) => setEditedRaw(e.target.value)}
                  onBlur={handleEditedRawBlur}
                  className="w-full text-xl font-medium text-gray-900 bg-transparent resize-none focus:outline-none focus:ring-2 focus:ring-gray-300 rounded-lg p-1 -ml-1"
                  style={{ fontFamily: "Noto Sans Devanagari, sans-serif" }}
                  rows={2}
                />
                <p className="text-xs text-gray-400 mt-1">
                  ✎ Edit if needed · <span className="text-red-500 font-bold">⁇</span> = undecoded token
                </p>
                <div className="flex items-center justify-between mt-3">
                  <Thumbs value={thumbRaw} onChange={(v) => handleThumb("raw", v)} />
                  <button
                    onClick={() => speakFor("raw")}
                    disabled={speakingFor !== null || !editedRaw}
                    className="flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-60 text-white font-medium py-2 px-3 rounded-lg transition-colors text-xs"
                  >
                    {speakingFor === "raw" ? <><Spinner /> Speaking…</> : "🔊 Translate & speak"}
                  </button>
                </div>
              </div>

              {/* ────── Corrected card ────── */}
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
                  onBlur={handleEditedCorrectedBlur}
                  className="w-full text-xl font-medium text-gray-900 bg-transparent resize-none focus:outline-none focus:ring-2 focus:ring-purple-300 rounded-lg p-1 -ml-1"
                  style={{ fontFamily: "Noto Sans Devanagari, sans-serif" }}
                  rows={2}
                />
                <p className="text-xs text-purple-500 mt-1">✎ Edit before speaking if needed</p>
                <p className="text-xs text-gray-400 mt-1">
                  Latency: ASR {transcribed.latency_ms.asr}ms · Post-proc {transcribed.latency_ms.postprocess}ms
                </p>
                <div className="flex items-center justify-between mt-3">
                  <Thumbs value={thumbCorrected} onChange={(v) => handleThumb("corrected", v)} />
                  <button
                    onClick={() => speakFor("corrected")}
                    disabled={speakingFor !== null || !postProcessed}
                    className="flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-60 text-white font-medium py-2 px-3 rounded-lg transition-colors text-xs"
                  >
                    {speakingFor === "corrected" ? <><Spinner /> Speaking…</> : "🔊 Translate & speak"}
                  </button>
                </div>
              </div>

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

        {/* Survey link — visible after the first transcription only. */}
        {hasEverTranscribed && (
          <div className="text-center pt-2">
            <Link
              href="/demo/live/survey"
              onClick={handleSurveyLinkClick}
              className="text-sm text-purple-700 hover:text-purple-900 underline underline-offset-2"
            >
              📝 Tell us what you think
            </Link>
          </div>
        )}
      </main>
    </div>
  );
}
