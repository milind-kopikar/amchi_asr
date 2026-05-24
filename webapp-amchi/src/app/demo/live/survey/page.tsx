"use client";

/**
 * /demo/live/survey — NPS-style user satisfaction survey for the
 * Amchi Konkani live demo. Two 1-to-5 Likert questions plus an
 * optional comments box.
 *
 * Submitted via POST /api/survey, which proxies to konkani_collector's
 * /api/asr-demo/survey endpoint and INSERTs into amchi_user_survey.
 *
 * Linked from /demo/live's bottom-of-page "Tell us what you think"
 * link, which only appears after at least one transcription has
 * completed in the current session.
 */

import Link from "next/link";
import { useEffect, useState } from "react";

import { getSessionId, adoptSessionId } from "@/lib/session";
import { submitSurvey, recordEvent } from "@/lib/feedback-client";

type Score = 1 | 2 | 3 | 4 | 5;

const SCALE_LABELS: Record<Score, string> = {
  1: "not at all likely",
  2: "",
  3: "",
  4: "",
  5: "extremely likely",
};

function Spinner() {
  return (
    <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
    </svg>
  );
}

/**
 * A single 1–5 Likert radio scale. `name` must be unique per question
 * on the same page (used to group the radios).
 */
function LikertScale({
  name,
  value,
  onChange,
}: {
  name: string;
  value: Score | null;
  onChange: (v: Score) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-2 mt-2" role="radiogroup">
      {[1, 2, 3, 4, 5].map((n) => {
        const score = n as Score;
        const selected = value === score;
        return (
          <label
            key={n}
            className={`flex-1 flex flex-col items-center cursor-pointer rounded-lg border-2 py-2 transition-colors ${
              selected
                ? "border-purple-500 bg-purple-50 text-purple-800"
                : "border-gray-200 hover:border-gray-300 text-gray-600"
            }`}
          >
            <input
              type="radio"
              name={name}
              value={n}
              checked={selected}
              onChange={() => onChange(score)}
              className="sr-only"
            />
            <span className="text-lg font-semibold">{n}</span>
            <span className="text-[10px] leading-tight text-center h-4">
              {SCALE_LABELS[score]}
            </span>
          </label>
        );
      })}
    </div>
  );
}

export default function SurveyPage() {
  const [sessionId, setSessionId] = useState<string>("");
  const [q1, setQ1] = useState<Score | null>(null);
  const [q2, setQ2] = useState<Score | null>(null);
  const [comments, setComments] = useState<string>("");
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Resolve session id with priority:
  //   1. ?s=<id> query param (link from /demo/live carries the recording session)
  //   2. existing sessionStorage value (same-tab navigation)
  //   3. generate a fresh id (genuine new session)
  //
  // adoptSessionId() handles the priority + persists the chosen id back
  // to sessionStorage so subsequent calls are consistent.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const fromUrl = params.get("s") || undefined;
    setSessionId(adoptSessionId(fromUrl));
  }, []);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (q1 === null || q2 === null) {
      setError("Please answer both questions.");
      return;
    }
    setSubmitting(true);
    setError(null);

    const result = await submitSurvey({
      session_id: sessionId,
      q1_clarity: q1,
      q2_likelihood: q2,
      comments: comments.trim() || undefined,
      user_agent: typeof navigator !== "undefined" ? navigator.userAgent : undefined,
    });

    if (result.ok) {
      // Best-effort event row — the survey itself already landed.
      recordEvent({ session_id: sessionId, event_type: "survey_submit" });
      setSubmitted(true);
    } else if (!result.ok) {
      setError(result.error || "Could not submit survey. Please try again.");
    }
    setSubmitting(false);
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="border-b border-gray-200 bg-white">
        <div className="max-w-lg mx-auto px-4 py-5 flex items-center gap-3">
          <Link href="/demo/live" className="text-gray-400 hover:text-gray-600 text-sm">← Back to demo</Link>
          <div>
            <h1 className="text-lg font-bold text-gray-900">Quick feedback</h1>
            <p className="text-xs text-gray-500">Two quick questions — helps us improve the model</p>
          </div>
        </div>
      </header>

      <main className="max-w-lg mx-auto px-4 py-8">
        {submitted ? (
          <div className="bg-white rounded-xl border border-emerald-200 p-6 space-y-4 text-center">
            <p className="text-4xl">🙏</p>
            <h2 className="text-lg font-semibold text-emerald-700">Thanks!</h2>
            <p className="text-sm text-gray-600">
              Your feedback has been recorded.
            </p>
            <Link
              href="/demo/live"
              className="inline-block text-sm text-purple-700 hover:text-purple-900 underline underline-offset-2"
            >
              ← Back to the demo
            </Link>
          </div>
        ) : (
          <form
            onSubmit={onSubmit}
            className="bg-white rounded-xl border border-gray-200 p-6 space-y-6"
          >
            <div>
              <p className="text-sm font-medium text-gray-900">
                1. How likely is a reader to understand what you said, just by reading the transcription?
              </p>
              <LikertScale name="q1_clarity" value={q1} onChange={setQ1} />
            </div>

            <div>
              <p className="text-sm font-medium text-gray-900">
                2. How likely are you to use this app for transcribing Amchi Konkani?
              </p>
              <LikertScale name="q2_likelihood" value={q2} onChange={setQ2} />
            </div>

            <div>
              <label htmlFor="comments" className="text-sm font-medium text-gray-900">
                Anything else? <span className="text-gray-400">(optional)</span>
              </label>
              <textarea
                id="comments"
                value={comments}
                onChange={(e) => setComments(e.target.value)}
                rows={3}
                placeholder="Suggestions, bugs, what worked, what didn't…"
                className="w-full mt-2 text-sm text-gray-900 border border-gray-200 rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-purple-300"
              />
            </div>

            {error && <p className="text-xs text-red-600">{error}</p>}

            <button
              type="submit"
              disabled={submitting || q1 === null || q2 === null}
              className="w-full flex items-center justify-center gap-2 bg-purple-600 hover:bg-purple-700 disabled:opacity-60 text-white font-semibold py-3 px-4 rounded-xl transition-colors"
            >
              {submitting ? <><Spinner /> Submitting…</> : "Submit"}
            </button>
          </form>
        )}
      </main>
    </div>
  );
}
