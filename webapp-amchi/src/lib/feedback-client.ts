/**
 * Browser-side client for the feedback / events / survey endpoints.
 *
 * All requests target the SAME-ORIGIN proxy routes (
 *   POST /api/feedback/init
 *   POST /api/feedback/update
 *   POST /api/events
 *   POST /api/survey
 * ) which inject the Bearer auth header and forward to the variant's
 * collector. The browser never sees COLLECTOR_API_KEY.
 *
 * Every function takes an optional `fetchFn` (defaults to global fetch)
 * so unit tests can inject a mock without spinning up Next.js.
 *
 * Failure semantics:
 *   - `initFeedback` rejects with an Error if the row could not be
 *     created — without an id, the rest of the session can't attach
 *     updates. Caller should handle (e.g., log + render fallback).
 *   - `updateFeedback`, `recordEvent`, `submitSurvey` are "best effort":
 *     they resolve to a status indicator and never throw. A failure here
 *     just means a click wasn't logged — the user-visible flow keeps
 *     working.
 */

// Caller passes in the chosen fetch implementation (defaulting to the
// browser's global one). Typed as the function shape we use rather than
// the broader DOM Fetch type so the tests don't need DOM lib types.
type FetchFn = (
    input: string,
    init?: { method?: string; headers?: Record<string, string>; body?: string }
) => Promise<{
    ok: boolean;
    status: number;
    json: () => Promise<unknown>;
    text: () => Promise<string>;
}>;

const defaultFetch: FetchFn = (input, init) =>
    (globalThis.fetch as unknown as FetchFn)(input, init);

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface InitFeedbackArgs {
    audio_base64: string;
    raw: string;
    corrected: string;
    mode?: string;
    latency_ms?: { asr?: number; postprocess?: number; total?: number };
    session_id: string;
    user_agent?: string;
}

export interface InitFeedbackResult {
    id: string;
    audio_url: string | null;
}

export type ThumbValue = "up" | "down" | null;
export type TtsChoice = "raw" | "corrected";

export interface UpdateFeedbackArgs {
    id: string;
    edited_raw?: string | null;
    edited_corrected?: string | null;
    thumb_raw?: ThumbValue;
    thumb_corrected?: ThumbValue;
    tts_choice?: TtsChoice;
    tts_language?: string;
}

export const EVENT_TYPES = [
    "record_click",
    "transcribe_click",
    "thumb_click",
    "tts_click",
    "edit_blur",
    "survey_link_click",
    "survey_submit",
] as const;
export type EventType = (typeof EVENT_TYPES)[number];

export interface RecordEventArgs {
    session_id: string;
    feedback_id?: string;
    event_type: EventType;
    event_target?: string;
    event_value?: string;
    user_agent?: string;
}

export interface SubmitSurveyArgs {
    session_id: string;
    q1_clarity: number;
    q2_likelihood: number;
    comments?: string;
    user_agent?: string;
}

export type BestEffortResult =
    | { ok: true; status: number }
    | { ok: false; status: number; error: string };

// ---------------------------------------------------------------------------
// initFeedback — REQUIRED to succeed (returns Promise<InitFeedbackResult>)
// ---------------------------------------------------------------------------

/**
 * Creates the feedback row right after a transcription completes. The
 * returned `id` is used as the key for all subsequent updateFeedback
 * and event calls within this transcription.
 *
 * Throws Error on failure — calling code should fall back to running
 * the demo WITHOUT feedback logging if this fails (rather than
 * blocking the UI).
 */
export async function initFeedback(
    args: InitFeedbackArgs,
    fetchFn: FetchFn = defaultFetch
): Promise<InitFeedbackResult> {
    if (typeof args.audio_base64 !== "string" || args.audio_base64.length === 0) {
        throw new Error("initFeedback: audio_base64 is required");
    }
    if (typeof args.session_id !== "string" || args.session_id.length === 0) {
        throw new Error("initFeedback: session_id is required");
    }

    let resp;
    try {
        resp = await fetchFn("/api/feedback/init", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(args),
        });
    } catch (err) {
        throw new Error(`initFeedback: network error: ${(err as Error).message}`);
    }

    if (!resp.ok) {
        let msg = `HTTP ${resp.status}`;
        try {
            const body = (await resp.json()) as { error?: string };
            if (body?.error) msg = body.error;
        } catch {
            // Body wasn't JSON; keep HTTP status as the message.
        }
        throw new Error(`initFeedback: ${msg}`);
    }

    let payload: unknown;
    try {
        payload = await resp.json();
    } catch (err) {
        throw new Error(`initFeedback: malformed response JSON: ${(err as Error).message}`);
    }
    const obj = payload as { id?: unknown; audio_url?: unknown } | null;
    if (!obj || typeof obj.id !== "string") {
        throw new Error("initFeedback: response missing string `id`");
    }
    const audio_url =
        typeof obj.audio_url === "string" ? obj.audio_url : null;
    return { id: obj.id, audio_url };
}

// ---------------------------------------------------------------------------
// updateFeedback — best effort
// ---------------------------------------------------------------------------

const THUMB_VALUES = new Set<ThumbValue>(["up", "down", null]);
const TTS_CHOICE_VALUES = new Set<TtsChoice>(["raw", "corrected"]);

/**
 * Patches one or more user-modifiable fields on a feedback row. Best-
 * effort: returns a status indicator rather than throwing, so callers
 * (e.g., onChange handlers) don't need try/catch.
 */
export async function updateFeedback(
    args: UpdateFeedbackArgs,
    fetchFn: FetchFn = defaultFetch
): Promise<BestEffortResult> {
    if (typeof args.id !== "string" || args.id.length === 0) {
        return { ok: false, status: 0, error: "updateFeedback: id is required" };
    }
    // Client-side validation mirrors the server's CHECK constraints so we
    // don't waste a round-trip on bad data. `undefined` means "don't set".
    if (args.thumb_raw !== undefined && !THUMB_VALUES.has(args.thumb_raw)) {
        return { ok: false, status: 0, error: "updateFeedback: thumb_raw must be 'up' | 'down' | null" };
    }
    if (args.thumb_corrected !== undefined && !THUMB_VALUES.has(args.thumb_corrected)) {
        return { ok: false, status: 0, error: "updateFeedback: thumb_corrected must be 'up' | 'down' | null" };
    }
    if (args.tts_choice !== undefined && !TTS_CHOICE_VALUES.has(args.tts_choice)) {
        return { ok: false, status: 0, error: "updateFeedback: tts_choice must be 'raw' | 'corrected'" };
    }

    try {
        const resp = await fetchFn("/api/feedback/update", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(args),
        });
        if (!resp.ok) {
            let msg = `HTTP ${resp.status}`;
            try {
                const body = (await resp.json()) as { error?: string };
                if (body?.error) msg = body.error;
            } catch {
                /* ignore */
            }
            return { ok: false, status: resp.status, error: msg };
        }
        return { ok: true, status: resp.status };
    } catch (err) {
        return { ok: false, status: 0, error: (err as Error).message };
    }
}

// ---------------------------------------------------------------------------
// recordEvent — best effort, fire-and-forget
// ---------------------------------------------------------------------------

const EVENT_TYPE_SET = new Set<string>(EVENT_TYPES);

/**
 * Records a single product-metrics event. Best-effort and intentionally
 * non-blocking — callers should not `await` if they can avoid it (an
 * unrecorded click is preferable to a stuck UI).
 */
export async function recordEvent(
    args: RecordEventArgs,
    fetchFn: FetchFn = defaultFetch
): Promise<BestEffortResult> {
    if (typeof args.session_id !== "string" || args.session_id.length === 0) {
        return { ok: false, status: 0, error: "recordEvent: session_id is required" };
    }
    if (!EVENT_TYPE_SET.has(args.event_type)) {
        return { ok: false, status: 0, error: `recordEvent: invalid event_type ${args.event_type}` };
    }

    try {
        const resp = await fetchFn("/api/events", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(args),
        });
        if (!resp.ok) {
            return { ok: false, status: resp.status, error: `HTTP ${resp.status}` };
        }
        return { ok: true, status: resp.status };
    } catch (err) {
        return { ok: false, status: 0, error: (err as Error).message };
    }
}

// ---------------------------------------------------------------------------
// submitSurvey — required to succeed (returns Promise<BestEffortResult>)
// ---------------------------------------------------------------------------

/**
 * Submits the post-demo NPS-style survey. Validates client-side that
 * the two Likert scores are integers in 1..5 and the comments (if any)
 * is a string.
 */
export async function submitSurvey(
    args: SubmitSurveyArgs,
    fetchFn: FetchFn = defaultFetch
): Promise<BestEffortResult> {
    if (typeof args.session_id !== "string" || args.session_id.length === 0) {
        return { ok: false, status: 0, error: "submitSurvey: session_id is required" };
    }
    if (!Number.isInteger(args.q1_clarity) || args.q1_clarity < 1 || args.q1_clarity > 5) {
        return { ok: false, status: 0, error: "submitSurvey: q1_clarity must be an integer in 1..5" };
    }
    if (!Number.isInteger(args.q2_likelihood) || args.q2_likelihood < 1 || args.q2_likelihood > 5) {
        return { ok: false, status: 0, error: "submitSurvey: q2_likelihood must be an integer in 1..5" };
    }
    if (args.comments !== undefined && args.comments !== null && typeof args.comments !== "string") {
        return { ok: false, status: 0, error: "submitSurvey: comments must be a string" };
    }

    try {
        const resp = await fetchFn("/api/survey", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(args),
        });
        if (!resp.ok) {
            let msg = `HTTP ${resp.status}`;
            try {
                const body = (await resp.json()) as { error?: string };
                if (body?.error) msg = body.error;
            } catch {
                /* ignore */
            }
            return { ok: false, status: resp.status, error: msg };
        }
        return { ok: true, status: resp.status };
    } catch (err) {
        return { ok: false, status: 0, error: (err as Error).message };
    }
}
