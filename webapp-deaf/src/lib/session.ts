/**
 * Stable per-tab anonymous session id.
 *
 * Used to group together all DB rows from a single demo session:
 *   - amchi_demo_feedback rows
 *   - amchi_demo_events rows
 *   - amchi_user_survey row
 *
 * Stored in `sessionStorage` (one origin / one tab) so refreshing the
 * page within the same tab keeps the id; closing the tab and reopening
 * gives a fresh id. This is intentional — survey + feedback + events
 * should correlate within a single demo session, not across days.
 *
 * Falls back to an in-memory id if `sessionStorage` is unavailable
 * (Safari private browsing, server-side rendering, sandboxed iframes).
 * In SSR contexts the in-memory id is per-module-import — fine for our
 * Next.js client-component usage but never relied on.
 */

const STORAGE_KEY = "asr_demo_session_id";

let inMemoryId: string | null = null;

/**
 * Generate a session id. UUID-shaped if `crypto.randomUUID` is available
 * (modern browsers + Node 19+); falls back to a math-random hex string
 * otherwise. Either is fine — uniqueness only needs to hold across
 * concurrent sessions, not globally forever.
 */
function generateId(): string {
    if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
        return crypto.randomUUID();
    }
    // Math.random fallback — 16 hex chars + timestamp; collision-rate is
    // negligible at demo-traffic scale.
    const rand = Math.random().toString(16).slice(2, 18).padEnd(16, "0");
    return `${Date.now().toString(16)}-${rand}`;
}

/**
 * Returns the current session id. Generates and persists one if none
 * exists yet. Safe to call repeatedly; calls are idempotent.
 */
export function getSessionId(): string {
    if (typeof window === "undefined") {
        // SSR / Node — return the in-memory id (or generate one).
        if (inMemoryId === null) inMemoryId = generateId();
        return inMemoryId;
    }

    try {
        const existing = window.sessionStorage.getItem(STORAGE_KEY);
        if (existing) return existing;
        const fresh = generateId();
        window.sessionStorage.setItem(STORAGE_KEY, fresh);
        return fresh;
    } catch {
        // sessionStorage threw — e.g. Safari private mode, third-party
        // iframe with storage access blocked. Fall back to in-memory.
        if (inMemoryId === null) inMemoryId = generateId();
        return inMemoryId;
    }
}

/**
 * Adopt a session id supplied externally (e.g., via a `?s=` URL query
 * param when navigating from /demo/live to /demo/live/survey). The
 * priority is:
 *
 *   1. If `candidate` is a non-empty string, use it and persist it back
 *      to sessionStorage. This is what makes new-tab navigation work —
 *      the recording-page session_id survives the link click.
 *   2. Else fall back to the existing sessionStorage value (same-tab
 *      navigation already has the right id).
 *   3. Else generate a fresh id (e.g., user landed on the survey
 *      page directly, no recording context).
 *
 * Returns the chosen id. Idempotent — calling repeatedly with the same
 * `candidate` is safe.
 */
export function adoptSessionId(candidate?: string): string {
    if (candidate && typeof candidate === "string" && candidate.length > 0) {
        // Sanity-cap the candidate length to defend against absurdly long
        // query strings. UUIDs are 36 chars; anything beyond ~128 is junk.
        const adopted = candidate.slice(0, 128);
        if (typeof window !== "undefined") {
            try {
                window.sessionStorage.setItem(STORAGE_KEY, adopted);
            } catch {
                // sessionStorage broken — fall back to in-memory.
                inMemoryId = adopted;
            }
        } else {
            inMemoryId = adopted;
        }
        return adopted;
    }
    return getSessionId();
}

/**
 * For tests only — clears both the storage and in-memory caches so a
 * fresh id is generated next call. Not exported in production usage.
 */
export function _resetSessionForTests(): void {
    inMemoryId = null;
    if (typeof window !== "undefined") {
        try {
            window.sessionStorage.removeItem(STORAGE_KEY);
        } catch {
            // ignore
        }
    }
}
