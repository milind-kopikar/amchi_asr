/**
 * Unit tests for the browser-side feedback client.
 *
 * Each test injects a `fetchFn` mock; no real network requests happen.
 */

import { describe, expect, it, vi } from "vitest";
import {
    initFeedback,
    updateFeedback,
    recordEvent,
    submitSurvey,
    EVENT_TYPES,
} from "./feedback-client";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

type FetchInit = { method?: string; headers?: Record<string, string>; body?: string };
type FetchResponse = {
    ok: boolean;
    status: number;
    json: () => Promise<unknown>;
    text: () => Promise<string>;
};
type FetchMock = (input: string, init?: FetchInit) => Promise<FetchResponse>;

function makeFetch(response: {
    ok?: boolean;
    status?: number;
    body?: unknown;
    /** override json() to throw */
    jsonThrows?: boolean;
}) {
    const ok = response.ok ?? true;
    const status = response.status ?? (ok ? 200 : 500);
    const body = response.body ?? {};
    return vi.fn<FetchMock>(async () => ({
        ok,
        status,
        json: async () => {
            if (response.jsonThrows) throw new Error("malformed JSON");
            return body;
        },
        text: async () => JSON.stringify(body),
    }));
}

const VALID_INIT = {
    audio_base64: "AAAA",
    raw: "x",
    corrected: "y",
    mode: "FILL",
    latency_ms: { asr: 100, postprocess: 50, total: 150 },
    session_id: "s",
};

// ---------------------------------------------------------------------------
// initFeedback
// ---------------------------------------------------------------------------

describe("initFeedback", () => {
    it("happy path returns id and audio_url", async () => {
        const fetchFn = makeFetch({
            ok: true,
            status: 201,
            body: { id: "row-1", audio_url: "https://r2/row-1.wav" },
        });
        const result = await initFeedback(VALID_INIT, fetchFn);
        expect(result.id).toBe("row-1");
        expect(result.audio_url).toBe("https://r2/row-1.wav");
        expect(fetchFn).toHaveBeenCalledTimes(1);
        const call = fetchFn.mock.calls[0];
        expect(call[0]).toBe("/api/feedback/init");
        expect(call[1]?.method).toBe("POST");
        expect(JSON.parse(call[1]?.body as string).session_id).toBe("s");
    });

    it("tolerates null audio_url in response", async () => {
        const fetchFn = makeFetch({
            ok: true,
            status: 201,
            body: { id: "row-1", audio_url: null },
        });
        const result = await initFeedback(VALID_INIT, fetchFn);
        expect(result.audio_url).toBeNull();
    });

    it("throws on missing audio_base64", async () => {
        const fetchFn = vi.fn();
        await expect(
            initFeedback({ ...VALID_INIT, audio_base64: "" }, fetchFn)
        ).rejects.toThrow(/audio_base64/);
        expect(fetchFn).not.toHaveBeenCalled();
    });

    it("throws on missing session_id", async () => {
        const fetchFn = vi.fn();
        await expect(
            initFeedback({ ...VALID_INIT, session_id: "" }, fetchFn)
        ).rejects.toThrow(/session_id/);
    });

    it("throws on network error", async () => {
        const fetchFn = vi.fn(async () => {
            throw new Error("connection refused");
        });
        await expect(initFeedback(VALID_INIT, fetchFn)).rejects.toThrow(/network error/);
    });

    it("throws on HTTP 5xx, surfaces server error message", async () => {
        const fetchFn = makeFetch({ ok: false, status: 500, body: { error: "db down" } });
        await expect(initFeedback(VALID_INIT, fetchFn)).rejects.toThrow(/db down/);
    });

    it("throws on HTTP 4xx without a parseable body", async () => {
        const fetchFn = makeFetch({ ok: false, status: 400, jsonThrows: true });
        await expect(initFeedback(VALID_INIT, fetchFn)).rejects.toThrow(/HTTP 400/);
    });

    it("throws when response body is missing id", async () => {
        const fetchFn = makeFetch({ ok: true, status: 201, body: { audio_url: "x" } });
        await expect(initFeedback(VALID_INIT, fetchFn)).rejects.toThrow(/id/);
    });
});

// ---------------------------------------------------------------------------
// updateFeedback
// ---------------------------------------------------------------------------

describe("updateFeedback", () => {
    it("happy path with thumb_raw", async () => {
        const fetchFn = makeFetch({ ok: true, status: 200, body: { ok: true } });
        const result = await updateFeedback({ id: "r1", thumb_raw: "up" }, fetchFn);
        expect(result.ok).toBe(true);
        expect(JSON.parse(fetchFn.mock.calls[0][1]?.body as string)).toEqual({
            id: "r1",
            thumb_raw: "up",
        });
    });

    it("happy path with multiple fields", async () => {
        const fetchFn = makeFetch({ ok: true, status: 200, body: { ok: true } });
        const result = await updateFeedback(
            {
                id: "r1",
                edited_corrected: "fixed text",
                thumb_corrected: "down",
                tts_choice: "raw",
                tts_language: "en-IN",
            },
            fetchFn
        );
        expect(result.ok).toBe(true);
    });

    it("returns ok=false for missing id (without fetching)", async () => {
        const fetchFn = vi.fn();
        const result = await updateFeedback({ id: "" } as never, fetchFn);
        expect(result.ok).toBe(false);
        expect(fetchFn).not.toHaveBeenCalled();
    });

    it("returns ok=false for invalid thumb_raw (without fetching)", async () => {
        const fetchFn = vi.fn();
        const result = await updateFeedback(
            { id: "r1", thumb_raw: "maybe" as never },
            fetchFn
        );
        expect(result.ok).toBe(false);
        expect(fetchFn).not.toHaveBeenCalled();
    });

    it("returns ok=false for invalid tts_choice", async () => {
        const fetchFn = vi.fn();
        const result = await updateFeedback(
            { id: "r1", tts_choice: "neither" as never },
            fetchFn
        );
        expect(result.ok).toBe(false);
    });

    it("accepts thumb_raw=null (clears the rating)", async () => {
        const fetchFn = makeFetch({ ok: true, status: 200, body: { ok: true } });
        const result = await updateFeedback({ id: "r1", thumb_raw: null }, fetchFn);
        expect(result.ok).toBe(true);
    });

    it("returns ok=false on server 404 (id not found)", async () => {
        const fetchFn = makeFetch({ ok: false, status: 404, body: { error: "not found" } });
        const result = await updateFeedback({ id: "ghost", thumb_raw: "up" }, fetchFn);
        expect(result.ok).toBe(false);
        if (!result.ok) expect(result.status).toBe(404);
    });

    it("returns ok=false on network error (does not throw)", async () => {
        const fetchFn = vi.fn(async () => {
            throw new Error("offline");
        });
        const result = await updateFeedback({ id: "r1", thumb_raw: "up" }, fetchFn);
        expect(result.ok).toBe(false);
        if (!result.ok) expect(result.error).toMatch(/offline/);
    });
});

// ---------------------------------------------------------------------------
// recordEvent
// ---------------------------------------------------------------------------

describe("recordEvent", () => {
    it.each(EVENT_TYPES)("accepts event_type=%s", async (event_type) => {
        const fetchFn = makeFetch({ ok: true, status: 202, body: { ok: true } });
        const result = await recordEvent(
            { session_id: "s", event_type, event_target: "raw", event_value: "up" },
            fetchFn
        );
        expect(result.ok).toBe(true);
        const body = JSON.parse(fetchFn.mock.calls[0][1]?.body as string);
        expect(body.event_type).toBe(event_type);
    });

    it("rejects unknown event_type without fetching", async () => {
        const fetchFn = vi.fn();
        const result = await recordEvent(
            { session_id: "s", event_type: "rage_click" as never },
            fetchFn
        );
        expect(result.ok).toBe(false);
        expect(fetchFn).not.toHaveBeenCalled();
    });

    it("rejects missing session_id without fetching", async () => {
        const fetchFn = vi.fn();
        const result = await recordEvent(
            { session_id: "", event_type: "record_click" },
            fetchFn
        );
        expect(result.ok).toBe(false);
        expect(fetchFn).not.toHaveBeenCalled();
    });

    it("returns ok=false on server failure (does not throw)", async () => {
        const fetchFn = makeFetch({ ok: false, status: 503 });
        const result = await recordEvent(
            { session_id: "s", event_type: "record_click" },
            fetchFn
        );
        expect(result.ok).toBe(false);
    });

    it("returns ok=false on network error (does not throw)", async () => {
        const fetchFn = vi.fn(async () => {
            throw new Error("dns failure");
        });
        const result = await recordEvent(
            { session_id: "s", event_type: "record_click" },
            fetchFn
        );
        expect(result.ok).toBe(false);
    });
});

// ---------------------------------------------------------------------------
// submitSurvey
// ---------------------------------------------------------------------------

describe("submitSurvey", () => {
    it("happy path", async () => {
        const fetchFn = makeFetch({ ok: true, status: 201, body: { ok: true } });
        const result = await submitSurvey(
            { session_id: "s", q1_clarity: 4, q2_likelihood: 5, comments: "great" },
            fetchFn
        );
        expect(result.ok).toBe(true);
    });

    it("works without comments", async () => {
        const fetchFn = makeFetch({ ok: true, status: 201, body: { ok: true } });
        const result = await submitSurvey(
            { session_id: "s", q1_clarity: 3, q2_likelihood: 3 },
            fetchFn
        );
        expect(result.ok).toBe(true);
    });

    it("rejects q1 < 1 without fetching", async () => {
        const fetchFn = vi.fn();
        const result = await submitSurvey(
            { session_id: "s", q1_clarity: 0, q2_likelihood: 3 },
            fetchFn
        );
        expect(result.ok).toBe(false);
        expect(fetchFn).not.toHaveBeenCalled();
    });

    it("rejects q1 > 5", async () => {
        const fetchFn = vi.fn();
        const result = await submitSurvey(
            { session_id: "s", q1_clarity: 6, q2_likelihood: 3 },
            fetchFn
        );
        expect(result.ok).toBe(false);
    });

    it("rejects non-integer q2", async () => {
        const fetchFn = vi.fn();
        const result = await submitSurvey(
            { session_id: "s", q1_clarity: 3, q2_likelihood: 3.5 },
            fetchFn
        );
        expect(result.ok).toBe(false);
    });

    it("rejects missing session_id", async () => {
        const fetchFn = vi.fn();
        const result = await submitSurvey(
            { session_id: "", q1_clarity: 3, q2_likelihood: 3 },
            fetchFn
        );
        expect(result.ok).toBe(false);
    });

    it("rejects non-string comments", async () => {
        const fetchFn = vi.fn();
        const result = await submitSurvey(
            { session_id: "s", q1_clarity: 3, q2_likelihood: 3, comments: 42 as never },
            fetchFn
        );
        expect(result.ok).toBe(false);
    });

    it("returns ok=false on server failure", async () => {
        const fetchFn = makeFetch({ ok: false, status: 500, body: { error: "db ate it" } });
        const result = await submitSurvey(
            { session_id: "s", q1_clarity: 3, q2_likelihood: 3 },
            fetchFn
        );
        expect(result.ok).toBe(false);
        if (!result.ok) expect(result.error).toMatch(/db ate it/);
    });
});
