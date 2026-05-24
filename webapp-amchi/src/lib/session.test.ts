/**
 * Unit tests for the anonymous session id helper.
 */

import { describe, expect, it, beforeEach, vi } from "vitest";
import { getSessionId, adoptSessionId, _resetSessionForTests } from "./session";

describe("getSessionId", () => {
    beforeEach(() => {
        _resetSessionForTests();
        // Restore globals between tests in case a test stubs them.
        vi.unstubAllGlobals();
    });

    it("returns a non-empty string", () => {
        const id = getSessionId();
        expect(typeof id).toBe("string");
        expect(id.length).toBeGreaterThan(0);
    });

    it("returns the same id on repeated calls in the same session", () => {
        const a = getSessionId();
        const b = getSessionId();
        const c = getSessionId();
        expect(a).toBe(b);
        expect(b).toBe(c);
    });

    it("returns UUID-like format when crypto.randomUUID is available", () => {
        const id = getSessionId();
        // RFC 4122 v4 shape: 8-4-4-4-12 hex chars. Crypto.randomUUID
        // is on by default in Node 19+ which vitest uses; if it's not
        // available we fall back to a different shape — accept either.
        const uuidLike = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
        const fallbackLike = /^[0-9a-f]+-[0-9a-f]{16}$/i;
        expect(uuidLike.test(id) || fallbackLike.test(id)).toBe(true);
    });

    it("persists across calls when sessionStorage is available", () => {
        // Stub window.sessionStorage with a working implementation
        const store: Record<string, string> = {};
        const fakeStorage = {
            getItem: (k: string) => (k in store ? store[k] : null),
            setItem: (k: string, v: string) => {
                store[k] = v;
            },
            removeItem: (k: string) => {
                delete store[k];
            },
        };
        vi.stubGlobal("window", { sessionStorage: fakeStorage });

        const first = getSessionId();
        // Simulate a "fresh module instance" by clearing in-memory only.
        // The id must still be returned from sessionStorage.
        _resetSessionForTests();
        // After reset, sessionStorage was also cleared — set a known value back.
        store["asr_demo_session_id"] = first;
        const second = getSessionId();
        expect(second).toBe(first);
    });

    it("falls back to in-memory when sessionStorage.setItem throws", () => {
        // Safari private-mode style: setItem throws QuotaExceededError.
        vi.stubGlobal("window", {
            sessionStorage: {
                getItem: () => null,
                setItem: () => {
                    throw new Error("QuotaExceededError");
                },
                removeItem: () => {},
            },
        });
        const a = getSessionId();
        const b = getSessionId();
        expect(a).toBe(b);
        expect(a.length).toBeGreaterThan(0);
    });

    it("falls back to in-memory when sessionStorage.getItem throws", () => {
        vi.stubGlobal("window", {
            sessionStorage: {
                getItem: () => {
                    throw new Error("access denied");
                },
                setItem: () => {},
                removeItem: () => {},
            },
        });
        const a = getSessionId();
        const b = getSessionId();
        expect(a).toBe(b);
    });

    it("works under SSR (no window)", () => {
        vi.stubGlobal("window", undefined);
        const a = getSessionId();
        const b = getSessionId();
        expect(a).toBe(b);
        expect(a.length).toBeGreaterThan(0);
    });
});

describe("adoptSessionId", () => {
    beforeEach(() => {
        _resetSessionForTests();
        vi.unstubAllGlobals();
    });

    it("adopts a candidate id verbatim", () => {
        const store: Record<string, string> = {};
        vi.stubGlobal("window", {
            sessionStorage: {
                getItem: (k: string) => (k in store ? store[k] : null),
                setItem: (k: string, v: string) => { store[k] = v; },
                removeItem: (k: string) => { delete store[k]; },
            },
        });
        const id = adoptSessionId("from-url-abc");
        expect(id).toBe("from-url-abc");
        // Persisted so getSessionId() returns the same id afterwards.
        expect(getSessionId()).toBe("from-url-abc");
    });

    it("falls back to getSessionId() when candidate is missing/empty", () => {
        const a = adoptSessionId();
        const b = adoptSessionId("");
        const c = adoptSessionId(undefined);
        expect(a).toBe(b);
        expect(b).toBe(c);
        expect(a.length).toBeGreaterThan(0);
    });

    it("overrides an existing sessionStorage value when given a candidate", () => {
        const store: Record<string, string> = { asr_demo_session_id: "old-id" };
        vi.stubGlobal("window", {
            sessionStorage: {
                getItem: (k: string) => (k in store ? store[k] : null),
                setItem: (k: string, v: string) => { store[k] = v; },
                removeItem: (k: string) => { delete store[k]; },
            },
        });
        expect(getSessionId()).toBe("old-id");
        const adopted = adoptSessionId("new-id-from-url");
        expect(adopted).toBe("new-id-from-url");
        expect(store["asr_demo_session_id"]).toBe("new-id-from-url");
        expect(getSessionId()).toBe("new-id-from-url");
    });

    it("caps absurdly long candidates at 128 chars", () => {
        const long = "x".repeat(500);
        const adopted = adoptSessionId(long);
        expect(adopted.length).toBe(128);
    });

    it("falls back to in-memory when sessionStorage.setItem throws", () => {
        vi.stubGlobal("window", {
            sessionStorage: {
                getItem: () => null,
                setItem: () => { throw new Error("denied"); },
                removeItem: () => {},
            },
        });
        const id = adoptSessionId("safari-private-id");
        expect(id).toBe("safari-private-id");
        // Subsequent getSessionId() should also return the same (from in-memory).
        expect(getSessionId()).toBe("safari-private-id");
    });
});
