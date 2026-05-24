/**
 * Unit tests for the anonymous session id helper.
 */

import { describe, expect, it, beforeEach, vi } from "vitest";
import { getSessionId, _resetSessionForTests } from "./session";

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
