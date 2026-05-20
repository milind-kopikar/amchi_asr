/**
 * Unit tests for ``runpod-client.ts``.
 *
 * Uses an injected ``fetchFn`` so no real HTTP requests are made.
 *
 * @module runpod-client.test
 */

import { describe, it, expect } from "vitest";
import { callRunpodTranscribe } from "./runpod-client";

/** Factory: build a fake fetch that returns the given JSON body and status. */
function fakeFetchReturning(body: unknown, status = 200): typeof fetch {
  const responseInit: ResponseInit = {
    status,
    headers: { "Content-Type": "application/json" },
  };
  return async () =>
    new Response(typeof body === "string" ? body : JSON.stringify(body), responseInit);
}

/** Factory: build a fake fetch that throws (network error). */
function fakeFetchThrowing(message: string): typeof fetch {
  return async () => {
    throw new Error(message);
  };
}

describe("callRunpodTranscribe — validation", () => {
  it("rejects missing api key", async () => {
    const result = await callRunpodTranscribe("", "endpoint-id", "QUFB", {
      fetchFn: fakeFetchReturning({ status: "COMPLETED" }),
    });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(500);
      expect(result.error).toContain("RUNPOD_API_KEY");
    }
  });

  it("rejects missing endpoint id", async () => {
    const result = await callRunpodTranscribe("key", "", "QUFB", {
      fetchFn: fakeFetchReturning({ status: "COMPLETED" }),
    });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(500);
      expect(result.error).toContain("endpoint id");
    }
  });

  it("rejects empty audio payload", async () => {
    const result = await callRunpodTranscribe("key", "endpoint-id", "", {
      fetchFn: fakeFetchReturning({ status: "COMPLETED" }),
    });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(400);
    }
  });
});

describe("callRunpodTranscribe — success path", () => {
  it("typical: passes through worker output fields", async () => {
    const body = {
      status: "COMPLETED",
      output: {
        raw: "हे किती",
        corrected: "हे किती आहे?",
        mode: "FILL",
        latency_ms: { asr: 270, postprocess: 1450, total: 1720 },
      },
    };
    const result = await callRunpodTranscribe("k", "e", "QUFB", {
      fetchFn: fakeFetchReturning(body),
    });
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.output.raw).toBe("हे किती");
      expect(result.output.corrected).toBe("हे किती आहे?");
      expect(result.output.mode).toBe("FILL");
      expect(result.output.latency_ms.total).toBe(1720);
    }
  });

  it("output may be an array (some RunPod responses wrap it)", async () => {
    const body = {
      status: "COMPLETED",
      output: [{ raw: "x", corrected: "y", mode: "PASSTHROUGH",
                 latency_ms: { asr: 100, postprocess: 0, total: 100 } }],
    };
    const result = await callRunpodTranscribe("k", "e", "QUFB", {
      fetchFn: fakeFetchReturning(body),
    });
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.output.corrected).toBe("y");
    }
  });

  it("falls back to 'transcription' field for backward compat", async () => {
    const body = {
      status: "COMPLETED",
      output: { transcription: "old format" },
    };
    const result = await callRunpodTranscribe("k", "e", "QUFB", {
      fetchFn: fakeFetchReturning(body),
    });
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.output.raw).toBe("old format");
      expect(result.output.corrected).toBe("old format");
    }
  });

  it("missing latency_ms field defaults to zeros", async () => {
    const body = { status: "COMPLETED", output: { raw: "x", corrected: "y", mode: "M" } };
    const result = await callRunpodTranscribe("k", "e", "QUFB", {
      fetchFn: fakeFetchReturning(body),
    });
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.output.latency_ms).toEqual({ asr: 0, postprocess: 0, total: 0 });
    }
  });
});

describe("callRunpodTranscribe — error paths", () => {
  it("HTTP 4xx is reported as 502 with the body snippet", async () => {
    const result = await callRunpodTranscribe("k", "e", "QUFB", {
      fetchFn: fakeFetchReturning("Unauthorized", 401),
    });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(502);
      expect(result.error).toContain("HTTP 401");
    }
  });

  it("network error is reported as 502", async () => {
    const result = await callRunpodTranscribe("k", "e", "QUFB", {
      fetchFn: fakeFetchThrowing("connect ECONNREFUSED"),
    });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(502);
      expect(result.error).toContain("ECONNREFUSED");
    }
  });

  it("timeout is reported as 504", async () => {
    const result = await callRunpodTranscribe("k", "e", "QUFB", {
      fetchFn: fakeFetchThrowing("request aborted"),
    });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(504);
    }
  });

  it("non-COMPLETED status is reported as a worker error", async () => {
    const body = { status: "FAILED", error: "OOM" };
    const result = await callRunpodTranscribe("k", "e", "QUFB", {
      fetchFn: fakeFetchReturning(body),
    });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error).toContain("FAILED");
      expect(result.error).toContain("OOM");
    }
  });

  it("missing output field is reported", async () => {
    const result = await callRunpodTranscribe("k", "e", "QUFB", {
      fetchFn: fakeFetchReturning({ status: "COMPLETED" }),
    });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error).toContain("no 'output' field");
    }
  });

  it("worker output.error is surfaced as the response error", async () => {
    const body = { status: "COMPLETED", output: { error: "Audio too large" } };
    const result = await callRunpodTranscribe("k", "e", "QUFB", {
      fetchFn: fakeFetchReturning(body),
    });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error).toContain("Audio too large");
    }
  });
});

describe("callRunpodTranscribe — request shape", () => {
  it("sends Authorization: Bearer <key> and JSON body with audio_base64", async () => {
    let capturedUrl = "";
    let capturedInit: RequestInit | undefined;
    const captureFetch: typeof fetch = async (url, init) => {
      capturedUrl = url.toString();
      capturedInit = init;
      return new Response(JSON.stringify({ status: "COMPLETED",
        output: { raw: "", corrected: "", mode: "X",
                  latency_ms: { asr: 0, postprocess: 0, total: 0 } } }),
                          { status: 200 });
    };
    await callRunpodTranscribe("my-key", "endpoint-xyz", "QUFB", { fetchFn: captureFetch });

    expect(capturedUrl).toBe("https://api.runpod.ai/v2/endpoint-xyz/runsync");
    expect(capturedInit?.method).toBe("POST");
    const headers = capturedInit?.headers as Record<string, string>;
    expect(headers.Authorization).toBe("Bearer my-key");
    expect(headers["Content-Type"]).toBe("application/json");
    const body = JSON.parse(capturedInit?.body as string);
    expect(body.input.audio_base64).toBe("QUFB");
  });
});
