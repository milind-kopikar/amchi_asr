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

/**
 * Factory: build a fake fetch that returns one JSON body per call, in order
 * (used to simulate a /runsync response followed by one or more /status
 * polls). Throws if called more times than there are bodies.
 */
function fakeFetchSequence(bodies: unknown[]): typeof fetch {
  let call = 0;
  return async () => {
    if (call >= bodies.length) {
      throw new Error(`fakeFetchSequence: no response configured for call #${call + 1}`);
    }
    const body = bodies[call];
    call += 1;
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };
}

/** Factory: build a fake fetch that always returns the same JSON body (used for infinite-poll tests). */
function fakeFetchAlwaysReturning(body: unknown): typeof fetch {
  return async () =>
    new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
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

describe("callRunpodTranscribe — cold-start polling", () => {
  it("typical: polls /status and succeeds once the job reaches COMPLETED", async () => {
    const result = await callRunpodTranscribe("k", "e", "QUFB", {
      fetchFn: fakeFetchSequence([
        { id: "job-1", status: "IN_PROGRESS" },
        { id: "job-1", status: "IN_PROGRESS" },
        {
          id: "job-1",
          status: "COMPLETED",
          output: { raw: "x", corrected: "y", mode: "FILL", latency_ms: { asr: 1, postprocess: 2, total: 3 } },
        },
      ]),
      pollIntervalMs: 5,
    });
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.output.corrected).toBe("y");
    }
  });

  it("polls GET /v2/{endpoint}/status/{id} with the Authorization header", async () => {
    const calls: { url: string; method?: string; headers?: Record<string, string> }[] = [];
    const fetchFn: typeof fetch = async (url, init) => {
      calls.push({ url: url.toString(), method: init?.method, headers: init?.headers as Record<string, string> });
      if (calls.length === 1) {
        return new Response(JSON.stringify({ id: "job-9", status: "IN_QUEUE" }), { status: 200 });
      }
      return new Response(
        JSON.stringify({
          id: "job-9",
          status: "COMPLETED",
          output: { raw: "a", corrected: "b", mode: "PASSTHROUGH", latency_ms: { asr: 0, postprocess: 0, total: 0 } },
        }),
        { status: 200 },
      );
    };
    await callRunpodTranscribe("my-key", "endpoint-xyz", "QUFB", { fetchFn, pollIntervalMs: 5 });

    expect(calls[1]?.url).toBe("https://api.runpod.ai/v2/endpoint-xyz/status/job-9");
    expect(calls[1]?.method).toBe("GET");
    expect(calls[1]?.headers?.Authorization).toBe("Bearer my-key");
  });

  it("edge: gives up with a 504 if the job never leaves IN_QUEUE before the deadline", async () => {
    const result = await callRunpodTranscribe("k", "e", "QUFB", {
      fetchFn: fakeFetchAlwaysReturning({ id: "job-2", status: "IN_QUEUE" }),
      timeoutMs: 20,
      pollIntervalMs: 5,
    });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(504);
      expect(result.error).toContain("job-2");
    }
  });

  it("malformed: a pending status with no job id cannot be polled and is reported", async () => {
    const result = await callRunpodTranscribe("k", "e", "QUFB", {
      fetchFn: fakeFetchAlwaysReturning({ status: "IN_PROGRESS" }),
      pollIntervalMs: 5,
    });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error).toContain("no job id");
    }
  });

  it("a FAILED status reached mid-poll is reported without further polling", async () => {
    const result = await callRunpodTranscribe("k", "e", "QUFB", {
      fetchFn: fakeFetchSequence([
        { id: "job-3", status: "IN_PROGRESS" },
        { id: "job-3", status: "FAILED", error: "CUDA OOM" },
      ]),
      pollIntervalMs: 5,
    });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error).toContain("FAILED");
      expect(result.error).toContain("CUDA OOM");
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
