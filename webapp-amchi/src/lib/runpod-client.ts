/**
 * Thin RunPod client used by the /api/transcribe API route.
 *
 * Extracted as a pure helper (taking an injectable ``fetchFn``) so it can be
 * unit-tested without touching Next.js machinery or making real HTTP calls.
 *
 * @module runpod-client
 */

const RUNPOD_BASE_URL = "https://api.runpod.ai/v2";

/** Default per-request timeout. RunPod /runsync can hold while a worker is cold. */
export const DEFAULT_RUNSYNC_TIMEOUT_MS = 120_000;

/** Default delay between /status polls once /runsync hands back a pending job. */
export const DEFAULT_POLL_INTERVAL_MS = 2_000;

/**
 * Statuses RunPod uses while a job hasn't finished yet. /runsync only holds
 * the connection open for ~90s server-side — on a cold-started worker
 * (container pull + checkpoint download + model load) that is routinely
 * exceeded, so RunPod hands back one of these instead of the final result.
 * We must poll /status until a terminal status, not treat this as failure.
 */
const PENDING_STATUSES = new Set(["IN_QUEUE", "IN_PROGRESS"]);

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function networkErrorResult(err: unknown): { ok: false; status: number; error: string } {
  const message = err instanceof Error ? err.message : String(err);
  const isTimeout = message.includes("aborted") || message.includes("timeout");
  return { ok: false, status: isTimeout ? 504 : 502, error: `Network error: ${message}` };
}

/** Shape of the RunPod worker's output payload (after we unwrap arrays). */
export interface TranscribeOutput {
  raw: string;
  corrected: string;
  mode: string;
  latency_ms: { asr: number; postprocess: number; total: number };
}

/** Result discriminated union returned by ``callRunpodTranscribe``. */
export type TranscribeResult =
  | { ok: true; output: TranscribeOutput }
  | { ok: false; status: number; error: string };

interface RawRunpodOutput {
  raw?: string;
  corrected?: string;
  transcription?: string;
  mode?: string;
  latency_ms?: { asr?: number; postprocess?: number; total?: number };
  error?: string;
}

interface RawRunpodResponse {
  id?: string;
  status?: string;
  output?: RawRunpodOutput | RawRunpodOutput[];
  error?: string;
}

/** Unwrap a COMPLETED RunPod payload into a TranscribeResult (or an error if malformed). */
function finalizeCompleted(payload: RawRunpodResponse): TranscribeResult {
  let raw = payload.output;
  if (Array.isArray(raw)) raw = raw[0];
  if (!raw || typeof raw !== "object") {
    return { ok: false, status: 502, error: "RunPod returned no 'output' field" };
  }
  if (raw.error) {
    return { ok: false, status: 502, error: raw.error };
  }
  return {
    ok: true,
    output: {
      raw: raw.raw ?? raw.transcription ?? "",
      corrected: raw.corrected ?? raw.transcription ?? "",
      mode: raw.mode ?? "UNKNOWN",
      latency_ms: {
        asr: raw.latency_ms?.asr ?? 0,
        postprocess: raw.latency_ms?.postprocess ?? 0,
        total: raw.latency_ms?.total ?? 0,
      },
    },
  };
}

/**
 * POST audio_base64 to a RunPod /runsync endpoint and return a normalised
 * response.
 *
 * @param apiKey       Bearer token from $RUNPOD_API_KEY.
 * @param endpointId   RunPod endpoint id (the path segment after /v2/).
 * @param audioBase64  Base64-encoded 16 kHz mono WAV.
 * @param opts         Optional overrides — useful for tests.
 * @returns A ``TranscribeResult`` discriminated union. Never throws.
 */
export async function callRunpodTranscribe(
  apiKey: string,
  endpointId: string,
  audioBase64: string,
  opts: {
    fetchFn?: typeof fetch;
    timeoutMs?: number;
    pollIntervalMs?: number;
  } = {},
): Promise<TranscribeResult> {
  if (!apiKey) {
    return { ok: false, status: 500, error: "Missing RUNPOD_API_KEY" };
  }
  if (!endpointId) {
    return { ok: false, status: 500, error: "Missing RUNPOD endpoint id" };
  }
  if (typeof audioBase64 !== "string" || audioBase64.length === 0) {
    return { ok: false, status: 400, error: "Empty audio payload" };
  }

  const fetchFn = opts.fetchFn ?? fetch;
  const timeoutMs = opts.timeoutMs ?? DEFAULT_RUNSYNC_TIMEOUT_MS;
  const pollIntervalMs = opts.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS;
  const deadline = Date.now() + timeoutMs;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    let payload: RawRunpodResponse;
    try {
      const response = await fetchFn(
        `${RUNPOD_BASE_URL}/${endpointId}/runsync`,
        {
          method: "POST",
          headers: {
            "Authorization": `Bearer ${apiKey}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ input: { audio_base64: audioBase64 } }),
          signal: controller.signal,
        },
      );
      if (!response.ok) {
        const text = await response.text().catch(() => "");
        return {
          ok: false,
          status: 502,
          error: `RunPod HTTP ${response.status}: ${text.slice(0, 300)}`,
        };
      }
      payload = (await response.json()) as RawRunpodResponse;
    } catch (err) {
      return networkErrorResult(err);
    }

    let status = (payload.status || "").toUpperCase();

    // /runsync's own hold period (~90s) can elapse before a cold-started
    // worker finishes. Poll /status until a terminal state or our deadline.
    while (PENDING_STATUSES.has(status)) {
      if (!payload.id) {
        return {
          ok: false,
          status: 502,
          error: `RunPod worker status ${status}: response had no job id to poll`,
        };
      }
      if (Date.now() >= deadline) {
        return {
          ok: false,
          status: 504,
          error: `RunPod job ${payload.id} still ${status} after ${timeoutMs}ms`,
        };
      }
      await sleep(pollIntervalMs);
      try {
        const pollResponse = await fetchFn(
          `${RUNPOD_BASE_URL}/${endpointId}/status/${payload.id}`,
          {
            method: "GET",
            headers: { "Authorization": `Bearer ${apiKey}` },
            signal: controller.signal,
          },
        );
        if (!pollResponse.ok) {
          const text = await pollResponse.text().catch(() => "");
          return {
            ok: false,
            status: 502,
            error: `RunPod HTTP ${pollResponse.status}: ${text.slice(0, 300)}`,
          };
        }
        payload = (await pollResponse.json()) as RawRunpodResponse;
      } catch (err) {
        return networkErrorResult(err);
      }
      status = (payload.status || "").toUpperCase();
    }

    if (status && status !== "COMPLETED") {
      return {
        ok: false,
        status: 502,
        error: `RunPod worker status ${status}${payload.error ? ": " + payload.error : ""}`,
      };
    }

    return finalizeCompleted(payload);
  } finally {
    clearTimeout(timeoutId);
  }
}
