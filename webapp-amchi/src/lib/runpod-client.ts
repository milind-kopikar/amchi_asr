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
  status?: string;
  output?: RawRunpodOutput | RawRunpodOutput[];
  error?: string;
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

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

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
    const message = err instanceof Error ? err.message : String(err);
    const isTimeout = message.includes("aborted") || message.includes("timeout");
    return {
      ok: false,
      status: isTimeout ? 504 : 502,
      error: `Network error: ${message}`,
    };
  } finally {
    clearTimeout(timeoutId);
  }

  // Validate worker status
  const status = (payload.status || "").toUpperCase();
  if (status && status !== "COMPLETED") {
    return {
      ok: false,
      status: 502,
      error: `RunPod worker status ${status}${payload.error ? ": " + payload.error : ""}`,
    };
  }

  // Unwrap output (may be a single object or a single-element array)
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
