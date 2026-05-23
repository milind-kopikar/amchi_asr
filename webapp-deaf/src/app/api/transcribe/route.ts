/**
 * /api/transcribe — Next.js API route that proxies the recorded audio to the
 * RunPod serverless endpoint for the deaf speech ASR.
 *
 * The route is intentionally thin — the heavy lifting (validation, error
 * mapping, response normalisation) lives in ``src/lib/runpod-client.ts`` so
 * it can be unit-tested without spinning up Next.js. See
 * ``src/lib/runpod-client.test.ts`` for the test coverage.
 *
 * Configure via Railway environment variables:
 *   RUNPOD_API_KEY               (required)
 *   RUNPOD_DEAF_ENDPOINT_ID      (required)
 *
 * Request body (JSON): { "audio_base64": "<base64 16 kHz mono WAV>" }
 * Response (JSON):     { raw, corrected, mode, latency_ms } on 200
 *                      { error: <message> }                  on 4xx / 5xx
 */

import { NextRequest, NextResponse } from "next/server";

import { callRunpodTranscribe } from "@/lib/runpod-client";

export async function POST(req: NextRequest) {
  // 1. Parse + validate request body
  let body: { audio_base64?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json(
      { error: "Request body must be valid JSON" },
      { status: 400 },
    );
  }
  if (!body || typeof body.audio_base64 !== "string" || body.audio_base64.length === 0) {
    return NextResponse.json(
      { error: "Missing or empty 'audio_base64' field" },
      { status: 400 },
    );
  }

  // 2. Forward to RunPod via the thin client (lets us unit-test the logic)
  const result = await callRunpodTranscribe(
    process.env.RUNPOD_API_KEY ?? "",
    process.env.RUNPOD_DEAF_ENDPOINT_ID ?? "",
    body.audio_base64,
  );

  if (!result.ok) {
    return NextResponse.json({ error: result.error }, { status: result.status });
  }
  return NextResponse.json(result.output);
}
