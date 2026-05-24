/**
 * POST /api/feedback/init — proxies to konkani_collector
 * (POST /api/asr-demo/feedback/init).
 *
 * Used by /demo/live right after a transcription completes. The
 * collector creates the feedback row, uploads the audio to R2 with
 * the row's UUID as the key, and returns { id, audio_url }.
 *
 * Env vars required on Railway (the webapp's service):
 *   - COLLECTOR_BASE_URL   e.g. https://konkanicollector-production.up.railway.app
 *   - COLLECTOR_API_KEY    shared secret matching the collector's env
 */

import { NextRequest } from "next/server";
import { proxyToCollector } from "@/lib/collector-proxy";

export async function POST(req: NextRequest) {
    return proxyToCollector(req, "/api/asr-demo/feedback/init");
}
