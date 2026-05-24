/**
 * POST /api/feedback/update — proxies to konkani_collector
 * (POST /api/asr-demo/feedback/update).
 *
 * Called from /demo/live on every user interaction (thumbs click, edit
 * blur, TTS click) to patch the existing feedback row identified by
 * `id` returned from `init`.
 */

import { NextRequest } from "next/server";
import { proxyToCollector } from "@/lib/collector-proxy";

export async function POST(req: NextRequest) {
    return proxyToCollector(req, "/api/asr-demo/feedback/update");
}
