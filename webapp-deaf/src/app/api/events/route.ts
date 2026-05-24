/**
 * POST /api/events — proxies to konkani_collector
 * (POST /api/asr-demo/events).
 *
 * Fire-and-forget product-metric click stream. Each row in
 * amchi_demo_events records one of: record_click, transcribe_click,
 * thumb_click, tts_click, edit_blur, survey_link_click, survey_submit.
 */

import { NextRequest } from "next/server";
import { proxyToCollector } from "@/lib/collector-proxy";

export async function POST(req: NextRequest) {
    return proxyToCollector(req, "/api/asr-demo/events");
}
