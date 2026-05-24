/**
 * POST /api/survey — proxies to konkani_collector
 * (POST /api/asr-demo/survey).
 *
 * Submitted from the /demo/live/survey page with the two Likert-scale
 * answers (q1_clarity, q2_likelihood, 1..5) and optional comments.
 */

import { NextRequest } from "next/server";
import { proxyToCollector } from "@/lib/collector-proxy";

export async function POST(req: NextRequest) {
    return proxyToCollector(req, "/api/asr-demo/survey");
}
