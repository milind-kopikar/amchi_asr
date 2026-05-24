/**
 * Server-side helper for the Next.js routes that proxy to the
 * collector backend. Injects the Bearer auth header (so the browser
 * never sees `COLLECTOR_API_KEY`) and forwards the request body.
 *
 * Used by the four route files under `src/app/api/`:
 *   - feedback/init/route.ts    → POST /api/asr-demo/feedback/init
 *   - feedback/update/route.ts  → POST /api/asr-demo/feedback/update
 *   - events/route.ts           → POST /api/asr-demo/events
 *   - survey/route.ts           → POST /api/asr-demo/survey
 */

import { NextRequest, NextResponse } from "next/server";

/**
 * Forwards a Next.js POST request to the collector. Reads
 * `COLLECTOR_BASE_URL` and `COLLECTOR_API_KEY` from process.env.
 *
 * @param req             the Next request
 * @param upstreamPath    e.g. "/api/asr-demo/events" — appended to COLLECTOR_BASE_URL
 * @returns a NextResponse mirroring the collector's status + body
 */
export async function proxyToCollector(
    req: NextRequest,
    upstreamPath: string
): Promise<NextResponse> {
    const base = process.env.COLLECTOR_BASE_URL;
    const apiKey = process.env.COLLECTOR_API_KEY;

    if (!base || !apiKey) {
        // Fail closed — the collector is the only thing that can persist
        // feedback. Surface a clear 503 so the client knows it's a config
        // issue, not a transient network glitch.
        return NextResponse.json(
            { error: "COLLECTOR_BASE_URL / COLLECTOR_API_KEY not configured on this server" },
            { status: 503 }
        );
    }

    let body: string;
    try {
        // Read once; we forward as raw JSON text (no re-parse) so the
        // collector sees exactly what the client sent.
        body = await req.text();
    } catch {
        return NextResponse.json({ error: "request body is required" }, { status: 400 });
    }

    let upstream: Response;
    try {
        upstream = await fetch(`${base}${upstreamPath}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${apiKey}`,
            },
            body,
        });
    } catch (err) {
        return NextResponse.json(
            { error: `collector unreachable: ${(err as Error).message}` },
            { status: 502 }
        );
    }

    // Pass status + body straight through. Many endpoints return 201 or
    // 202; we don't want to flatten everything to 200.
    const text = await upstream.text();
    return new NextResponse(text, {
        status: upstream.status,
        headers: { "Content-Type": "application/json" },
    });
}
