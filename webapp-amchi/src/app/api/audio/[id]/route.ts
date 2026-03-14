import { NextRequest, NextResponse } from "next/server";

// Amchi Konkani recordings are stored in the Konkani Collector
const UPSTREAM_BASE =
  "https://konkanicollector-production.up.railway.app/api/recordings";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  if (!/^\d+$/.test(id)) {
    return NextResponse.json({ error: "Invalid id" }, { status: 400 });
  }

  const upstream = `${UPSTREAM_BASE}/${id}/audio`;

  // iOS (WebKit) requires range request support to play audio. Forward any
  // Range header from the client to the upstream server.
  const rangeHeader = req.headers.get("range");
  const fetchHeaders: HeadersInit = {};
  if (rangeHeader) fetchHeaders["Range"] = rangeHeader;

  let response: Response;
  try {
    response = await fetch(upstream, {
      cache: rangeHeader ? "no-store" : "force-cache",
      headers: fetchHeaders,
    });
  } catch {
    return NextResponse.json({ error: "Upstream fetch failed" }, { status: 502 });
  }

  if (!response.ok && response.status !== 206) {
    return NextResponse.json(
      { error: `Upstream returned ${response.status}` },
      { status: response.status }
    );
  }

  const audioBuffer = await response.arrayBuffer();
  const contentType = response.headers.get("content-type") ?? "audio/wav";

  const responseHeaders: Record<string, string> = {
    "Content-Type": contentType,
    "Accept-Ranges": "bytes",
    "Content-Length": String(audioBuffer.byteLength),
    "Cache-Control": rangeHeader ? "no-cache" : "public, max-age=86400, immutable",
  };

  const contentRange = response.headers.get("content-range");
  if (contentRange) responseHeaders["Content-Range"] = contentRange;

  return new NextResponse(audioBuffer, {
    status: response.status === 206 ? 206 : 200,
    headers: responseHeaders,
  });
}
