import { NextRequest, NextResponse } from "next/server";

// Amchi Konkani recordings are stored in the Konkani Collector
const UPSTREAM_BASE =
  "https://konkanicollector-production.up.railway.app/api/recordings";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  if (!/^\d+$/.test(id)) {
    return NextResponse.json({ error: "Invalid id" }, { status: 400 });
  }

  const upstream = `${UPSTREAM_BASE}/${id}/audio`;

  let response: Response;
  try {
    response = await fetch(upstream, { cache: "force-cache" });
  } catch {
    return NextResponse.json({ error: "Upstream fetch failed" }, { status: 502 });
  }

  if (!response.ok) {
    return NextResponse.json(
      { error: `Upstream returned ${response.status}` },
      { status: response.status }
    );
  }

  const audioBuffer = await response.arrayBuffer();
  const contentType = response.headers.get("content-type") ?? "audio/wav";

  return new NextResponse(audioBuffer, {
    headers: {
      "Content-Type": contentType,
      "Cache-Control": "public, max-age=86400, immutable",
    },
  });
}
