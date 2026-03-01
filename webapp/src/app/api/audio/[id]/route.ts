import { NextRequest, NextResponse } from "next/server";

const UPSTREAM_BASE =
  "https://deafspeechcollector-production.up.railway.app/api/recordings";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  // Basic validation — id must be a positive integer
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
