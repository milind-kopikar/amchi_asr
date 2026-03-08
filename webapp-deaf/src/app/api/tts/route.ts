import { NextRequest, NextResponse } from "next/server";

const GOOGLE_TTS_URL =
  "https://texttospeech.googleapis.com/v1/text:synthesize";

export async function POST(req: NextRequest) {
  const apiKey = process.env.GOOGLE_TTS_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      { error: "TTS not configured" },
      { status: 503 }
    );
  }

  let text: string;
  let gender: "MALE" | "FEMALE" = "MALE";
  try {
    const body = await req.json();
    text = String(body.text ?? "").trim();
    if (body.gender === "FEMALE") gender = "FEMALE";
  } catch {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  if (!text) {
    return NextResponse.json({ error: "text is required" }, { status: 400 });
  }

  const response = await fetch(`${GOOGLE_TTS_URL}?key=${apiKey}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      input: { text },
      voice: { languageCode: "mr-IN", ssmlGender: gender },
      audioConfig: { audioEncoding: "MP3" },
    }),
  });

  if (!response.ok) {
    const detail = await response.text();
    return NextResponse.json(
      { error: "TTS upstream error", detail },
      { status: response.status }
    );
  }

  const { audioContent } = await response.json();
  return NextResponse.json({ audioContent });
}
