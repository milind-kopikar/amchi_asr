import { NextRequest, NextResponse } from "next/server";

// Translates Amchi Konkani → English via Gemini, then speaks with Google TTS
const GEMINI_URL =
  "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent";
const GOOGLE_TTS_URL =
  "https://texttospeech.googleapis.com/v1/text:synthesize";

const TRANSLATE_PROMPT = `You are an expert in Amchi Konkani (GSB Konkani), a language spoken by the Chitrapur Saraswat Brahmin community. It is closely related to Marathi and Goan Konkani, written in Devanagari script.

Translate the following Amchi Konkani sentence into natural English. If you are unsure of a specific word, use your knowledge of Marathi to infer the meaning.

Amchi Konkani sentence: {text}

Rules:
- Return ONLY the English translation — no explanation, no Devanagari, no original text
- Keep the translation concise and natural
- Preserve names (Rohan, Kartik, Swamiji, Shirali) as-is

English translation:`;

export async function POST(req: NextRequest) {
  const geminiKey = process.env.GEMINI_API_KEY;
  const ttsKey = process.env.GOOGLE_TTS_API_KEY;

  if (!geminiKey) {
    return NextResponse.json({ error: "Translation not configured" }, { status: 503 });
  }
  if (!ttsKey) {
    return NextResponse.json({ error: "TTS not configured" }, { status: 503 });
  }

  let text: string;
  try {
    const body = await req.json();
    text = String(body.text ?? "").trim();
  } catch {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  if (!text) {
    return NextResponse.json({ error: "text is required" }, { status: 400 });
  }

  // Step 1: Translate Amchi Konkani → English via Gemini
  const geminiRes = await fetch(`${GEMINI_URL}?key=${geminiKey}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ parts: [{ text: TRANSLATE_PROMPT.replace("{text}", text) }] }],
      generationConfig: { temperature: 0.2, maxOutputTokens: 200 },
    }),
  });

  if (!geminiRes.ok) {
    const detail = await geminiRes.text();
    return NextResponse.json(
      { error: "Translation failed", detail },
      { status: geminiRes.status }
    );
  }

  const geminiJson = await geminiRes.json();
  let englishText: string =
    geminiJson?.candidates?.[0]?.content?.parts?.[0]?.text?.trim() ?? "";
  englishText = englishText.split("\n")[0].trim().replace(/^["""]+|["""]+$/g, "").trim();

  if (!englishText) {
    return NextResponse.json({ error: "Translation returned empty result" }, { status: 502 });
  }

  // Step 2: English TTS (Indian English female voice)
  const ttsRes = await fetch(`${GOOGLE_TTS_URL}?key=${ttsKey}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      input: { text: englishText },
      voice: { languageCode: "en-IN", ssmlGender: "FEMALE" },
      audioConfig: { audioEncoding: "MP3" },
    }),
  });

  if (!ttsRes.ok) {
    const detail = await ttsRes.text();
    return NextResponse.json(
      { error: "TTS upstream error", detail },
      { status: ttsRes.status }
    );
  }

  const { audioContent } = await ttsRes.json();
  return NextResponse.json({ audioContent, englishText });
}
