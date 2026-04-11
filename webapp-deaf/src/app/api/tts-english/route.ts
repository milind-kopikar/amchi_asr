import { NextRequest, NextResponse } from "next/server";

const GEMINI_URL =
  "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent";
const GOOGLE_TTS_URL =
  "https://texttospeech.googleapis.com/v1/text:synthesize";

const TRANSLATE_PROMPT = `You are an expert in Marathi, a language spoken in Maharashtra, India, written in Devanagari script.

Translate the following Marathi sentence into natural English.

Marathi sentence: {text}

Rules:
- Return ONLY the English translation — no explanation, no Devanagari, no original text
- Keep the translation concise and natural
- If the sentence is a question, preserve the question form

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
  let skipTranslation = false;
  try {
    const body = await req.json();
    text = String(body.text ?? "").trim();
    skipTranslation = body.skipTranslation === true;
  } catch {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  if (!text) {
    return NextResponse.json({ error: "text is required" }, { status: 400 });
  }

  // If text is already in English, skip Gemini translation
  if (skipTranslation) {
    const ttsKey = process.env.GOOGLE_TTS_API_KEY;
    if (!ttsKey) return NextResponse.json({ error: "TTS not configured" }, { status: 503 });
    const ttsRes = await fetch(`${GOOGLE_TTS_URL}?key=${ttsKey}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        input: { text },
        voice: { languageCode: "en-IN", ssmlGender: "FEMALE" },
        audioConfig: { audioEncoding: "MP3" },
      }),
    });
    if (!ttsRes.ok) {
      const detail = await ttsRes.text();
      return NextResponse.json({ error: "TTS upstream error", detail }, { status: ttsRes.status });
    }
    const { audioContent } = await ttsRes.json();
    return NextResponse.json({ audioContent, englishText: text });
  }

  // Step 1: Translate Marathi → English via Gemini
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
