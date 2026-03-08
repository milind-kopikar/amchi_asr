import { NextRequest, NextResponse } from "next/server";

const GEMINI_URL =
  "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent";

type Mode = "RECONSTRUCT" | "FILL" | "PASSTHROUGH";

function countGarbled(text: string): number {
  return (text.match(/⁇/g) ?? []).length;
}

function countWords(text: string): number {
  return text
    .replace(/⁇/g, "")
    .trim()
    .split(/\s+/)
    .filter(Boolean).length;
}

function decideMode(prediction: string): Mode {
  const garbled = countGarbled(prediction);
  if (garbled === 0) return "PASSTHROUGH";
  const total = garbled + countWords(prediction);
  return garbled / total > 0.4 ? "RECONSTRUCT" : "FILL";
}

const FILL_PROMPT = `You are a Marathi ASR post-processor. The following is an ASR output from a deaf speaker. Some tokens were undecodable and are marked as ⁇.

ASR output with gaps: "{prediction}"

Task: Replace each ⁇ with the most likely Marathi word to produce a fluent, grammatically correct Marathi sentence. Use the surrounding context as guidance.

Rules:
- Return ONLY the corrected Marathi sentence — no explanation, no English, no quotes
- Keep all legible words exactly as they appear
- Fill only the ⁇ gaps

Corrected sentence:`;

const RECONSTRUCT_PROMPT = `You are a Marathi ASR post-processor. The following is a heavily garbled ASR output from a deaf speaker. The ⁇ symbol marks undecodable audio. The few legible words are clues.

Garbled ASR output: "{prediction}"

Task: Reconstruct the most likely complete Marathi sentence. Use the legible words as anchors. The sentence is likely a simple everyday statement or question about daily activities.

Rules:
- Return ONLY the reconstructed Marathi sentence — no explanation, no English, no quotes
- Keep any legible words in the same relative order

Reconstructed sentence:`;

export async function POST(req: NextRequest) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      { error: "Post-processing not configured (GEMINI_API_KEY missing)" },
      { status: 503 }
    );
  }

  let prediction: string;
  try {
    const body = await req.json();
    prediction = String(body.prediction ?? "").trim();
  } catch {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  if (!prediction) {
    return NextResponse.json({ error: "prediction is required" }, { status: 400 });
  }

  const mode = decideMode(prediction);

  if (mode === "PASSTHROUGH") {
    return NextResponse.json({ result: prediction, mode });
  }

  const prompt = (mode === "RECONSTRUCT" ? RECONSTRUCT_PROMPT : FILL_PROMPT).replace(
    "{prediction}",
    prediction
  );

  const geminiRes = await fetch(`${GEMINI_URL}?key=${apiKey}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: { temperature: 0.2, maxOutputTokens: 200 },
    }),
  });

  if (!geminiRes.ok) {
    const detail = await geminiRes.text();
    return NextResponse.json(
      { error: "Gemini error", detail },
      { status: geminiRes.status }
    );
  }

  const json = await geminiRes.json();
  let result: string =
    json?.candidates?.[0]?.content?.parts?.[0]?.text?.trim() ?? "";
  result = result.split("\n")[0].trim().replace(/^["'"]+|["'"]+$/g, "").trim();

  // Fallback: strip ⁇ tokens if Gemini returns empty
  if (!result) {
    result = prediction.replace(/⁇/g, "").replace(/\s+/g, " ").trim();
  }

  return NextResponse.json({ result, mode });
}
