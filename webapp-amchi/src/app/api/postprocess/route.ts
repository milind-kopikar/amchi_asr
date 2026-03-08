import { NextRequest, NextResponse } from "next/server";

const GEMINI_URL =
  "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent";

const FILL_PROMPT = `You are an expert in Amchi Konkani (GSB Konkani), a language spoken by the Chitrapur Saraswat Brahmin community. It is written in Devanagari script and closely related to Marathi and Goan Konkani.

The following is an automatic speech recognition (ASR) output with some undecodable tokens marked as "⁇". Please fill in the gaps using your knowledge of Amchi Konkani and context from the surrounding words.

ASR output: {text}

Rules:
- Return ONLY the corrected Devanagari sentence — no explanation, no English, no markdown
- Replace each "⁇" with the most likely Amchi Konkani word given the context
- Do not add or remove words beyond replacing "⁇" tokens
- Preserve the original words that are not "⁇"

Corrected sentence:`;

const RECONSTRUCT_PROMPT = `You are an expert in Amchi Konkani (GSB Konkani), a language spoken by the Chitrapur Saraswat Brahmin community. It is written in Devanagari script and closely related to Marathi and Goan Konkani.

The following is a heavily garbled ASR output. Many tokens are undecodable ("⁇"). Using the legible fragments as clues, reconstruct the most likely Amchi Konkani sentence.

Garbled ASR output: {text}

Rules:
- Return ONLY the reconstructed Devanagari sentence — no explanation, no English, no markdown
- Use the legible words as anchors and fill in the rest coherently
- The sentence should sound natural in Amchi Konkani

Reconstructed sentence:`;

export async function POST(req: NextRequest) {
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

  // Count ⁇ tokens vs total words
  const tokens = text.split(/\s+/);
  const garbledCount = tokens.filter((t) => t.includes("⁇")).length;
  const garbledRatio = tokens.length > 0 ? garbledCount / tokens.length : 0;

  // Decide mode
  let mode: "PASSTHROUGH" | "FILL" | "RECONSTRUCT";
  if (garbledCount === 0) {
    mode = "PASSTHROUGH";
  } else if (garbledRatio <= 0.4) {
    mode = "FILL";
  } else {
    mode = "RECONSTRUCT";
  }

  if (mode === "PASSTHROUGH") {
    return NextResponse.json({ corrected: text, mode });
  }

  const geminiKey = process.env.GEMINI_API_KEY;
  if (!geminiKey) {
    // Fallback: strip ⁇ tokens
    const stripped = text.replace(/⁇/g, "").replace(/\s+/g, " ").trim();
    return NextResponse.json({ corrected: stripped, mode });
  }

  const prompt =
    mode === "FILL"
      ? FILL_PROMPT.replace("{text}", text)
      : RECONSTRUCT_PROMPT.replace("{text}", text);

  const res = await fetch(`${GEMINI_URL}?key=${geminiKey}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: { temperature: 0.2, maxOutputTokens: 300 },
    }),
  });

  if (!res.ok) {
    const detail = await res.text();
    return NextResponse.json(
      { error: "Gemini error", detail },
      { status: res.status }
    );
  }

  const json = await res.json();
  let corrected: string =
    json?.candidates?.[0]?.content?.parts?.[0]?.text?.trim() ?? "";
  corrected = corrected.split("\n")[0].trim().replace(/^["""]+|["""]+$/g, "").trim();

  if (!corrected) {
    const stripped = text.replace(/⁇/g, "").replace(/\s+/g, " ").trim();
    return NextResponse.json({ corrected: stripped, mode });
  }

  return NextResponse.json({ corrected, mode });
}
