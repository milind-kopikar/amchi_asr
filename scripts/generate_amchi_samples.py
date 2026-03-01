#!/usr/bin/env python3
"""
Generate webapp/public/amchi_samples.json from the Amchi Konkani post-processed results.

For each sample, calls Gemini to produce a romanized (English-alphabet) transliteration
of the reference sentence.

Usage:
  python3 scripts/generate_amchi_samples.py \
      --input  results/amchi_postprocessed_results.json \
      --output webapp/public/amchi_samples.json \
      --api_key AIzaSy...
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import google.genai as genai

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GEMINI_MODEL = "gemini-2.5-flash"

ROMANIZE_PROMPT = """Transliterate the following Amchi Konkani (GSB Konkani) sentence from Devanagari script into Roman/English alphabet.

Rules:
- Use standard phonetic transliteration (similar to IAST or common Marathi romanization)
- Preserve the meaning and sounds faithfully
- Do NOT translate to English — just transliterate character-by-character
- Return ONLY the romanized text, nothing else

Devanagari: {text}

Romanized:"""


def romanize(client, text: str) -> str:
    prompt = ROMANIZE_PROMPT.format(text=text)
    for attempt in range(3):
        try:
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            result = resp.text.strip().split("\n")[0].strip()
            result = re.sub(r'^[\"\'\u2018\u2019\u201c\u201d]+|[\"\'\u2018\u2019\u201c\u201d]+$', "", result).strip()
            return result
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  [Gemini error: {e}]")
                return ""
    return ""


def extract_recording_id(audio_path: str) -> int:
    """Extract numeric ID from path like 'data/amchi/test/audio/570.wav'"""
    stem = Path(audio_path).stem
    return int(stem)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/amchi_postprocessed_results.json")
    parser.add_argument("--output", default="webapp/public/amchi_samples.json")
    parser.add_argument("--api_key", default=os.environ.get("GEMINI_API_KEY", ""))
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    if not args.api_key:
        parser.error("Gemini API key required. Pass --api_key or set GEMINI_API_KEY.")

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    client = genai.Client(api_key=args.api_key)

    MODE_LABELS = {
        "FILL": "Dictionary-guided correction",
        "FILL_REVERTED": "Correction reverted (no improvement)",
        "RECONSTRUCT": "Full reconstruction",
        "RECONSTRUCT_REVERTED": "Reconstruction reverted",
        "PASSTHROUGH": "Light cleanup",
        "PASSTHROUGH_REVERTED": "Cleanup reverted (no improvement)",
        "SKIP": "Perfect transcription",
    }

    samples = []
    raw = data["per_sample"]
    print(f"Processing {len(raw)} samples...")

    for i, s in enumerate(raw):
        rec_id = extract_recording_id(s["audio"])
        print(f"[{i+1:2d}/{len(raw)}] ID={rec_id}  romanizing reference...", end=" ", flush=True)

        roman = romanize(client, s["reference"])
        print(roman[:50] if roman else "(failed)")

        samples.append({
            "id": rec_id,
            "reference_devanagari": s["reference"],
            "reference_roman": roman,
            "raw_asr": s["prediction"],
            "corrected": s["corrected"],
            "wer_before": s["wer_before"],
            "wer_after": s["wer_after"],
            "cer_before": s["cer_before"],
            "cer_after": s["cer_after"],
            "mode": s["mode"],
            "mode_label": MODE_LABELS.get(s["mode"], s["mode"]),
        })

        if args.delay > 0:
            time.sleep(args.delay)

    summary = data["summary"]
    output = {
        "meta": {
            "total_samples": len(samples),
            "good_demo_samples": sum(1 for s in samples if s["wer_after"] <= 0.75),
            "story": "रोहनाची कथा — Rohan's Journey to the Chitrapur Math",
            "model": "Fine-tuned IndicConformer (GSB Konkani, 20 epochs)",
            "postprocessing": f"Gemini {GEMINI_MODEL} + Amchi Konkani dictionary ({summary['total_samples']} samples)",
            "mean_wer_before": summary["mean_wer_before"],
            "mean_wer_after": summary["mean_wer_after"],
            "mode_legend": MODE_LABELS,
        },
        "samples": samples,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(samples)} samples to {args.output}")
    print(f"Good demo samples (wer_after <= 0.75): {output['meta']['good_demo_samples']}")


if __name__ == "__main__":
    main()
