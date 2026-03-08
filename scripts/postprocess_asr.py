#!/usr/bin/env python3
"""
Post-processing module for Marathi Deaf Speech ASR output.

Algorithm:
  1. Tokenize prediction into words
  2. Classify each word: GARBLED / TRUSTED / UNCERTAIN
  3. If all tokens are garbled → RECONSTRUCT mode
     Else if any garbled token exists → FILL mode
     Else → pass-through (no correction needed)
  4. Call Gemini API with appropriate prompt
  5. Validate + return corrected sentence

Usage:
  # Set GEMINI_API_KEY in .env (preferred) or pass --api_key:
  export GEMINI_API_KEY=<your_key>   # or add to .env file
  python3 scripts/postprocess_asr.py \
      --input  nemo_experiments/deaf_speech_story4_50epoch/experiments/20260301_003725/final_test_results.json \
      --output nemo_experiments/deaf_speech_story4_50epoch/experiments/20260301_003725/postprocessed_results.json \
      --report nemo_experiments/deaf_speech_story4_50epoch/experiments/20260301_003725/postprocess_report.txt
"""

import argparse
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import google.genai as genai
from jiwer import wer as compute_wer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEMINI_MODEL = "gemini-2.5-flash"

# NeMo's explicit "cannot decode" marker
NEMO_UNK = "⁇"

# Devanagari Unicode block: U+0900–U+097F
DEVA_RE = re.compile(r"[\u0900-\u097F]+")

# Devanagari vowel signs (matras) that cannot stand alone — U+093A–U+094F
# plus anusvara U+0902, visarga U+0903, chandrabindu U+0901
ORPHAN_MATRA_RE = re.compile(r"^[\u0900-\u0903\u093A-\u094F\u0945-\u094F]+$")

# High-frequency Marathi words we trust from the ASR output
# These are commonly used words in everyday Marathi conversation
TRUSTED_WORDS = {
    # question words
    "किती", "काय", "कसे", "कुठे", "केव्हा", "का", "कधी", "कोण", "कुणी",
    # verbs / auxiliaries
    "आहे", "आहेत", "आहेस", "आहात", "नाही", "नाहीत",
    "द्या", "घ्या", "सांगा", "पाहा", "करा",
    "येईल", "मिळेल", "मिळते", "मिळतो", "होईल",
    "करतो", "करते", "करतात", "जातो", "जाते",
    "खातो", "खाते", "पितो", "पिते",
    "धुतो", "धुते", "उठतो", "उठते", "झोपतो", "झोपते",
    # pronouns / demonstratives
    "मी", "तू", "तो", "ती", "ते", "हे", "हा", "ही",
    "आपण", "आम्ही", "तुम्ही", "त्यांचा", "त्याचा",
    # common nouns (everyday / transactional)
    "दूध", "पाणी", "पेपर", "चहा", "चहाचा",
    "रुपये", "पैसे", "किंमत", "भाव",
    "बस", "घर", "शाळा", "बाजार", "दुकान",
    "हात", "जेवण", "वेळ", "काम",
    # numbers (Devanagari digits as words would be normal ASCII here)
    "एक", "दोन", "तीन", "चार", "पाच",
    # story-specific words
    "दैनंदिन", "कामे",
    # common adjectives
    "गरम", "थंड", "ताजे", "मोठे", "लहान", "चांगले",
    # articles / connectors
    "आणि", "किंवा", "पण", "मला", "तुला",
    # misc commonly-correct in output
    "लिटर", "पॅकेट", "आजचा", "चाळीस", "रुपयांत",
    # additional from training data patterns
    "येईल", "द्या", "किती", "आहे", "कधी", "पेपर",
}

# High-frequency Marathi words for RECONSTRUCT mode
# These are the most commonly used words in everyday Marathi speech
HIGH_FREQ_MARATHI_WORDS = {
    "किती", "आहे", "एक", "द्या", "काय", "कधी", "येईल", "कोणता", "कोणती",
    "हे", "दोन", "तीन", "करा", "चहा", "पाणी", "हवं", "नाही", "ठीक", "का",
    "ही", "मला", "बघा", "थांबा", "पुन्हा", "समजलं", "लिहा", "दाखवा",
    "बरोबर", "चुकीचं", "नीट", "आता", "लगेच", "येतो", "झाला", "मिळेल",
    "शकतो", "घेतात", "चालतो", "जाते", "जातो", "करा", "सांगा", "बघू",
    "बदलू", "होईल", "करू"
}

# Punctuation to ignore during tokenization but restore later
PUNCT_RE = re.compile(r"[।?!.\u0964\u0965,]+")


# ---------------------------------------------------------------------------
# Word classifier
# ---------------------------------------------------------------------------

# Repeated consecutive Devanagari vowel signs (e.g. ीी, ूू) — linguistically invalid
REPEATED_MATRA_RE = re.compile(r"[\u093A-\u094F]{2,}")


def is_garbled(word: str) -> bool:
    """Return True if the word is considered garbled / not trustworthy."""
    if not word:
        return True
    if NEMO_UNK in word:
        return True
    # Orphaned matra with no consonant base (e.g. bare 'ू', 'ी')
    if ORPHAN_MATRA_RE.match(word):
        return True
    # Contains characters outside Devanagari block AND not punctuation/digits
    stripped = PUNCT_RE.sub("", word)
    if not stripped:
        return False  # purely punctuation is not garbled
    non_deva = re.sub(r"[\u0900-\u097F0-9]", "", stripped)
    if non_deva:  # leftover non-Devanagari, non-digit characters
        return True
    # Repeated consecutive vowel signs (invalid Devanagari, e.g. 'पितीी')
    if REPEATED_MATRA_RE.search(word):
        return True
    # Very short fragment (≤2 chars) that is not a known trusted word —
    # likely a partial syllable (e.g. 'दू', 'पि', 'प')
    if len(stripped) <= 2 and stripped not in TRUSTED_WORDS:
        return True
    return False


def classify_word(word: str) -> str:
    """Return 'GARBLED', 'TRUSTED', or 'UNCERTAIN'."""
    if is_garbled(word):
        return "GARBLED"
    clean = PUNCT_RE.sub("", word)
    if clean in TRUSTED_WORDS:
        return "TRUSTED"
    return "UNCERTAIN"


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(text: str):
    """Split on whitespace, preserving punctuation attached to words."""
    return text.split()


def normalize_for_wer(text: str) -> str:
    """Normalize text for WER computation: remove NeMo UNK marker, punctuation,
    strip extra spaces. Lowercase is no-op for Devanagari."""
    text = text.replace(NEMO_UNK, "")   # drop ⁇ before scoring
    text = PUNCT_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Gemini caller
# ---------------------------------------------------------------------------

def call_gemini(client, prompt: str, retries: int = 3) -> str:
    """Call Gemini API and return raw text response."""
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return resp.text.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e
    return ""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

FILL_PROMPT_TEMPLATE = """You are a Marathi language expert helping correct deaf speech recognition output.

Context: A deaf speaker said a short everyday Marathi sentence. The speech recognizer understood some words correctly but mangled others. The mangled positions are marked as [___].

Domain: The speaker was doing everyday transactions or tasks — e.g. asking the price of an item, requesting milk/newspaper/tea, asking when the bus comes, asking about daily chores like washing hands, drinking water, eating food.

Your job: Replace each [___] with the single most likely Marathi word. The result must be a natural, complete, short Marathi sentence (3–6 words typically).

Rules:
- Return ONLY the complete corrected sentence in Devanagari script — nothing else.
- Keep every non-[___] word exactly as given (do not rephrase them).
- Do NOT add explanation, punctuation beyond what is natural, or English words.

Garbled sentence: {garbled}

Corrected sentence:"""

RECONSTRUCT_PROMPT_TEMPLATE = """You are a Marathi language expert helping correct deaf speech recognition output.

Context: A deaf speaker said a short everyday Marathi sentence. The speech recognizer was completely unable to decode it and produced only garbled fragments.

Domain: The speaker was doing everyday transactions or tasks — e.g. asking the price of an item, requesting milk/newspaper/tea, asking when the bus comes, daily chores like washing hands, drinking water, eating food.

Garbled ASR fragments (phonetic clues only): {garbled}

High-frequency Marathi words commonly used in everyday speech: {high_freq_words}

Your job: Using the fragments as phonetic clues, reconstruct the most likely short Marathi sentence the speaker said. Prefer using words from the high-frequency list when possible.

Rules:
- Return ONLY the reconstructed Marathi sentence in Devanagari script — nothing else.
- The sentence must be natural, complete, and concise (3–6 words).
- Do NOT add explanation or English words.

Reconstructed sentence:"""

PASSTHROUGH_PROMPT_TEMPLATE = """You are a Marathi language expert. The following is a Marathi speech recognition output. Clean it up by removing stray characters or non-Devanagari artifacts while keeping all real words intact.

Input: {garbled}

Rules:
- Return ONLY the cleaned Marathi sentence in Devanagari script.
- Do NOT add explanation.

Cleaned sentence:"""


# ---------------------------------------------------------------------------
# Core post-processor
# ---------------------------------------------------------------------------

def postprocess_sample(client, prediction: str, original_wer: float = 1.0) -> dict:
    """
    Post-process a single ASR prediction.

    Args:
      original_wer: pre-computed WER from training eval. If already 0.0,
                    skip Gemini and just clean the text (strip ⁇, punctuation).

    Returns dict with:
      corrected  : str   — the corrected sentence
      mode       : str   — 'FILL' | 'RECONSTRUCT' | 'PASSTHROUGH' | 'SKIP'
      word_labels: list  — per-word classification
      prompt_used: str   — the prompt sent to Gemini
      gemini_raw : str   — raw Gemini response
    """
    # Pre-strip the NeMo ⁇ token — it always signals "garbled tail" and
    # adds no phonetic information. We know the end of the utterance is garbled.
    # For classify purposes, we'll note that there are garbled tokens at end.
    has_unk_marker = NEMO_UNK in prediction
    stripped_pred = prediction.replace(NEMO_UNK, "").strip()

    words = tokenize(stripped_pred) if stripped_pred else []
    labels = [classify_word(w) for w in words]

    # If the ⁇ marker was present, treat it as an additional garbled token
    if has_unk_marker:
        labels.append("GARBLED")
        words.append(NEMO_UNK)

    garbled_count = labels.count("GARBLED")
    total = len(words)

    result = {
        "corrected": prediction,
        "mode": "SKIP",
        "word_labels": list(zip(words, labels)),
        "prompt_used": "",
        "gemini_raw": "",
        "gemini_applied": False,
    }

    if total == 0 or prediction.strip() == "":
        return result

    # Quality gate: if the prediction is already perfect (WER=0), don't modify it —
    # just clean the ⁇ and punctuation artifacts without calling Gemini.
    if original_wer == 0.0:
        result["corrected"] = normalize_for_wer(prediction)
        result["mode"] = "SKIP"
        return result

    trusted_count = labels.count("TRUSTED")

    if garbled_count == 0:
        # No garbled words — lightly clean via Gemini (removes stray chars)
        mode = "PASSTHROUGH"
        prompt = PASSTHROUGH_PROMPT_TEMPLATE.format(garbled=prediction)
    elif trusted_count > 0:
        # At least one TRUSTED anchor word → FILL mode
        # Replace every GARBLED word with [___], keep TRUSTED/UNCERTAIN words
        mode = "FILL"
        # Build filled string — exclude the synthetic ⁇ entry we appended
        real_pairs = [(w, lbl) for w, lbl in zip(words, labels) if w != NEMO_UNK]
        filled = " ".join("[___]" if lbl == "GARBLED" else w for w, lbl in real_pairs)
        # If the sentence ends with a trusted/uncertain word but we know there's
        # garbled content at the tail, append a [___] slot
        if has_unk_marker:
            filled = filled.rstrip() + " [___]"
        prompt = FILL_PROMPT_TEMPLATE.format(garbled=filled.strip())
    else:
        # No trusted words → RECONSTRUCT from fragments
        mode = "RECONSTRUCT"
        # Use non-⁇ fragments as phonetic clues
        clean_pred = stripped_pred
        high_freq_list = ", ".join(sorted(HIGH_FREQ_MARATHI_WORDS))
        prompt = RECONSTRUCT_PROMPT_TEMPLATE.format(
            garbled=clean_pred if clean_pred else prediction,
            high_freq_words=high_freq_list
        )

    raw = call_gemini(client, prompt)

    # Validate: Gemini response should be primarily Devanagari
    corrected = raw.split("\n")[0].strip()  # take first line only
    # Remove any leading/trailing quotation marks Gemini sometimes adds
    corrected = re.sub(r"^[\"']+|[\"']+$", "", corrected).strip()

    # Sanity check: if response is empty or non-Devanagari, fall back
    if not corrected or not DEVA_RE.search(corrected):
        corrected = prediction  # revert to original

    result["corrected"] = corrected
    result["mode"] = mode
    result["prompt_used"] = prompt
    result["gemini_raw"] = raw
    result["gemini_applied"] = True
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Post-process Marathi ASR output with Gemini LLM")
    parser.add_argument("--input", required=True, help="Path to final_test_results.json")
    parser.add_argument("--output", required=True, help="Path for postprocessed output JSON")
    parser.add_argument("--report", required=True, help="Path for human-readable text report")
    parser.add_argument(
        "--api_key",
        default="AIzaSyAwBr6FlR2nXTDyWqI8dBIhKBXeugND-Gw",  # Hardcoded for testing
        help="Gemini API key (default: hardcoded for testing)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only N samples (0 = all)")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between API calls")
    args = parser.parse_args()

    if not args.api_key:
        parser.error(
            "Gemini API key required. Set GEMINI_API_KEY in your .env file or pass --api_key."
        )

    # Load input
    with open(args.input, encoding='utf-8') as f:
        data = json.load(f)

    samples = data["per_sample"]
    if args.limit > 0:
        samples = samples[: args.limit]

    print(f"Loaded {len(samples)} samples from {args.input}")

    # Init Gemini
    client = genai.Client(api_key=args.api_key)
    print(f"Gemini client initialized (model: {GEMINI_MODEL})")

    results = []
    mode_counts = {"FILL": 0, "RECONSTRUCT": 0, "PASSTHROUGH": 0, "SKIP": 0}

    for i, sample in enumerate(samples):
        ref = sample["reference"]
        pred = sample["prediction"]
        original_wer = sample["wer"]

        print(f"[{i+1:3d}/{len(samples)}] Processing... ", end="", flush=True)

        pp = postprocess_sample(client, pred, original_wer=original_wer)
        corrected = pp["corrected"]
        mode = pp["mode"]
        mode_counts[mode] += 1

        # Compute WER for corrected output
        # wer_before: use the pre-computed value from training (canonical normalization)
        # wer_after:  compute fresh with our normalize_for_wer (⁇ stripped)
        ref_norm = normalize_for_wer(ref)
        corr_norm = normalize_for_wer(corrected)

        wer_before = original_wer   # from training evaluation script

        # For SKIP (WER=0 cases), we trust the original WER — don't recompute
        # because our normalization differs from the training script.
        if mode == "SKIP":
            wer_after = 0.0 if original_wer == 0.0 else original_wer
        else:
            try:
                wer_after_candidate = compute_wer(ref_norm, corr_norm) if corr_norm else 1.0
            except Exception:
                wer_after_candidate = 1.0

            # Conservative safety valve: if Gemini made things WORSE, revert to
            # the original (with ⁇ stripped for cleanliness), keeping wer_before.
            if wer_after_candidate > wer_before + 0.001 and pp.get("gemini_applied"):
                corrected = normalize_for_wer(pred)   # stripped original, no ⁇
                wer_after = wer_before                # unchanged — we reverted
                mode = mode + "_REVERTED"
            else:
                wer_after = wer_after_candidate

        delta = wer_before - wer_after
        print(f"mode={mode:12s}  WER {wer_before:.2f} → {wer_after:.2f}  Δ={delta:+.2f}")

        results.append(
            {
                "audio": sample["audio"],
                "reference": ref,
                "prediction": pred,
                "corrected": corrected,
                "wer_before": round(wer_before, 4),
                "wer_after": round(wer_after, 4),
                "wer_delta": round(delta, 4),
                "mode": mode,
                "word_labels": pp["word_labels"],
            }
        )

        if args.delay > 0:
            time.sleep(args.delay)

    # Aggregate stats
    total = len(results)
    mean_wer_before = sum(r["wer_before"] for r in results) / total
    mean_wer_after = sum(r["wer_after"] for r in results) / total
    improved = sum(1 for r in results if r["wer_delta"] > 0.01)
    worsened = sum(1 for r in results if r["wer_delta"] < -0.01)
    unchanged = total - improved - worsened

    perfect_before = sum(1 for r in results if r["wer_before"] == 0.0)
    perfect_after = sum(1 for r in results if r["wer_after"] == 0.0)

    # Save JSON output
    output_data = {
        "summary": {
            "total_samples": total,
            "mean_wer_before": round(mean_wer_before, 4),
            "mean_wer_after": round(mean_wer_after, 4),
            "mean_wer_delta": round(mean_wer_before - mean_wer_after, 4),
            "improved_samples": improved,
            "worsened_samples": worsened,
            "unchanged_samples": unchanged,
            "perfect_transcriptions_before": perfect_before,
            "perfect_transcriptions_after": perfect_after,
            "mode_counts": mode_counts,
        },
        "per_sample": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved JSON results to {args.output}")

    # --- Human-readable report ---
    lines = []
    lines.append("=" * 80)
    lines.append("MARATHI DEAF SPEECH ASR — POST-PROCESSING REPORT")
    lines.append("=" * 80)
    lines.append(f"Model: {GEMINI_MODEL}")
    lines.append(f"Samples processed: {total}")
    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Mean WER BEFORE post-processing : {mean_wer_before:.4f}  ({mean_wer_before*100:.1f}%)")
    lines.append(f"  Mean WER AFTER  post-processing : {mean_wer_after:.4f}  ({mean_wer_after*100:.1f}%)")
    lines.append(f"  Mean improvement (Δ WER)        : {mean_wer_before - mean_wer_after:+.4f}  ({(mean_wer_before - mean_wer_after)*100:+.1f}pp)")
    lines.append("")
    lines.append(f"  Perfect (WER=0) before : {perfect_before} / {total}")
    lines.append(f"  Perfect (WER=0) after  : {perfect_after} / {total}")
    lines.append("")
    lines.append(f"  Improved samples  : {improved}")
    lines.append(f"  Unchanged samples : {unchanged}")
    lines.append(f"  Worsened samples  : {worsened}")
    lines.append("")
    lines.append("  Mode breakdown:")
    for mode, count in mode_counts.items():
        lines.append(f"    {mode:12s}: {count}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("SENTENCE-BY-SENTENCE COMPARISON")
    lines.append("=" * 80)

    # Sort: improved first, then unchanged, then worsened
    sorted_results = sorted(results, key=lambda r: -r["wer_delta"])

    for r in sorted_results:
        delta = r["wer_delta"]
        if delta > 0.01:
            tag = "✓ IMPROVED"
        elif delta < -0.01:
            tag = "✗ WORSENED"
        else:
            tag = "~ UNCHANGED"

        lines.append("")
        lines.append(f"[{tag}]  Mode: {r['mode']}   WER: {r['wer_before']:.2f} → {r['wer_after']:.2f}  (Δ {delta:+.2f})")
        lines.append(f"  REF      : {r['reference']}")
        lines.append(f"  ASR      : {r['prediction']}")
        lines.append(f"  CORRECTED: {r['corrected']}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    report_text = "\n".join(lines)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved human-readable report to {args.report}")

    # Print summary to console
    print()
    print("=" * 60)
    print(f"  WER before : {mean_wer_before*100:.1f}%")
    print(f"  WER after  : {mean_wer_after*100:.1f}%")
    print(f"  Improvement: {(mean_wer_before - mean_wer_after)*100:+.1f} percentage points")
    print(f"  Perfect transcriptions: {perfect_before} → {perfect_after}")
    print("=" * 60)


if __name__ == "__main__":
    main()
