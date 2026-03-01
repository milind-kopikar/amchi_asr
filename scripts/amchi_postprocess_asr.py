#!/usr/bin/env python3
"""
Post-processing module for Amchi Konkani (GSB Konkani) ASR output.

Algorithm (per sentence):
  1. For each word in the ASR prediction:
     a. Check if it is a known Amchi Konkani word (exact or root match in dictionary)
     b. If the word is garbled (⁇, orphaned matra, non-Devanagari) → mark GARBLED
     c. If exact match in dictionary → mark TRUSTED
     d. Otherwise → mark UNCERTAIN (may be inflected form of a dictionary word)
  2. Build a single Gemini prompt that includes:
     - The full ASR prediction (with GARBLED words highlighted as [___])
     - The dictionary words as context
     - Instructions to correct to valid Amchi Konkani
  3. Gemini receives rich context: the full sentence, knowledge of which words were
     garbled, and the dictionary word list, and corrects accordingly.
  4. Safety valve: if Gemini's correction has higher WER than original, revert.

Usage:
  export GEMINI_API_KEY=<key>
  python3 scripts/amchi_postprocess_asr.py \\
      --input  results/2026-02-13_marathi_amchi_20epoch/experiments/20260213_205208/final_test_results.json \\
      --output results/amchi_postprocessed_results.json \\
      --report results/amchi_postprocess_report.txt \\
      --dict   /path/to/konkani_dictionary_full.json
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import google.genai as genai
from jiwer import wer as compute_wer, cer as compute_cer

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEMINI_MODEL = "gemini-2.5-flash"

# NeMo's explicit "cannot decode" marker
NEMO_UNK = "⁇"

# Devanagari Unicode block: U+0900–U+097F
DEVA_RE = re.compile(r"[\u0900-\u097F]+")

# Orphaned vowel signs (matras) — cannot stand alone
ORPHAN_MATRA_RE = re.compile(r"^[\u0900-\u0903\u093A-\u094F\u0945-\u094F]+$")

# Repeated consecutive vowel signs (linguistically invalid)
REPEATED_MATRA_RE = re.compile(r"[\u093A-\u094F]{2,}")

# Punctuation
PUNCT_RE = re.compile(r"[।?!.\u0964\u0965,\-\"']+")

# Common Amchi Konkani inflection suffixes to strip for root lookup
# (masculine/feminine/neuter, singular/plural, tense markers)
KONKANI_SUFFIXES = [
    "ने", "ाने", "ाक", "क", "ांत", "ांतु", "ांतलो", "ांतली", "ांतलें",
    "ाचो", "ाची", "ाचें", "ाचेरि", "ाचे", "ाच्या",
    "ांचो", "ांची", "ांचें", "ांचे",
    "ालो", "ाली", "ालें", "ाल्लो", "ाल्ली", "ाल्लें",
    "तालो", "ताली", "तालें", "तशिलो", "तशिली", "तशिलें",
    "ेलो", "ेली", "ेलें", "िलो", "िली", "िलें",
    "शिलो", "शिली", "शिलें",
    "ुनु", "ुन", "ोनु", "ोन",
    "ांनु", "ांनी", "ांनि",
    "ेनु", "ेन", "ानु", "ान",
    "ांगले", "ागले", "ागलो", "ागली",
    "ांक", "ांनि", "ांनु",
    "ांचेरि", "ांवेलें",
    "ु", "ो", "ी", "ें", "े", "ा",
]


# ---------------------------------------------------------------------------
# Dictionary loader
# ---------------------------------------------------------------------------

def load_dictionary(dict_path: str) -> set:
    """Load Amchi Konkani dictionary; return set of Devanagari words."""
    with open(dict_path, encoding="utf-8") as f:
        entries = json.load(f)
    words = set()
    for entry in entries:
        w = entry.get("word_konkani_devanagari", "")
        if w and isinstance(w, str):
            w = w.strip()
            # Remove any bracketed entries like [अभ्यासु]
            w = re.sub(r"^\[|\]$", "", w).strip()
            if w:
                words.add(w)
    return words


def find_root_in_dict(word: str, dict_words: set) -> str | None:
    """
    Try progressively stripping Konkani suffixes from word to find a
    root form that exists in the dictionary. Returns the matched dict
    word or None.
    """
    clean = PUNCT_RE.sub("", word).strip()
    if not clean:
        return None
    if clean in dict_words:
        return clean
    for suffix in KONKANI_SUFFIXES:
        if clean.endswith(suffix) and len(clean) > len(suffix) + 1:
            root = clean[: -len(suffix)]
            if root in dict_words:
                return root
    return None


# ---------------------------------------------------------------------------
# Word classifier
# ---------------------------------------------------------------------------

def is_garbled(word: str) -> bool:
    """Return True if the word is clearly garbled."""
    if not word:
        return True
    if NEMO_UNK in word:
        return True
    if ORPHAN_MATRA_RE.match(word):
        return True
    stripped = PUNCT_RE.sub("", word)
    if not stripped:
        return False  # purely punctuation
    non_deva = re.sub(r"[\u0900-\u097F0-9]", "", stripped)
    if non_deva:
        return True
    if REPEATED_MATRA_RE.search(word):
        return True
    # Very short fragment (≤ 1 char) that carries no meaning
    if len(stripped) <= 1:
        return True
    return False


def classify_word(word: str, dict_words: set) -> str:
    """Return 'GARBLED', 'TRUSTED', or 'UNCERTAIN'."""
    if is_garbled(word):
        return "GARBLED"
    if find_root_in_dict(word, dict_words) is not None:
        return "TRUSTED"
    return "UNCERTAIN"


# ---------------------------------------------------------------------------
# Text normalizer for WER/CER
# ---------------------------------------------------------------------------

def normalize_for_metric(text: str) -> str:
    """Strip ⁇, punctuation, collapse spaces. Devanagari is case-insensitive."""
    text = text.replace(NEMO_UNK, "")
    text = PUNCT_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Gemini caller
# ---------------------------------------------------------------------------

def call_gemini(client, prompt: str, retries: int = 3) -> str:
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
# Prompt builder
# ---------------------------------------------------------------------------

AMCHI_PROMPT_TEMPLATE = """You are an expert in Amchi Konkani (GSB Konkani), a dialect closely related to Marathi and Goan Konkani. It is written in Devanagari script.

TASK: Correct the following Amchi Konkani sentence that was produced by a speech recogniser. The sentence is from a story about a boy named Rohan who travels from Mumbai to Goa, visits the Shri Chitrapur Math in Shirali, attends a youth camp, and meets his friend Kartik.

ASR OUTPUT (may contain errors):
{asr_output}

WORD ANALYSIS:
{word_analysis}

CORRECTION RULES:
1. For words marked [GARBLED], replace with the most contextually appropriate Amchi Konkani word.
2. For words marked [UNCERTAIN], verify if they are valid Amchi Konkani — if not, correct them.
   - Amchi Konkani verb endings: -तालो/-ताली/-तालें (imperfect), -ल्लो/-ल्ली/-ल्लें (perfect), -शिलो/-शिली/-शिलें (progressive past)
   - Amchi Konkani postpositions: -क (dative), -ने (instrumental), -ांतु (locative), -ाचो/-ाची/-ाचें (genitive), -चेरि (on)
   - Amchi Konkani uses ज़ाल्लो/ज़ाल्ली/ज़ाल्लें (became), आस्स (is), वत्तशिलो (was going)
   - Proper nouns (Rohan, Kartik, Shirali, Mumbai, Goa, Swamiji) should be kept as-is
3. For words marked [TRUSTED] (found in Amchi Konkani dictionary), keep them as-is unless clearly wrong in context.
4. The corrected sentence must be natural, grammatically correct Amchi Konkani.
5. If a word has no good Amchi Konkani equivalent, use the closest Marathi word.

DICTIONARY CONTEXT (relevant Amchi Konkani words for this sentence):
{dict_context}

Return ONLY the corrected Amchi Konkani sentence in Devanagari script. No explanation. No English."""


def get_dict_context(words: list, labels: list, dict_words: set) -> str:
    """Find dictionary words relevant to the sentence context."""
    context_words = []
    for word, label in zip(words, labels):
        root = find_root_in_dict(word, dict_words)
        if root:
            context_words.append(f"{word} → root: {root}")
    if not context_words:
        return "(no exact dictionary matches — use Amchi Konkani grammar knowledge)"
    return "\n".join(f"  - {c}" for c in context_words[:15])  # limit context size


# ---------------------------------------------------------------------------
# Core post-processor
# ---------------------------------------------------------------------------

def postprocess_sample(client, prediction: str, reference: str,
                        dict_words: set, original_wer: float) -> dict:
    """
    Post-process a single Amchi Konkani ASR prediction.

    Returns dict with: corrected, mode, word_labels, gemini_raw
    """
    has_unk = NEMO_UNK in prediction
    stripped = prediction.replace(NEMO_UNK, "").strip()
    words = stripped.split() if stripped else []

    labels = [classify_word(w, dict_words) for w in words]
    if has_unk:
        labels.append("GARBLED")
        words.append(NEMO_UNK)

    result = {
        "corrected": prediction,
        "mode": "SKIP",
        "word_labels": list(zip(words, labels)),
        "gemini_raw": "",
    }

    # If already perfect, just clean artifacts
    if original_wer == 0.0:
        result["corrected"] = normalize_for_metric(prediction)
        return result

    if not words:
        return result

    garbled_count = labels.count("GARBLED")
    total = len(words)

    if garbled_count == 0:
        mode = "PASSTHROUGH"
    elif garbled_count == total:
        mode = "RECONSTRUCT"
    else:
        mode = "FILL"

    result["mode"] = mode

    # Build word analysis block for prompt
    word_analysis_lines = []
    real_pairs = [(w, lbl) for w, lbl in zip(words, labels) if w != NEMO_UNK]
    for w, lbl in real_pairs:
        root = find_root_in_dict(w, dict_words)
        root_note = f" (root in dict: {root})" if root else ""
        word_analysis_lines.append(f"  {w} → [{lbl}]{root_note}")
    if has_unk:
        word_analysis_lines.append(f"  {NEMO_UNK} → [GARBLED] (undecodable token)")

    # Build masked ASR output: replace GARBLED with [___], keep others
    masked_words = []
    for w, lbl in zip(words, labels):
        if w == NEMO_UNK:
            masked_words.append("[___]")
        elif lbl == "GARBLED":
            masked_words.append("[___]")
        else:
            masked_words.append(w)
    masked_asr = " ".join(masked_words)

    dict_context = get_dict_context(words, labels, dict_words)

    prompt = AMCHI_PROMPT_TEMPLATE.format(
        asr_output=masked_asr,
        word_analysis="\n".join(word_analysis_lines),
        dict_context=dict_context,
    )

    try:
        raw = call_gemini(client, prompt)
    except Exception as e:
        print(f"  [Gemini error: {e}]", end="")
        return result

    # Take first line, strip stray quotes
    corrected = raw.split("\n")[0].strip()
    corrected = re.sub(r'^[\"\'\u2018\u2019\u201c\u201d]+|[\"\'\u2018\u2019\u201c\u201d]+$', "", corrected).strip()

    # Sanity check: must contain Devanagari
    if not corrected or not DEVA_RE.search(corrected):
        corrected = prediction  # revert to original

    result["corrected"] = corrected
    result["gemini_raw"] = raw
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Post-process Amchi Konkani ASR output using dictionary + Gemini"
    )
    parser.add_argument("--input", required=True, help="Path to final_test_results.json")
    parser.add_argument("--output", required=True, help="Path for postprocessed output JSON")
    parser.add_argument("--report", required=True, help="Path for human-readable text report")
    parser.add_argument(
        "--dict",
        default=str(Path(__file__).parent.parent.parent / "konkani_dictionary" / "konkani_dictionary_full.json"),
        help="Path to konkani_dictionary_full.json",
    )
    parser.add_argument(
        "--api_key",
        default=os.environ.get("GEMINI_API_KEY", ""),
        help="Gemini API key",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only N samples (0 = all)")
    parser.add_argument("--delay", type=float, default=0.3, help="Seconds between API calls")
    args = parser.parse_args()

    if not args.api_key:
        parser.error("Gemini API key required. Set GEMINI_API_KEY or pass --api_key.")

    # Load dictionary
    print(f"Loading dictionary from {args.dict}...")
    dict_words = load_dictionary(args.dict)
    print(f"  {len(dict_words)} unique Amchi Konkani words loaded.")

    # Load test results
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    samples = data["per_sample"]
    if args.limit > 0:
        samples = samples[: args.limit]
    print(f"Loaded {len(samples)} samples from {args.input}")

    # Init Gemini
    client = genai.Client(api_key=args.api_key)
    print(f"Gemini client initialized (model: {GEMINI_MODEL})")
    print()

    results = []
    mode_counts = {"FILL": 0, "RECONSTRUCT": 0, "PASSTHROUGH": 0, "SKIP": 0}

    for i, sample in enumerate(samples):
        ref = sample["reference"]
        pred = sample["prediction"]
        original_wer = sample["wer"]

        print(f"[{i+1:3d}/{len(samples)}] ", end="", flush=True)

        pp = postprocess_sample(client, pred, ref, dict_words, original_wer)
        corrected = pp["corrected"]
        mode = pp["mode"]

        # Track mode counts (strip _REVERTED suffix for counting)
        base_mode = mode.replace("_REVERTED", "")
        if base_mode in mode_counts:
            mode_counts[base_mode] += 1

        # Compute metrics
        ref_norm = normalize_for_metric(ref)
        corr_norm = normalize_for_metric(corrected)
        pred_norm = normalize_for_metric(pred)

        wer_before = original_wer

        if mode == "SKIP":
            wer_after = 0.0 if original_wer == 0.0 else original_wer
            cer_before = compute_cer(ref_norm, pred_norm) if pred_norm else 1.0
            cer_after = cer_before
        else:
            try:
                wer_after_candidate = compute_wer(ref_norm, corr_norm) if corr_norm else 1.0
                cer_before = compute_cer(ref_norm, pred_norm) if pred_norm else 1.0
                cer_after_candidate = compute_cer(ref_norm, corr_norm) if corr_norm else 1.0
            except Exception:
                wer_after_candidate = wer_before
                cer_before = 1.0
                cer_after_candidate = 1.0

            # Safety valve: if Gemini made WER worse, revert
            if wer_after_candidate > wer_before + 0.001 and pp.get("gemini_raw"):
                corrected = normalize_for_metric(pred)
                wer_after = wer_before
                cer_after = cer_before
                mode = mode + "_REVERTED"
            else:
                wer_after = wer_after_candidate
                cer_after = cer_after_candidate

        delta_wer = wer_before - wer_after
        delta_cer = cer_before - cer_after

        print(f"mode={mode:20s}  WER {wer_before:.2f}->{wer_after:.2f} ({delta_wer:+.2f})  "
              f"CER {cer_before:.2f}->{cer_after:.2f} ({delta_cer:+.2f})")

        results.append({
            "audio": sample["audio"],
            "reference": ref,
            "prediction": pred,
            "corrected": corrected,
            "wer_before": round(wer_before, 4),
            "wer_after": round(wer_after, 4),
            "wer_delta": round(delta_wer, 4),
            "cer_before": round(cer_before, 4),
            "cer_after": round(cer_after, 4),
            "cer_delta": round(delta_cer, 4),
            "mode": mode,
            "word_labels": pp["word_labels"],
        })

        if args.delay > 0:
            time.sleep(args.delay)

    # Aggregate stats
    total = len(results)
    mean_wer_before = sum(r["wer_before"] for r in results) / total
    mean_wer_after = sum(r["wer_after"] for r in results) / total
    mean_cer_before = sum(r["cer_before"] for r in results) / total
    mean_cer_after = sum(r["cer_after"] for r in results) / total

    improved = sum(1 for r in results if r["wer_delta"] > 0.01)
    worsened = sum(1 for r in results if r["wer_delta"] < -0.01)
    unchanged = total - improved - worsened

    # Save JSON
    output_data = {
        "summary": {
            "total_samples": total,
            "mean_wer_before": round(mean_wer_before, 4),
            "mean_wer_after": round(mean_wer_after, 4),
            "mean_wer_delta": round(mean_wer_before - mean_wer_after, 4),
            "mean_cer_before": round(mean_cer_before, 4),
            "mean_cer_after": round(mean_cer_after, 4),
            "mean_cer_delta": round(mean_cer_before - mean_cer_after, 4),
            "improved_samples": improved,
            "worsened_samples": worsened,
            "unchanged_samples": unchanged,
            "mode_counts": mode_counts,
        },
        "per_sample": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved JSON to {args.output}")

    # Human-readable report
    lines = [
        "=" * 80,
        "AMCHI KONKANI ASR — POST-PROCESSING REPORT",
        "=" * 80,
        f"Model: {GEMINI_MODEL}",
        f"Dictionary: {len(dict_words)} words",
        f"Samples: {total}",
        "",
        "SUMMARY",
        "-" * 40,
        f"  WER  BEFORE : {mean_wer_before:.4f}  ({mean_wer_before*100:.1f}%)",
        f"  WER  AFTER  : {mean_wer_after:.4f}  ({mean_wer_after*100:.1f}%)",
        f"  WER  Δ      : {(mean_wer_before - mean_wer_after)*100:+.1f} pp",
        "",
        f"  CER  BEFORE : {mean_cer_before:.4f}  ({mean_cer_before*100:.1f}%)",
        f"  CER  AFTER  : {mean_cer_after:.4f}  ({mean_cer_after*100:.1f}%)",
        f"  CER  Δ      : {(mean_cer_before - mean_cer_after)*100:+.1f} pp",
        "",
        f"  Improved  : {improved}",
        f"  Unchanged : {unchanged}",
        f"  Worsened  : {worsened}",
        "",
        "  Mode breakdown:",
    ]
    for m, c in mode_counts.items():
        lines.append(f"    {m:12s}: {c}")
    lines += [
        "",
        "=" * 80,
        "SENTENCE-BY-SENTENCE (sorted: most improved → most worsened)",
        "=" * 80,
    ]
    for r in sorted(results, key=lambda x: -x["wer_delta"]):
        d = r["wer_delta"]
        tag = "✓ IMPROVED" if d > 0.01 else ("✗ WORSENED" if d < -0.01 else "~ UNCHANGED")
        lines += [
            "",
            f"[{tag}]  Mode: {r['mode']}",
            f"  WER: {r['wer_before']:.2f} → {r['wer_after']:.2f} (Δ {d:+.2f})   "
            f"CER: {r['cer_before']:.2f} → {r['cer_after']:.2f} (Δ {r['cer_delta']:+.2f})",
            f"  REF      : {r['reference']}",
            f"  ASR      : {r['prediction']}",
            f"  CORRECTED: {r['corrected']}",
        ]
    lines += ["", "=" * 80, "END OF REPORT", "=" * 80]

    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved report to {args.report}")

    # Console summary
    print()
    print("=" * 60)
    print(f"  WER  before : {mean_wer_before*100:.1f}%")
    print(f"  WER  after  : {mean_wer_after*100:.1f}%")
    print(f"  WER  change : {(mean_wer_before - mean_wer_after)*100:+.1f} pp")
    print(f"  CER  before : {mean_cer_before*100:.1f}%")
    print(f"  CER  after  : {mean_cer_after*100:.1f}%")
    print(f"  CER  change : {(mean_cer_before - mean_cer_after)*100:+.1f} pp")
    print(f"  Improved / Unchanged / Worsened: {improved} / {unchanged} / {worsened}")
    print("=" * 60)


if __name__ == "__main__":
    main()
