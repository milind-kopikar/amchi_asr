#!/usr/bin/env python3
"""
Run inference on the test set using the best checkpoint and report WER/CER.
Use this to demonstrate the ~55% WER from the 20-epoch run.

  python scripts/demo_test_set_wer.py --checkpoint results/.../checkpoints/marathi_amchi_20epoch-epoch=18-val_wer=0.550.ckpt --manifest data/amchi/test/manifest.jsonl

Optional: --output results_demo.json to write per-sample results.
"""
import argparse
import json
import re
import sys
from pathlib import Path

# Repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.amchi_inference import load_model_from_ckpt, transcribe_audio


def _normalize(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"[^\w\s]", "", t)
    return " ".join(t.split())


def _wer(ref: str, hyp: str) -> float:
    r, h = _normalize(ref).split(), _normalize(hyp).split()
    m, n = len(r), len(h)
    if m == 0:
        return 0.0 if n == 0 else 1.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n] / m


def _cer(ref: str, hyp: str) -> float:
    r, h = list(_normalize(ref)), list(_normalize(hyp))
    m, n = len(r), len(h)
    if m == 0:
        return 0.0 if n == 0 else 1.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n] / m


def main():
    ap = argparse.ArgumentParser(description="Demo: run inference on test set and report WER/CER")
    ap.add_argument("--checkpoint", required=True, help="Path to best .ckpt")
    ap.add_argument("--manifest", default="data/amchi/test/manifest.jsonl", help="Test manifest JSONL")
    ap.add_argument("--output", help="Optional: write per-sample JSON here")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"Manifest not found: {manifest}", file=sys.stderr)
        sys.exit(1)

    lines = []
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(json.loads(line))

    audio_paths = []
    refs = []
    for rec in lines:
        path = rec.get("audio_filepath") or rec.get("audio_path")
        text = rec.get("text", "")
        if path:
            audio_paths.append(path)
            refs.append(text)

    if not audio_paths:
        print("No audio paths in manifest.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {args.checkpoint} ...")
    model = load_model_from_ckpt(args.checkpoint, device=args.device)
    print(f"Transcribing {len(audio_paths)} test files ...")

    per_sample = []
    for i, (path, ref) in enumerate(zip(audio_paths, refs)):
        if not Path(path).exists():
            print(f"  Skip (missing): {path}")
            continue
        preds = transcribe_audio(model, path)
        pred = preds[0] if preds else ""
        wer = _wer(ref, pred) if ref else None
        cer = _cer(ref, pred) if ref else None
        per_sample.append({
            "audio": path,
            "reference": ref,
            "prediction": pred,
            "wer": wer,
            "cer": cer,
        })
        print(f"  [{i+1}/{len(audio_paths)}] WER={wer:.2%} CER={cer:.2%}  {path}")

    valid = [p for p in per_sample if p["wer"] is not None]
    mean_wer = sum(p["wer"] for p in valid) / len(valid) if valid else None
    mean_cer = sum(p["cer"] for p in valid) / len(valid) if valid else None

    print()
    print("Summary (test set)")
    print(f"  Samples: {len(valid)}")
    print(f"  Mean WER: {mean_wer:.2%}" if mean_wer is not None else "  Mean WER: N/A")
    print(f"  Mean CER: {mean_cer:.2%}" if mean_cer is not None else "  Mean CER: N/A")

    if args.output:
        out = {
            "checkpoint": args.checkpoint,
            "manifest": str(manifest),
            "summary": {"total_samples": len(valid), "mean_wer": mean_wer, "mean_cer": mean_cer},
            "per_sample": per_sample,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"  Results written to {args.output}")


if __name__ == "__main__":
    main()
