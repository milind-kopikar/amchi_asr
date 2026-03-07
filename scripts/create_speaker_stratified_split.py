#!/usr/bin/env python3
"""
Generate a speaker-stratified train/dev/test split for Amchi Konkani ASR.

Reads the existing story-based manifests, pools all samples, then re-splits
each speaker's recordings as:
  - 70% train / 15% dev / 15% test   (for speakers with >= 10 samples)
  - 100% train                        (for speakers with < 10 samples)

Audio files are NOT moved. The new manifests reference the original audio paths.

Usage:
    python3 scripts/create_speaker_stratified_split.py \
        --train_manifest  data/amchi/train/manifest.jsonl \
        --dev_manifest    data/amchi/dev/manifest.jsonl \
        --test_manifest   data/amchi/test/manifest.jsonl \
        --output_dir      data/amchi_stratified \
        --seed            42

Output:
    data/amchi_stratified/train/manifest.jsonl
    data/amchi_stratified/dev/manifest.jsonl
    data/amchi_stratified/test/manifest.jsonl
    data/amchi_stratified/split_report.txt
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_manifest(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def write_manifest(samples, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Speaker-stratified train/dev/test split")
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--dev_manifest",   required=True)
    parser.add_argument("--test_manifest",  required=True)
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--train_frac",     type=float, default=0.70)
    parser.add_argument("--dev_frac",       type=float, default=0.15)
    # test_frac = 1 - train_frac - dev_frac
    parser.add_argument("--min_samples_to_split", type=int, default=10,
                        help="Speakers with fewer samples go entirely to train")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.output_dir)

    # ── Load all samples from all three existing splits ───────────────────────
    all_samples = []
    for path in [args.train_manifest, args.dev_manifest, args.test_manifest]:
        loaded = load_manifest(path)
        all_samples.extend(loaded)
    print(f"Total samples loaded: {len(all_samples)}")

    # ── Group by speaker ──────────────────────────────────────────────────────
    by_speaker = defaultdict(list)
    for s in all_samples:
        spk = s.get("speaker_id", "unknown")
        by_speaker[spk].append(s)

    print(f"\nSpeakers found: {len(by_speaker)}")
    for spk, samples in sorted(by_speaker.items(), key=lambda x: -len(x[1])):
        print(f"  {spk.split('@')[0]:28s}: {len(samples):4d} samples")

    # ── Stratified split ──────────────────────────────────────────────────────
    train_samples, dev_samples, test_samples = [], [], []

    split_report_lines = [
        "Speaker-stratified split report",
        f"Seed: {args.seed}  |  train={args.train_frac:.0%}  dev={args.dev_frac:.0%}  "
        f"test={1-args.train_frac-args.dev_frac:.0%}  (min_to_split={args.min_samples_to_split})",
        "",
        f"  {'Speaker':<30}  {'Total':>5}  {'Train':>5}  {'Dev':>5}  {'Test':>5}  Note",
        "  " + "─" * 70,
    ]

    for spk, samples in sorted(by_speaker.items(), key=lambda x: -len(x[1])):
        shuffled = list(samples)
        rng.shuffle(shuffled)
        n = len(shuffled)

        if n < args.min_samples_to_split:
            # All samples go to train
            train_samples.extend(shuffled)
            note = "too few → all train"
            n_tr, n_de, n_te = n, 0, 0
        else:
            n_tr = round(n * args.train_frac)
            n_de = round(n * args.dev_frac)
            n_te = n - n_tr - n_de  # remainder goes to test

            # Guarantee at least 1 sample in dev and test for eligible speakers
            if n_de == 0 and n >= args.min_samples_to_split:
                n_de = 1
                n_tr = n - n_de - n_te
            if n_te == 0 and n >= args.min_samples_to_split:
                n_te = 1
                n_tr = n - n_de - n_te

            tr  = shuffled[:n_tr]
            de  = shuffled[n_tr:n_tr + n_de]
            te  = shuffled[n_tr + n_de:]
            n_te = len(te)

            # Assign new sample_ids so they reflect their new split
            for i, s in enumerate(tr):
                s = dict(s); s["sample_id"] = f"train_{len(train_samples)+i:04d}"
                train_samples.append(s)
            for i, s in enumerate(de):
                s = dict(s); s["sample_id"] = f"dev_{len(dev_samples)+i:04d}"
                dev_samples.append(s)
            for i, s in enumerate(te):
                s = dict(s); s["sample_id"] = f"test_{len(test_samples)+i:04d}"
                test_samples.append(s)

            note = ""
            pfx = spk.split("@")[0]
            split_report_lines.append(
                f"  {pfx:<30}  {n:>5}  {n_tr:>5}  {n_de:>5}  {n_te:>5}  {note}"
            )
            continue

        # For all-train speakers, still update sample_ids
        for i, s in enumerate(shuffled):
            s = dict(s); s["sample_id"] = f"train_{len(train_samples)+i:04d}"
            train_samples.append(s)

        pfx = spk.split("@")[0]
        split_report_lines.append(
            f"  {pfx:<30}  {n:>5}  {n_tr:>5}  {n_de:>5}  {n_te:>5}  {note}"
        )

    # Fix sample IDs sequentially after all speakers processed
    for i, s in enumerate(train_samples):
        s["sample_id"] = f"train_{i:04d}"
    for i, s in enumerate(dev_samples):
        s["sample_id"] = f"dev_{i:04d}"
    for i, s in enumerate(test_samples):
        s["sample_id"] = f"test_{i:04d}"

    # ── Totals ────────────────────────────────────────────────────────────────
    split_report_lines += [
        "  " + "─" * 70,
        f"  {'TOTAL':<30}  {len(all_samples):>5}  {len(train_samples):>5}  "
        f"{len(dev_samples):>5}  {len(test_samples):>5}",
        "",
        "Test speaker breakdown:",
    ]

    test_spk_counts = defaultdict(int)
    for s in test_samples:
        test_spk_counts[s.get("speaker_id", "unknown")] += 1
    for spk, cnt in sorted(test_spk_counts.items(), key=lambda x: -x[1]):
        split_report_lines.append(f"  {spk.split('@')[0]:<30}: {cnt:>4} test samples")

    split_report_lines += [
        "",
        "Note: audio files are NOT moved. Manifests reference original paths.",
        f"Note: text overlap between splits is expected and acceptable for CTC ASR.",
        f"      Different speakers produce different spectrograms for the same sentence.",
    ]

    report = "\n".join(split_report_lines)
    print("\n" + report)

    # ── Write manifests ───────────────────────────────────────────────────────
    write_manifest(train_samples, out / "train" / "manifest.jsonl")
    write_manifest(dev_samples,   out / "dev"   / "manifest.jsonl")
    write_manifest(test_samples,  out / "test"  / "manifest.jsonl")

    report_path = out / "split_report.txt"
    report_path.write_text(report, encoding="utf-8")

    print(f"\n✓ Manifests written to {out}/{{train,dev,test}}/manifest.jsonl")
    print(f"✓ Report written to {report_path}")
    print(f"\nNext step: run training with configs/amchi_konkani_run_c_stratified.yaml")


if __name__ == "__main__":
    main()
