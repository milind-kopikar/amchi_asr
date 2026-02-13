#!/usr/bin/env python3
"""
Validate one-sample smoke test results: CER and validation loss must improve over epochs.

Success: both val_loss and CER decrease from first to last epoch.
Failure: if either does not improve, exit with code 1.

Usage:
  python scripts/validate_smoke_one_sample.py [--experiment_dir RESULTS/SMOKE_TESTS/EXPERIMENTS/TIMESTAMP]
  If --experiment_dir is omitted, finds the latest experiment under results/smoke_tests/experiments/.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path


def get_mean_cer_from_samples_json(path: Path) -> float | None:
    """Read samples_epoch_XX.json and return mean CER over samples, or None if missing/invalid."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = data.get("samples", [])
        if not samples:
            return None
        cers = [s.get("cer") for s in samples if s.get("cer") is not None]
        return sum(cers) / len(cers) if cers else None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Validate one-sample smoke: CER and val_loss must improve.")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="Path to experiment dir (e.g. results/smoke_tests/experiments/20260213_201603). "
             "Default: latest under results/smoke_tests/experiments/.",
    )
    args = parser.parse_args()

    if args.experiment_dir:
        experiment_dir = Path(args.experiment_dir)
    else:
        base = Path("results/smoke_tests/experiments")
        if not base.exists():
            print(f"FAIL: Experiment base not found: {base}", file=sys.stderr)
            sys.exit(1)
        subdirs = sorted([d for d in base.iterdir() if d.is_dir()], reverse=True)
        if not subdirs:
            print(f"FAIL: No experiment subdirs under {base}", file=sys.stderr)
            sys.exit(1)
        experiment_dir = subdirs[0]
        print(f"Using latest experiment: {experiment_dir}")

    if not experiment_dir.is_dir():
        print(f"FAIL: Not a directory: {experiment_dir}", file=sys.stderr)
        sys.exit(1)

    # 1. Validation loss from epoch_metrics.csv
    csv_path = experiment_dir / "epoch_metrics.csv"
    if not csv_path.exists():
        print(f"FAIL: Missing {csv_path}", file=sys.stderr)
        sys.exit(1)

    val_losses = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                v = float(row.get("val_loss", ""))
                val_losses.append(v)
            except (ValueError, KeyError):
                pass

    if len(val_losses) < 2:
        print("FAIL: Need at least 2 epochs in epoch_metrics.csv to check improvement.", file=sys.stderr)
        sys.exit(1)

    val_loss_improved = val_losses[-1] < val_losses[0]
    print(f"  val_loss: first={val_losses[0]:.4f} last={val_losses[-1]:.4f} -> {'IMPROVED' if val_loss_improved else 'NOT IMPROVED'}")

    # 2. CER from samples_epoch_XX.json (epoch 0 to N-1)
    cers = []
    for epoch in range(len(val_losses)):
        p = experiment_dir / f"samples_epoch_{epoch:02d}.json"
        cer = get_mean_cer_from_samples_json(p)
        if cer is not None:
            cers.append((epoch, cer))

    if len(cers) < 2:
        print("FAIL: Need CER for at least 2 epochs (samples_epoch_00.json, ...).", file=sys.stderr)
        sys.exit(1)

    # Use first and last epoch for which we have CER
    first_cer = cers[0][1]
    last_cer = cers[-1][1]
    cer_improved = last_cer < first_cer
    print(f"  CER:      first={first_cer:.4f} last={last_cer:.4f} -> {'IMPROVED' if cer_improved else 'NOT IMPROVED'}")

    if val_loss_improved and cer_improved:
        print("PASS: Both validation loss and CER improved over epochs.")
        sys.exit(0)
    else:
        print("FAIL: At least one of validation loss or CER did not improve.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
