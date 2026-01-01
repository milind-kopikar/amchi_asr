#!/usr/bin/env python3
"""Run a 20-epoch micro-overfit on tiny manifests and verify the sample is memorized.

This is intentionally opt-in (not part of CI) because it runs training.
Usage:
  python3 scripts/run_micro_overfit.py

Returns non-zero if check fails.
"""
import os
import sys
import subprocess
import json
import time
from pathlib import Path

# small helper to compute simple token-level WER

def _compute_wer(ref: str, hyp: str) -> float:
    r = ref.split()
    h = hyp.split()
    m = len(r)
    n = len(h)
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


def find_latest_experiment(output_root: Path) -> Path:
    exp_root = output_root / 'experiments'
    if not exp_root.exists():
        return None
    candidates = [p for p in exp_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def has_devanagari(s: str) -> bool:
    return any(0x0900 <= ord(ch) <= 0x097F for ch in s)


def main():
    # Use a dedicated single-sample overfit config (train/val/test all point to same tiny manifest)
    config = 'configs/konkani_finetune_overfit20_single.yaml'

    # Run preflight checks first to avoid known tokenizer/model mismatches
    print('Running preflight checks before micro-overfit...')
    ret = subprocess.run([sys.executable, 'scripts/preflight_checks.py'], check=False)
    if ret.returncode != 0:
        print('Preflight checks failed; aborting micro-overfit')
        sys.exit(2)

    cmd = [
        'bash', '-lc',
        'rm -rf results/experiments/* results/checkpoints/* || true && APPLY_CONV_PATCH=1 python3 scripts/fine_tune.py --config %s' % config
    ]
    print('Running micro-overfit (this may take a few minutes)...')
    ret = subprocess.run(' '.join(cmd), shell=True)
    if ret.returncode != 0:
        print('micro-overfit training failed')
        sys.exit(3)

    exp = find_latest_experiment(Path('results'))
    if not exp:
        print('No experiment folder found after micro-overfit')
        sys.exit(3)

    final = exp / 'final_test_results.json'
    if not final.exists():
        print('final_test_results.json not found in', exp)
        sys.exit(4)

    with open(final, 'r', encoding='utf-8') as fh:
        obj = json.load(fh)

    per = obj.get('per_sample') or obj.get('per_sample', [])
    if not per:
        print('No per_sample entries found in final_test_results.json')
        sys.exit(5)

    sample = per[0]
    ref = sample.get('reference', '')
    pred = sample.get('prediction', '')
    print('Reference:', ref)
    print('Prediction:', pred)

    deva = has_devanagari(pred)
    wer = None
    try:
        wer = _compute_wer(ref, pred)
    except Exception:
        wer = None

    print('Contains Devanagari:', deva)
    print('WER (token-level):', wer)

    # Accept criteria: contains Devanagari and WER <= 0.2 (meaning near-exact)
    if not deva:
        print('FAIL: prediction does not contain Devanagari characters')
        sys.exit(6)
    if wer is None or wer > 0.2:
        print('WARN: prediction WER is high; not considered memorized (WER > 0.2)')
        sys.exit(7)

    print('PASS: micro-overfit produced accurate Devanagari prediction')
    sys.exit(0)


if __name__ == '__main__':
    main()
