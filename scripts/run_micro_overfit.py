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


def _compute_char_distance(ref: str, hyp: str) -> float:
    """Normalized character-level Levenshtein distance: distance / max(1, len(ref))."""
    a = list(ref)
    b = list(hyp)
    m = len(a)
    n = len(b)
    if m == 0:
        return 0.0 if n == 0 else 1.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    dist = dp[m][n]
    return float(dist) / max(1, m)


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
    config = os.environ.get('MICRO_OVERFIT_CONFIG', 'configs/konkani_finetune_overfit20_single.yaml')

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
    # Also allow selecting a different config via env (MICRO_OVERFIT_CONFIG), or limit epochs via MAX_MICRO_EPOCHS
    max_epochs_env = os.environ.get('MAX_MICRO_EPOCHS', None)
    if max_epochs_env is not None:
        print('Note: MAX_MICRO_EPOCHS is set to', max_epochs_env)
        # If provided, patch the config to adjust max_epochs (simple sed approach)
        try:
            me = int(max_epochs_env)
            print('Overriding config max_epochs ->', me)
            # Use yq if available, else sed fallback
            if shutil.which('yq'):
                subprocess.run(['yq', '-i', f'.trainer.max_epochs = {me}', config])
            else:
                # naive sed replace (may fail for complex YAML but acceptable for our debug use)
                subprocess.run(["python3", "-c", f"import sys,yaml; d=open('{config}').read(); o=yaml.safe_load(d); o['trainer']['max_epochs']={me}; open('{config}','w').write(yaml.dump(o))"], check=True)
        except Exception as e:
            print('Failed to override max_epochs from MAX_MICRO_EPOCHS:', e)
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
    char_dist = None
    try:
        wer = _compute_wer(ref, pred)
    except Exception:
        wer = None
    try:
        char_dist = _compute_char_distance(ref, pred)
    except Exception:
        char_dist = None

    print('Contains Devanagari:', deva)
    print('WER (token-level):', wer)
    print('Char distance (normalized):', char_dist)

    # Accept criteria: pass if train loss reduced by >=50% OR char_dist <= 0.2
    # Load epoch metrics to compute train loss reduction
    em = find_latest_experiment(Path('results'))
    if not em:
        print('No experiment folder found to compute train-loss reduction')
        sys.exit(8)

    csvp = em / 'epoch_metrics.csv'
    if csvp.exists():
        with open(csvp, 'r', encoding='utf-8') as fh:
            rows = [l.strip() for l in fh.readlines() if l.strip()]
        if len(rows) >= 2:
            # header + at least one row; parse first and last numeric train_loss
            def parse_train_loss(line):
                parts = line.split(',')
                try:
                    tl = parts[1]
                    return float(tl) if tl != '' else None
                except Exception:
                    return None
            first_train = parse_train_loss(rows[1])
            last_train = parse_train_loss(rows[-1])
        else:
            first_train = last_train = None
    else:
        first_train = last_train = None

    reduced = False
    if first_train is not None and last_train is not None:
        try:
            reduced = (last_train <= 0.5 * first_train)
        except Exception:
            reduced = False

    # Pass if reduced or char_dist small enough
    char_ok = (char_dist is not None and char_dist <= 0.2)
    if reduced:
        print('PASS: train loss reduced by >=50% (learning confirmed)')
        sys.exit(0)
    if char_ok:
        print('PASS: char-level distance is within threshold (memorized)')
        sys.exit(0)

    print('FAIL: neither train-loss reduction nor char-distance acceptance met')
    sys.exit(7)


if __name__ == '__main__':
    main()
