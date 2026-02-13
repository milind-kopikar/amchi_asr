# One-Sample Smoke Test

This folder holds the **one-sample smoke test**: the same single audio and transcript are used for train, dev, and test. The goal is to verify that the finetuning pipeline runs and that the model can improve on that sample over a few epochs.

## What It Does

- **Manifest:** `smoke_one_sample.jsonl` (one line; same sample for train, validation, and test).
- **Config:** `configs/smoke_tests_5epoch.yaml` (5 epochs, output here).
- **Output:** All artifacts (checkpoints, logs, metrics) go under `results/smoke_tests/` so they stay separate from real training runs.

## Success Criteria

The smoke test **passes** only if **both** of the following improve (decrease) over epochs:

1. **Validation loss** — last epoch < first epoch (from `epoch_metrics.csv`).
2. **CER (Character Error Rate)** — last epoch < first epoch (from `samples_epoch_XX.json`).

If either validation loss or CER does not improve, the smoke test is **failed** (exit code 1).

## How to Run

From the repo root:

```bash
export APPLY_CONV_PATCH=1
./scripts/run_smoke_test_one_sample.sh
```

Or as part of the full preflight suite:

```bash
./scripts/run_all_preflight.sh
```

The one-sample smoke test is step 6 in that suite.

## Validation

After training, validation is done by:

```bash
python scripts/validate_smoke_one_sample.py [--experiment_dir results/smoke_tests/experiments/TIMESTAMP]
```

If `--experiment_dir` is omitted, the script uses the latest experiment under `results/smoke_tests/experiments/`.

## Artifacts

- `smoke_one_sample.jsonl` — single-sample manifest (created from `data/amchi/train/manifest.jsonl` if missing).
- `checkpoints/` — best checkpoint from the 5-epoch run.
- `experiments/<timestamp>/` — `epoch_metrics.csv`, `samples_epoch_00.json` … `samples_epoch_04.json`, `final_test_results.json`.

## See Also

- `scripts/README_SMOKE.md` — overview of smoke tests.
- `configs/smoke_tests_5epoch.yaml` — config used for this test.
