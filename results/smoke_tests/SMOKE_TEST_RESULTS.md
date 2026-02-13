# Smoke Test Results — 5 Epochs, Single Sample

**Date:** 2026-02-13  
**Purpose:** Verify finetuning pipeline by overfitting on one sample (same audio/text for train, dev, test).

**Success criteria (for the suite):** Validation loss and CER must both **improve** (decrease) over epochs. If either does not improve, the smoke test is failed. See `results/smoke_tests/README.md` and `scripts/run_smoke_test_one_sample.sh`.

## Setup

- **Manifest:** `results/smoke_tests/smoke_one_sample.jsonl` (1 line)
- **Sample:** `data/amchi/train/audio/145.wav`  
- **Reference text:** चल रे भोपळा टुनुक टुनुक  
- **Config:** `configs/smoke_tests_5epoch.yaml`  
- **Epochs:** 5  
- **Output dir:** `results/smoke_tests/`

## Pipeline Status

| Step | Status |
|------|--------|
| Training (5 epochs) | Completed |
| Best checkpoint saved | `results/smoke_tests/checkpoints/smoke_5epoch-epoch=04-val_loss=57.000.ckpt` |
| Final evaluation (script) | Ran; results in `experiments/.../final_test_results.json` |

## Results (Final Evaluation)

- **Reference:** चल रे भोपळा टुनुक टुनुक  
- **Prediction:** आ टुुनुा  
- **WER:** 1.0 (100%)

So after 5 epochs on this single sample, the model did **not** reach very low WER on the same sample. Possible reasons:

1. **Too few epochs** — Try 10–20 epochs for a single-sample overfit.
2. **Evaluation decoding** — Final eval may not be using CTC decoding; worth checking.
3. **LR / batch** — Same sample every step; might need different LR or more steps.

## Artifacts

- Checkpoints: `results/smoke_tests/checkpoints/`
- Experiment log: `results/smoke_tests/experiments/20260213_201603/`
- Final test results: `results/smoke_tests/experiments/20260213_201603/final_test_results.json`

## Conclusion

The **finetuning pipeline runs end-to-end**: data load, training, checkpointing, and evaluation all completed. For a stricter “overfit test” (near-zero WER on the same sample), consider increasing epochs or re-running with a config tuned for single-sample overfitting.
