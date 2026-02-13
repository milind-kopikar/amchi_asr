# Learnings from Test Runs (February 2026)

This document records what we learned from running the preflight suite, smoke tests, and the 20-epoch Marathi Amchi fine-tuning. Use it to avoid repeating mistakes and to keep the test suite aligned with reality.

---

## 1. What Works

- **Python 3.11 + upstream NeMo:** We use **Python 3.11** with **upstream** NVIDIA NeMo (`nemo_toolkit[all]`). Do **not** use the AI4Bharat NeMo fork for this setup—it targets Python 3.9 and is incompatible with 3.11.
- **CTC-only fine-tuning:** Using decoder_type CTC and training only the auxiliary CTC head avoids RNNT joint/decoder validation and GPU JIT (Numba) issues. See `REPRODUCTION_NOTES.md`.
- **20-epoch full run:** The pipeline runs end-to-end with `configs/marathi_amchi_20epoch.yaml`: 20 epochs, `save_top_k=3`, best checkpoint by `val_wer`, with CER and WER logged per epoch.
- **RunPod / A40:** Training was verified on RunPod with an A40 GPU; `CUDA_VISIBLE_DEVICES=0` was used. Ensure the instance has a GPU and that CUDA is visible before starting.
- **CER in results:** We log CER (Character Error Rate) in addition to WER in `samples_epoch_XX.json` and in the test results. CER is important for Devanagari where small spelling differences inflate WER.

---

## 2. Smoke Test

- **Purpose:** Quickly verify that the full pipeline runs and that the model can improve on a single sample (train/dev/test all use the same sample). This catches environment and config errors before long runs.
- **Location:** One-sample smoke test output lives under `results/smoke_tests/`. Config: `configs/smoke_tests_5epoch.yaml`. Runner: `./scripts/run_smoke_test_one_sample.sh`. Validator: `scripts/validate_smoke_one_sample.py`.
- **Success criteria:** Both **validation loss** and **CER** must improve (decrease) over the 5 epochs. If either fails to improve, the smoke test fails (exit code 1).
- **Integration:** The one-sample smoke test is **step 6** of the full preflight suite: `./scripts/run_all_preflight.sh`. See `results/smoke_tests/README.md` and `scripts/README_SMOKE.md`.

---

## 3. GPU Check in the Test Suite

- **Lesson learned:** We discovered that training can appear to “run” on CPU if the GPU is not visible (e.g. `CUDA_VISIBLE_DEVICES` unset or misconfigured), leading to very slow runs or late failures. The test suite should **verify that a GPU is available and usable** before starting any training or heavy tests.
- **Action taken:** A **GPU check** was added to the preflight suite and runs **first** (before library, data, or smoke tests). If no CUDA-capable GPU is detected, the script exits with a clear error so we do not waste time on CPU-only runs.
- **How to run:** The check runs automatically as step 1 of `./scripts/run_all_preflight.sh`. It can also be run standalone: `python3 scripts/check_gpu.py`.

---

## 4. What We Do Not Push to Git

- **Data files:** The `data/` directory (audio and manifests) is large and is **not** committed. It remains in `.gitignore`.
- **Large checkpoints:** Result directories may contain large `.ckpt` files and `checkpoints/` folders. These are ignored via `.gitignore` so we only commit lightweight result artifacts (e.g. `epoch_metrics.csv`, `samples_epoch_*.json`, `final_test_results.json`, `run_metadata.json`, READMEs).

---

**See also:** `MASTER_REPRODUCTION_GUIDE.md`, `REPRODUCTION_NOTES.md`, `results/smoke_tests/README.md`, `scripts/README_SMOKE.md`.
