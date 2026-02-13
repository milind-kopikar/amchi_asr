# Canonical Environment for Amchi-ASR ✅

This folder documents the *single canonical environment* we use for preflight tests and actual training runs (including the 1-epoch test and the micro-overfit check).

Principles:
- One canonical environment definition used for all tests and training runs (preflight, micro-overfit, and full training).
- Reproducible setup script to provision the environment (`scripts/ensure_env.sh`).
- A validator script to snapshot the exact installed versions at runtime (`scripts/dump_env_info.py`) and save a copy under `results/env_specs/` for traceability.

Core requirements (summary):
- OS: Linux
- Python: 3.11.x (use `venv_py311` in repo)
- System packages: `ffmpeg`, `build-essential`
- Key Python packages (examples):
  - `torch` / `torchvision` / `torchaudio` compatible with the machine CUDA (we recommend `torch>=2.4` for GPU; setup chooses wheel via `setup_env.sh`)
  - `nemo_toolkit` — use **upstream** (`nemo_toolkit[all]`) with Python 3.11. The AI4Bharat fork requires Python 3.9 and is not used for normal setup.
  - `sentencepiece` (critical: tokenizer)
  - `librosa`, `pynini`, `pandas`, `jiwer`, etc. (see root `requirements.txt`)

Important environment variables:
- `APPLY_CONV_PATCH=1` (ensures runtime conv_asr patch is used)
- `CUDA_VISIBLE_DEVICES` set to a non-empty value for GPU training (setup script sets to `0` if empty)

Files added here:
- `scripts/ensure_env.sh`: idempotent script to create `venv_py311`, install system deps, install packages, and patch NeMo.
- `scripts/dump_env_info.py`: records installed versions and important env vars into `results/env_specs/env_info-<timestamp>.json`.

Usage:
- Provision environment (idempotent):
  ./scripts/ensure_env.sh

- After provisioning, verify exact versions and record them for tracing a training run:
  python3 scripts/dump_env_info.py

- The micro-overfit runner (`scripts/run_micro_overfit.py`) will run `scripts/preflight_checks.py` before training and we will use the canonical environment created by `scripts/ensure_env.sh` to run the micro-overfit test.

If you'd like, I can now provision the canonical environment here (requires installing packages). Confirm and I will proceed to run `scripts/ensure_env.sh`, snapshot the environment with `dump_env_info.py`, then run the micro-overfit check under the canonical environment.

Automatic base model download (optional):

- Set `AUTO_DOWNLOAD_MODEL=1` to have `scripts/ensure_env.sh` attempt to download the base `.nemo` model (AI4Bharat IndicConformer) and extract tokenizer files into `models/tokenizer/` during provisioning. `scripts/preflight_checks.py` will also try to fetch the model when `AUTO_DOWNLOAD_MODEL=1` if the model path in your config is missing.

Quick unit-test for micro-overfit acceptance (fast mode):

- You can run a fast, unit-level acceptance check without running full training by setting env vars to skip preflight and training and pointing to prepopulated results (used by our tests):

  SKIP_MICRO_PREFLIGHT=1 SKIP_MICRO_TRAIN=1 RUN_MICRO_OVERFIT=1 pytest tests/test_micro_overfit_acceptance.py -q

This runs the acceptance logic (train-loss reduction OR char-distance threshold) against synthetic experiment files under `results/experiments/` and is useful for CI-level checks or quick verification.