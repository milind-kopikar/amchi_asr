# Smoke Test & Runbook — MMS ASR (facebook/mms-1b-all) 🔧

## Purpose ✅
This document captures what we learned running smoke tests (train + inference) and provides a reproducible runbook to validate an end‑to‑end fine‑tune and inference pipeline using Hugging Face models — specifically Meta's MMS (`facebook/mms-1b-all`).

## Key learnings ⚠️
- Disk space: HF models download into `HF_HOME` (`~/.cache/huggingface` by default). Ensure at least 30–50GB of free disk when working with large models. If disk is small, set `HF_HOME` to a mounted larger disk.
- Memory: Large models can cause OOM on CPU; use GPU + 32+ GB host RAM for reliable fine‑tuning. For CPU-only smoke runs, use smaller models (e.g., `facebook/wav2vec2-base-960h`).
- Use transformers' Auto classes (`AutoProcessor`, `AutoModelForCTC`) for model compatibility and avoid framework-specific forks.
- Avoid NeMo for this pipeline (project decision): MMS runs on standard Transformers and avoids NeMo version problems.

## Smoke test goals (what to verify) ✅
1. Dataset access: verify `data_smoke/` or live Railway URLs are reachable. 2. Model download and load. 3. Short training smoke (max_steps=5) on 3/1/1 split. 4. Inference smoke: transcribe a test audio and compare / log results. 5. Repeat inference across languages by switching language adapter codes.

## Minimal environment (recommended) 🧰
- Ubuntu 22.04
- Python 3.8+ (3.10 recommended)
- venv or conda
- For full fine-tuning: GPU (NVIDIA), CUDA matching PyTorch wheel, and 50+ GB disk, 32–64 GB RAM

## HF cache and disk tips 💾
- Set HF cache to large disk:
  export HF_HOME="/path/to/large/disk/huggingface"
- If disk is full, free `/root/.cache/huggingface` or delete old models before continuing.

## Quick manual commands (one-off) ⏱️
```bash
# create venv
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# GPU or CPU selection (choose appropriate torch wheel)
# CPU quick test
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
pip install transformers datasets librosa soundfile scipy evaluate jiwer

# Download smoke data
python scripts/download_smoke_test_data.py --output_dir data_smoke
# copy to expected location
mkdir -p data/test_run
cp data_smoke/train/manifest.jsonl data/test_run/train.tsv
cp data_smoke/dev/manifest.jsonl data/test_run/dev.tsv
cp data_smoke/test/manifest.jsonl data/test_run/test.tsv

# Run training smoke
python scripts/fine_tune_hf.py

# Run inference smoke (example)
python scripts/smoke_infer.py --model facebook/wav2vec2-base-960h --manifest data_smoke/test/manifest.jsonl
```

## MMS-specific notes (facebook/mms-1b-all) 📚
- Supports many languages including Konkani (kok). Use this model if you want a single base for multilingual experiments.
- To switch language adapter, set language code in config (kok, mar, hin, san). Some adapter activation may be model-specific; the smoke scripts include a `--lang` flag that will attempt to pass the language to the inference pipeline when supported.

## RunPod recommendations 🧪
- For smoke tests: CPU instance with 50 GB disk and 32 GB RAM is fine for testing download/load and tiny training with smaller models.
- For full fine-tuning (MMS-1B): use GPU instance (e.g., RTX 4090 or A100) with 40+ GB GPU RAM if possible, host with 50+ GB storage and ≥64GB host RAM for reliability.

---

For detailed automated smoke steps, see `scripts/smoke_run.sh` and `scripts/smoke_infer.py` included in the repo.

## RunPod Quick Start (exact commands) 🚀
Follow these exact steps on a fresh RunPod instance (50GB disk recommended, GPU recommended for MMS-1B fine-tuning):

1) Clone the branch containing the smoke helpers:
```bash
# clone only the smoke branch
git clone --branch smoke-mms --single-branch https://github.com/milind-kopikar/amchi_asr.git
cd amchi_asr
```

2) Prepare HF cache and credentials (use large disk mount):
```bash
export HF_HOME="/path/to/large/disk/huggingface"
export HF_TOKEN="<your_hf_token_here>"
mkdir -p "$HF_HOME"
```

3) Run the automated smoke runner (idempotent):
```bash
# CPU smoke
./scripts/smoke_run.sh --hf-home "$HF_HOME"
# or for GPU (ensure correct torch CUDA wheel installed)
./scripts/smoke_run.sh --hf-home "$HF_HOME" --gpu
```

4) Inspect results and logs:
- Training log: `results/smoke/train.log`
- Inference printed to stdout by `scripts/smoke_infer.py` (also visible in run output)

5) Optional: run parts manually
```bash
# create venv and activate
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip
# install minimal packages (choose GPU/CPU appropriately)
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
python -m pip install transformers datasets librosa soundfile scipy evaluate jiwer
# download smoke data
python scripts/download_smoke_test_data.py --output_dir data_smoke
# copy manifests
mkdir -p data/test_run
cp data_smoke/train/manifest.jsonl data/test_run/train.tsv
cp data_smoke/dev/manifest.jsonl data/test_run/dev.tsv
cp data_smoke/test/manifest.jsonl data/test_run/test.tsv
# run fine-tune smoke
python scripts/fine_tune_hf.py
# run inference smoke
python scripts/smoke_infer.py --model facebook/mms-1b-all --manifest data_smoke/test/manifest.jsonl --limit 10
```

## Troubleshooting ⚠️
- Disk full during model download: run `df -h` to check free space. If small, set `HF_HOME` to a larger mount or delete old cache (`rm -rf $HF_HOME/*`).
- OOM / process killed: add swap or use a GPU instance. Example add swap:
```bash
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```
- Hugging Face auth errors: ensure `HF_TOKEN` is set or run `huggingface-cli login`.
- Need to change base model or language: edit `configs/mms_smoke.yaml` or set `--model` on `scripts/smoke_infer.py`.

---
**Tip:** After verification, create a PR or merge `smoke-mms` into `master` using GitHub PR page: https://github.com/milind-kopikar/amchi_asr/pull/new/smoke-mms

If you want, I can also add a short single-command helper to launch the smoke run in a screen/tmux session on the RunPod. Let me know.
