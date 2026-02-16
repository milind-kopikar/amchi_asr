# MMS Fine-tuning — Repro & Recovery Guide (RunPod / Local)

Purpose: A compact, step-by-step guide to reproduce the current MMS-based finetune (facebook/mms-1b-all), recover from failures (disk full / missing audio), and switch between MMS and AI4Bharat/NeMo environments.

## Environment & Versions (current)
- **Python:** 3.11.10
- **Torch:** 2.4.1+cu124
- **Transformers:** 4.57.3
- **Datasets:** 4.4.2
- **Evaluate:** 0.4.6
- **librosa:** 0.11.0
- **soundfile:** 0.13.1
- **huggingface_hub:** 0.36.0
- **accelerate:** 1.12.0
- **ffmpeg:** 4.4.2 (system binary)

> Tip: Save these versions in `requirements_mms.txt` (in the repo) for fast environment recreation.

## File layout (important files)
- `config.yaml` — main experiment config (smoke_test, training hyperparams, data manifest paths)
- `data/` — where downloads and manifests live: `data/train/audio/`, `data/dev/audio/`, `data/test/audio/`, `data/*/manifest.jsonl`
- `data/vocab.json` — Devanagari vocab used for tokenizer
- `data/processor_devanagari/` — saved Wav2Vec2Processor (tokenizer + feature_extractor)
- `results/mms_devanagari_finetune/` — training outputs, checkpoints & logs
  - `train.log` — training log (stdout/stderr captured)
  - `checkpoint-*` — HF checkpoints (large; rotate/delete when disk low)
  - `final_model/` — final saved model + processor
- `scripts/` — utilities:
  - `download_data_from_railway.py` — download & split recordings (use `--train_dev_ratio 0.9` to get 90/10 train/dev)
  - `train.py` — robust training script (preflight checks, `--preflight-only`, `--auto-download-missing`)
  - `build_vocab.py` / `create_processor.py` — build vocab + processor from manifests + extra corpus
  - `generate_smoke_report.py` / `evaluate_model.py` — evaluation & report generation

## Preflight checks (built into `scripts/train.py`)
- Run: `python scripts/train.py --preflight-only`
- What it does:
  - Loads `processor` from `config.yaml` path
  - Loads manifests (with `smoke_test` override if enabled)
  - Resolves Windows-style paths and searches `data/{train,dev,test}/audio` and `data_smoke/*` for missing files
  - Optionally attempts auto-download of missing audio from Railway if `--auto-download-missing` and `RAILWAY_URL` env var are set
  - Fails early with explicit counts & examples of missing files if any remain

## Reproduce full production run (MMS)
1. Download data (90/10 train/dev; 15% test default):
   ```bash
   python scripts/download_data_from_railway.py --output_dir data --seed 42 --train_dev_ratio 0.9
   ```
2. Build vocab & processor (if you changed manifests or added `extra_corpus.txt`):
   ```bash
   python scripts/build_vocab.py --manifests data/train/manifest.jsonl data/dev/manifest.jsonl
   python scripts/create_processor.py --vocab data/vocab.json --output_dir data/processor_devanagari
   ```
3. Run preflight:
   ```bash
   python scripts/train.py --preflight-only
   ```
4. Start production training (run in background to allow overnight runs):
   ```bash
   nohup python scripts/train.py > results/mms_devanagari_finetune/train.log 2>&1 &
   echo $!  # prints PID
   tail -f results/mms_devanagari_finetune/train.log  # monitor
   ```

## Recovery (disk full / partial checkpoints)
- If training crashes while saving (e.g., "No space left on device"):
  - Check disk: `df -h /`
  - Remove or archive the largest checkpoints: `rm -rf results/mms_devanagari_finetune/checkpoint-<N>`
  - Restart training (it will resume from the latest valid checkpoint):
    ```bash
    nohup python scripts/train.py > results/mms_devanagari_finetune/train.log 2>&1 &
    ```
  - Consider reducing `save_steps` or `save_total_limit` in `config.yaml` to avoid future fill-ups.

## Resuming / Safe restart notes
- Trainer resumes from the latest valid checkpoint automatically if it exists in the output dir.
- If a checkpoint was partially written and failed due to I/O error, delete that directory and resume from the previous checkpoint.

## Switching to NeMo (AI4Bharat)
- Use a distinct virtual environment and separate `requirements_nemo.txt` file because NeMo pins older packages.
- Example:
  ```bash
  python -m venv .venv_nemo && source .venv_nemo/bin/activate
  pip install nemo_toolkit[asr]==1.19.0 hydra-core==1.1.0 omegaconf==2.2.3
  ```
- Point NeMo configs to `data/{train,dev,test}/manifest.jsonl` and follow NeMo-specific docs in this README.

## Notes & Best practices
- Run `python scripts/train.py --preflight-only` before a long job.
- Ensure RunPod or host has ample disk (≥ 40GB recommended for model downloads and checkpoints).
- Persist `results/` externally if the instance is ephemeral.

---

