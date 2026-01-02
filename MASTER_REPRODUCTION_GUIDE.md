# Amchi ASR: Master Reproduction Guide

**Version:** 1.0  
**Date:** January 2, 2026  
**Author:** GitHub Copilot (Agent)

This guide serves as the **single source of truth** for setting up, verifying, and running the Amchi ASR fine-tuning pipeline on a fresh environment (e.g., RunPod). It consolidates all knowledge gained during the initial setup phase.

**Recent updates (2026-01-02):**
- Fixed WER calculation in `scripts/fine_tune.py` so validation WER is computed correctly during smoke tests.
- Resolved Trainer startup crash by preferring `pytorch_lightning` import and removing `LearningRateMonitor` where necessary; extended 5-epoch smoke test now completes successfully and produces checkpoints in `nemo_experiments/checkpoints`.
- Added robust inference script `scripts/smoke_test_inference.py` that loads checkpoints, patches loss_name, switches to CTC decoding, and performs single-sample transcription for verification.

---

## 1. System Requirements

### Hardware
- **GPU:** NVIDIA GPU with at least 24GB VRAM (A10G, A40, A100 recommended).
- **Storage:** Persistent volume recommended (e.g., `/workspace` on RunPod).
- **RAM:** 32GB+ system RAM.

### Software Environment
- **OS:** Linux (Ubuntu 20.04/22.04 recommended).
- **Python:** 3.10 or 3.11 (Tested on 3.11).
- **CUDA:** 11.8 or 12.x (Compatible with PyTorch version).

---

## 2. Initial Setup (Fresh Instance)

### Step 1: Clone Repositories
You need the main project repo and the specific NeMo fork/branch used by AI4Bharat.

```bash
cd /workspace

# 1. Clone Amchi ASR (Your Project)
git clone https://github.com/milind-kopikar/amchi_asr.git
cd amchi_asr

# 2. Clone NeMo (AI4Bharat Fork) - CRITICAL
# We use the 'nemo-v2' branch which contains necessary IndicConformer implementations.
git clone -b nemo-v2 https://github.com/AI4Bharat/NeMo.git NeMo_ai4bharat
```

### Step 2: System Dependencies
Install required system libraries for audio processing.

```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
```

### Step 3: Python Environment
Install PyTorch, NeMo, and project dependencies.

```bash
# 1. Install PyTorch (adjust CUDA version if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. Install NeMo from the cloned AI4Bharat repo
cd /workspace/NeMo_ai4bharat
pip install -e .[asr]

# 3. Install Project Requirements
cd /workspace/amchi_asr
pip install -r requirements.txt
```

---

## 3. File Structure & Data Organization

The pipeline expects a specific directory structure. Ensure your `/workspace/amchi_asr` looks like this:

```
/workspace/amchi_asr/
├── configs/                  # YAML configuration files
│   ├── tmp_marathi_1epoch_ctc_golden.yaml  # 1-epoch smoke test
│   └── tmp_marathi_5epoch_ctc_smoke.yaml   # 5-epoch extended test
├── data/                     # Audio data and manifests
│   ├── dev/                  # Development/Validation audio
│   ├── train/                # Training audio
│   └── test/                 # Testing audio
├── models/                   # Pre-trained models
│   └── indicconformer_stt_mr_hybrid_ctc_rnnt_large/
│       └── indicconformer_stt_mr_hybrid_rnnt_large.nemo
├── nemo_experiments/         # Output directory for checkpoints/logs
├── scripts/                  # Python and Shell scripts
│   ├── fine_tune.py          # Main training script
│   ├── run_all_preflight.sh  # MASTER PREFLIGHT SCRIPT
│   ├── robust_smoke_test.sh  # 1-epoch test
│   └── extended_smoke_test.sh # 5-epoch test
├── tokenizers/               # Tokenizer models
│   └── marathi_tokenizer.model
└── tiny_one_sample.jsonl     # Tiny manifest for smoke tests
```

### Data Locations
- **Audio:** Place `.wav` files in `data/`. Must be **16kHz Mono**.
- **Manifests:** JSONL files (e.g., `train_manifest.json`) mapping audio paths to text.
- **Model:** Download the AI4Bharat Marathi model to `models/`.

---

## 4. Configuration Details

### Key Config Parameters (YAML)
We use a **CTC-based fine-tuning** approach on top of the Hybrid (RNNT/CTC) IndicConformer model.

*   **Model Architecture:** `IndicConformer` (Large)
*   **Tokenizer:** SentencePiece BPE (`tokenizers/marathi_tokenizer.model`)
*   **Loss Function:** CTC (`loss_name: "default"` patched from "ctc" during load)
*   **Checkpointing:**
    *   `save_top_k: 3` (Keeps best 3 models based on WER/Loss)
    *   `monitor: "val_loss"` (or `val_wer`)

### Critical "Gotchas" & Fixes
1.  **Loss Name Patch:** The pre-trained model config uses `loss_name: ctc`, but NeMo's runtime expects `default` or specific RNNT losses. Our scripts (`fine_tune.py`, `smoke_test_inference.py`) automatically patch this in memory.
2.  **Tokenizer Path:** The config expects `tokenizer.dir` and `tokenizer.model_path`. Ensure these point to absolute paths or valid relative paths within `tokenizers/`.
3.  **Hybrid vs CTC:** The model is Hybrid, but we fine-tune using the CTC decoder head for stability and speed on smaller datasets.

---

## 5. Preflight Checks (The "Green Light")

Before starting a long training run, **ALWAYS** run the master preflight script. This verifies the entire stack.

```bash
cd /workspace/amchi_asr
./scripts/run_all_preflight.sh
```

**What this script does:**
1.  **Environment Check:** Verifies Python libs (`check_libs.py`).
2.  **Data/Model Check:** Verifies files exist (`check_data.py`, `check_model.py`).
3.  **Audio Check:** Verifies sample rate/channels (`check_audio_properties.py`).
4.  **1-Epoch Smoke Test:** Runs a quick training loop to ensure no crashes.
5.  **5-Epoch Extended Test:** Runs a longer loop to verify **loss reduction** (learning) and **checkpoint saving**.

**If this script passes, your environment is 100% ready.**

---

## 6. Running Full Training

To run the full fine-tuning on your dataset:

1.  **Prepare Manifests:** Create `train_manifest.json`, `val_manifest.json`, `test_manifest.json`.
2.  **Create Config:** Copy `configs/tmp_marathi_5epoch_ctc_smoke.yaml` to `configs/my_full_run.yaml`.
    *   Update `manifest_filepath` for train/val/test.
    *   Update `max_epochs` (e.g., 50 or 100).
    *   Update `exp_manager.name` (e.g., "Amchi_Konkani_Run1").
3.  **Run Command:**

```bash
python scripts/fine_tune.py \
    --config "configs/my_full_run.yaml" \
    --output_dir "nemo_experiments"
```

---

## 7. Troubleshooting

*   **`pynvml` warnings:** Ignore these; they are harmless deprecation warnings from PyTorch/NeMo.
*   **`Hypothesis object is not JSON serializable`:** This is a known issue in some NeMo versions during final test evaluation. It does not affect the training or checkpoint saving. The smoke tests handle this.
*   **Loss is NaN:** If loss is NaN in the first epoch, try reducing the learning rate (`optim.lr`) or increasing `warmup_steps`.

---

**End of Guide**
