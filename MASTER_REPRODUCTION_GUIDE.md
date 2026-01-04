# Amchi ASR: Master Reproduction Guide

**Version:** 1.1  
**Date:** January 3, 2026  
**Author:** GitHub Copilot (Agent)

This guide serves as the **single source of truth** for setting up, verifying, and running the Amchi ASR fine-tuning pipeline on a fresh environment (e.g., RunPod). It consolidates all knowledge gained during the setup and pilot phases.

---

## 1. System Requirements

### Hardware
- **GPU:** NVIDIA GPU with at least 24GB VRAM (A10G, A40, A100 recommended).
- **Storage:** Persistent volume recommended (e.g., /workspace on RunPod).
- **RAM:** 32GB+ system RAM.

### Software Environment
- **OS:** Linux (Ubuntu 20.04/22.04 recommended).
- **Python:** 3.10 or 3.11 (Tested on 3.11).
- **CUDA:** 11.8 or 12.x (Compatible with PyTorch version).

---

## 2. Initial Setup (Fresh Instance)

### Step 1: Clone Repository
\`\`\`bash
cd /workspace
git clone https://github.com/milind-kopikar/amchi_asr.git
cd amchi_asr
\`\`\`

### Step 2: Run Environment Setup
This script installs system dependencies (ffmpeg, build-essential), reinstalls PyTorch for CUDA 11.8, installs the AI4Bharat NeMo fork, and applies critical runtime patches.

\`\`\`bash
bash setup_env.sh
\`\`\`

### Step 3: Setup Marathi/Konkani Model & Tokenizer
This script downloads the AI4Bharat model and extracts the **correct** language-specific tokenizer from the multilingual .nemo archive.

**For Marathi:**
\`\`\`bash
# Ensure you have logged in to Hugging Face
huggingface-cli login --token YOUR_TOKEN
python scripts/download_model_from_hf.py --repo_id ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large --local_dir models/indicconformer_stt_mr_hybrid_ctc_rnnt_large
# The fine_tune.py script will automatically extract the tokenizer on first run
\`\`\`

**For Konkani:**
\`\`\`bash
bash scripts/setup_konkani.sh
\`\`\`

---

## 3. Critical Technical Lessons & Architecture 🧠

### A. The Tokenizer Trap 🕵️‍♂️
The default tokenizer.model files provided on HuggingFace for AI4Bharat models are often generic or missing language-specific characters (like the Marathi/Konkani 'ळ').
- **Solution:** Always extract the tokenizer directly from the .nemo archive.
- **Method:** Inspect model_config.yaml inside the .nemo to find the hash of the language-specific tokenizer (e.g., def9dd6f... for Konkani) and extract it using tar.

### B. CTC-Only Fine-Tuning Strategy ⚖️
We use a **CTC-only** approach on top of the Hybrid (RNNT/CTC) IndicConformer model.
- **Rationale:** Avoids RNNT-specific joint/decoder validation and GPU JIT complexity (Numba issues) during training.
- **Implementation:** We instantiate the model as a Hybrid model but only train the CTC head. The scripts/fine_tune.py script handles the necessary config patches (e.g., setting aux_ctc.decoder.num_classes to match the local tokenizer).

### C. Data Loading & Filtering 📊
NeMo's default max_duration is often set to 16.7s or 20s. If your audio is longer, it will be silently filtered out.
- **Fix:** We set max_duration: 30.0 in our configs to ensure all samples are processed.

### D. Robust Inference 🧪
Loading fine-tuned checkpoints for inference requires specific handling because the checkpoint config may contain absolute paths or training-specific loss names that crash standard NeMo loaders.
- **Solution:** Use scripts/smoke_test_inference.py which patches the config in memory before instantiation.

### E. Research Logging & CER 📈
We have implemented a custom `SampleLoggerCallback` that provides per-epoch insights.
- **JSON Samples:** Saves `samples_epoch_XX.json` containing reference, hypothesis, WER, and **CER** (Character Error Rate).
- **CER Metric:** Crucial for Devanagari scripts where word-level errors (WER) can be misleadingly high due to small spelling variations.
- **Normalization:** The script now automatically strips Devanagari punctuation and normalizes whitespace before calculating WER/CER to ensure fair evaluation.
- **Visibility:** By default, it logs up to 40 samples to cover the entire validation set.

---

## 4. Marathi Pilot Results (January 4, 2026) 🏆

We conducted a 20-epoch pilot using the Marathi `indicconformer` model on the Story 4/5 dataset.

### Protocol
- **Training Set:** Story 1, 2, 3.
- **Dev Set:** Story 4 (used for validation during training).
- **Test Set:** Story 5 (used for final evaluation).
- **Note:** This protocol ensures consistency with previous Konkani experiments.

### Performance
- **Final Test WER:** 0.351
- **Final Test CER:** 0.142
- **Epochs:** 20
- **Learning Rate:** 0.0001
- **Optimizer:** AdamW with Noam Annealing.

### Artifacts
Results are stored in `nemo_experiments/marathi_pilot_v3/`, including:
- `final_test_results.json`: Summary of final metrics.
- `samples_epoch_19.json`: Final epoch transcriptions.
- `epoch_metrics.csv`: Training curves.

**Note:** A full snapshot of the data configuration used for this run is available in [DATA_SNAPSHOT_AMCHI_KONKANI.md](DATA_SNAPSHOT_AMCHI_KONKANI.md).

---

## 5. Deaf Speech Training Guide 👂

Training for deaf speech requires different hyperparameters due to the high acoustic variance and typically smaller datasets.

### Recommended Settings
- **Epochs:** 50+ (Small datasets need more passes).
- **Batch Size:** 4 (Provides more gradient updates per epoch).
- **Learning Rate:** 1e-4 (Stable for adaptation).
- **Normalization:** Ensure punctuation stripping is enabled in `fine_tune.py`.

### Data Protocol
- **Multi-User:** Always prefer combining data from multiple deaf speakers to help the model learn a generalized "deaf speech" acoustic profile.
- **Augmentation:** Speed and pitch perturbation are highly recommended for future runs.

---

## 6. Configuration Guide (YAML)

### Key Parameters
*   **Model:** `models/indicconformer_stt_mr_hybrid_ctc_rnnt_large/indicconformer_stt_mr_hybrid_rnnt_large.nemo`
*   **Tokenizer:** `tokenizers/marathi_tokenizer.model`
*   **Trainer:**
    *   max_epochs: 20
    *   accumulate_grad_batches: 1
*   **Exp Manager:**
    *   save_top_k: 3 (Keeps best 3 models based on val_wer)
    *   create_tensorboard_logger: true

---

## 5. Verification & Training Commands

### Preflight Check
Before a long run, verify the stack with a 1-epoch test:
\`\`\`bash
python scripts/fine_tune.py --config configs/marathi_pilot_1epoch_test.yaml --output_dir nemo_experiments/marathi_pilot_1epoch_test
\`\`\`

### Full Pilot Training (20 Epochs)
\`\`\`bash
export APPLY_CONV_PATCH=1
python scripts/fine_tune.py --config configs/marathi_pilot_20epoch.yaml --output_dir nemo_experiments/marathi_pilot_20epoch
\`\`\`

---

## 6. Troubleshooting

*   **Hypothesis objects in logs:** If your samples.json contains raw Hypothesis strings, ensure you are using the latest scripts/fine_tune.py which includes the normalization patch.
*   **Loss is NaN:** Check your learning rate. We use 0.0001 for stability.
*   **GPU Visibility:** Ensure CUDA_VISIBLE_DEVICES is set (usually 0).

---

**End of Master Guide**
