# Amchi-ASR — Golden Environment Setup & Notes ✅

## Overview 🎯
This document captures the exact setup, fixes, and patches required to get the AI4Bharat + NeMo ASR experimentation environment working (the *state that took ~6 hours to reproduce*). Use `setup_env.sh` to reapply the environment setup & the critical NeMo patch.

**Repro tip:** For the exact `.nemo` edit & restore recipe used to get the CTC smoke test working (offline aux_ctc edit, strict=False restore, tokenizer guidance, model storage policy), see `REPRODUCTION_NOTES.md` at the repo root. This file is the canonical short recipe you should follow when re-downloading or restoring models.

> Note: This process is specifically for the AI4Bharat + NeMo approach used in this repo.

---

## Quick usage 🔧
1. Make the script executable: `chmod +x setup_env.sh`
2. Run it as root (it installs system packages): `sudo ./setup_env.sh`
3. Run the Konkani setup script: `bash scripts/setup_konkani.sh`

The `setup_env.sh` script will:
- Install `ffmpeg` to fix pydub warnings
- Reinstall PyTorch wheels for CUDA (cu118) to enable GPU
- Install required Python packages (nemo_toolkit and key libs)
- Copy our vendored `patches/conv_asr_fixed.py` over the installed NeMo `conv_asr.py`

The `scripts/setup_konkani.sh` script will:
- Download the AI4Bharat Konkani model.
- Extract the **correct** Konkani-specific tokenizer from the multilingual `.nemo` archive. (This is critical as the default HuggingFace `tokenizer.model` often lacks Devanagari support).

Important environment variables set/used by the project:
- `APPLY_CONV_PATCH=1` (recommended): enables the runtime conv_asr patch that fixes language-id and mask handling.
- `LD_LIBRARY_PATH` should include your CUDA lib directory (for example `/usr/local/cuda/lib64`) if CUDA is installed in a non-standard location.

Data splitting policy (story-based):
- The `download_data_from_railway.py` script supports a `--use_story_split` flag that enforces deterministic, leakage-free splits.
- **Canonical split (do not swap):** **Story 4 = dev, Story 5 = test.** When redoing the setup or documenting the split, always use this convention.
- **Canonical Split (Marathi Pilot):**
  - **Train:** Stories 1, 2, 3 (472 samples)
  - **Validation (Dev):** Story 4 - "भोलागली रेलयात्रा" (37 samples). Used during finetuning.
  - **Test:** Story 5 - "रोहन होड ज़ाल्लो!" (37 samples). Held out for final evaluation only.
- This split ensures that the model is tested on entirely unseen stories and speakers.

## Bootstrap / First-Time Host Notes 🐣
- **Hugging Face login (CRITICAL):** The AI4Bharat models are gated or require authentication. On a new machine, you **must** run `huggingface-cli login` and provide your token before the training script can download the weights.
  ```bash
  huggingface-cli login --token YOUR_HF_TOKEN
  ```
- **Model Weights:** If the `.nemo` file is 0 bytes, it means the LFS download failed. Use `scripts/download_model_from_hf.py` after logging in to fetch the real weights.

- **Chicken & Egg (bootstrap command):** This repository and the setup script may not exist yet on a fresh host. Use this command to bootstrap a fresh machine in one step (replace YOUR_USER):

  ```bash
  git clone https://github.com/YOUR_USER/amchi_asr.git && cd amchi_asr && sudo ./setup_env.sh
  ```

- **Cython / build deps:** Some packages (e.g., `pynini`) require `Cython` and build tools to be installed before they can build. We install `Cython` early in the script (`pip install Cython`) and the script now installs `build-essential` via `apt-get` on fresh hosts so compilers and make are available for building wheels. If you prefer to pre-seed system build deps yourself, run `sudo apt-get install -y build-essential` before running `setup_env.sh`.

- **Hugging Face login reminder:** The `setup_env.sh` script will now print a final reminder to run `huggingface-cli login` (or set `HF_TOKEN`) after the setup finishes so you're ready to download model files.

- **Automatic base model download (optional):** If you set `AUTO_DOWNLOAD_MODEL=1` in your environment, `setup_env.sh` will attempt to download the base `.nemo` model (AI4Bharat IndicConformer) and extract tokenizer files into `models/tokenizer/`. Preflight checks (`scripts/preflight_checks.py`) also support `AUTO_DOWNLOAD_MODEL=1` and will try to fetch the model if it's missing during preflight.

- **Experiments / checkpointing / log location:** By default training configs point to a persistent experiments directory at `/workspace/amchi_asr/experiments`. The default checkpoint policy is to **save only the top 5** checkpoints (`save_top_k: 5`) to limit disk usage. You can change `exp_manager.exp_dir` in your config files if you prefer a different mount path.

---

## What we learned / environment details 📚
- **Python version:** Use **Python 3.11** (venv: `venv_py311`). This is the **standard and only recommended** environment for this project.
- **NeMo:** Use **upstream** NVIDIA NeMo (`nemo_toolkit[all]`), **not** the AI4Bharat NeMo fork. The AI4Bharat fork requires Python 3.9 (it pins `llvmlite==0.38.1`, which has no wheel for Python 3.11) and will fail on 3.11 with "Could not find a version that satisfies the requirement llvmlite==0.38.1". For RunPod or fresh setup: create `venv_py311`, install PyTorch, then `pip install "nemo_toolkit[all]" pynini librosa`. If using `setup_env.sh`, set `USE_UPSTREAM_NEMO=1` before running it so the script installs upstream NeMo instead of attempting the fork.
- If you ever need the AI4Bharat fork (e.g. for debugging fork-specific behavior), use Python 3.9 and see `AI4BHARAT_SETUP_GUIDE.md`. For normal training and inference in this repo, **always use Python 3.11 + upstream NeMo.**

## Preflight & Smoke Testing 🚀
Before running full experiments, it is critical to verify the environment and model configuration.

1.  **Configure the .nemo Model:**
    The AI4Bharat .nemo models may require specific configuration tweaks (e.g., fixing tokenizer paths, handling vocabulary mismatches).
    **CRITICAL:** Review `/workspace/amchi_asr/REPRODUCTION_NOTES.md` for the exact steps to identify the correct tokenizer and configure the model. For Marathi, this involves finding the tokenizer containing 'ळ'.

2.  **Run the Robust Smoke Test:**
    We have a dedicated script to verify the model, tokenizer, data availability, and GPU training pipeline.
    
    ```bash
    ./scripts/robust_smoke_test.sh
    ```
    
    This script will:
    - Verify the correct tokenizer is present.
    - Verify the base model exists.
    - Check that all audio files referenced in the manifest actually exist.
    - Run a 1-epoch fine-tuning job on the GPU.
    - Ensure the pipeline produces valid output.

    If this test passes, your environment is ready for full-scale training.

- Important packages & versions (examples):
  - torch / torchvision / torchaudio: installed from the official PyTorch CUDA 11.8 wheel (matching CUDA available in the runpod).
  - nemo_toolkit: `pip install "nemo_toolkit[all]"` (used for training & decoding features).
  - pynini, librosa: other libs used by preprocessing and tokenization.

## Unicode / Devanagari (देवनागरी) handling 🪔
- Ensure all transcript text in manifests and evaluation inputs is encoded in **UTF-8** and written in Devanagari script (for example: `तुमी कसो आसा`). This repository assumes transcripts use Devanagari and the SentencePiece tokenizer pieces are trained for Devanagari.
- Verify the local SentencePiece files in the unpacked model folder (e.g., `*_tokenizer.vocab` / `*_tokenizer.model`) are the ones referenced by your `model_config.yaml` and that `aux_ctc.decoder.vocabulary` has been replaced with the real tokenizer pieces (not the earlier dummy `token_0..token_255`).
- When reading or writing manifests/datasets in Python, always open files with `encoding='utf-8'` to preserve Unicode characters (e.g., `open(path, 'r', encoding='utf-8')`).
- If you see garbled characters (mojibake) in terminals or logs, set your locale to UTF-8 before running scripts:

  ```bash
  export LC_ALL=en_US.UTF-8
  export LANG=en_US.UTF-8
  ```

Tokenizer consistency (important):
- Training loads the tokenizer from `config.model.tokenizer.dir` on disk, whereas running inference with a `.nemo` via `restore_from()` will use the tokenizer embedded inside the `.nemo` archive.
- If these differ, transcripts may be encoded as `<unk>` during training which will teach the model to predict `?` placeholders. Always verify the local tokenizer matches the `.nemo` tokenizer before training.

Quick verification and fix (example):

```bash
# 1) Compare checksums/sizes
md5sum models/tokenizer/*tokenizer.model
# 2) Show model_config inside .nemo to find the expected tokenizer
tar -tf models/your_model.nemo | grep tokenizer
# 3) Extract the referenced tokenizer and copy it into place (backup original)
mkdir -p temp_investigation && tar -xf models/your_model.nemo -C temp_investigation <tokenizer.model>
cp models/tokenizer/<local>.model models/tokenizer/backup/<local>.model.bak
cp temp_investigation/<tokenizer.model> models/tokenizer/<local>.model
# 4) Run a quick validation script (we include debug_tokenizer.py) to ensure a Devanagari sample encodes correctly
python3 debug_tokenizer.py
```

Include this check in your pre-deployment checklist to avoid regressions.

---

## Key problems & fixes implemented 🔧
1. Symptom: Decoding/inference crashed with TypeError or IndexError while building language-class masks in `ConvASRDecoder.forward`. Also, missing decoder vocabulary caused zero-class outputs in some cases.
2. Root cause: `conv_asr.py` assumed language ids and language masks were simple Python ints/lists. When `language_ids` was a torch scalar or when `language_masks` entries were `torch.Tensor` or numpy arrays, the old code did `torch.tensor(list_of_tensors)` or similar and raised exceptions.
3. Fixes applied (on-disk edits in installed NeMo):
   - Added `_LanguageMaskList` wrapper class to allow tensor indices in `language_masks` lookups safely (handles scalar tensors and 1-D single-element tensors).
   - Wrapped `language_masks` passed to `ConvASRDecoder.__init__` with `_LanguageMaskList`.
   - Reworked mask normalization in `ConvASRDecoder.forward` to:
     - Normalize `language_ids` to the batch size (support scalar torch ids and lists).
     - Convert mask entries into a boolean mask of shape `[C]` (clip out-of-range indices, accept bool masks, index tensors, lists, numpy arrays).
     - Stack masks to `[B, C]` and expand to `[B, T, C]`, then use `torch.masked_select` and `view(B, T, -1)`.
   - Applied a small compatibility change to the SentencePiece tokenizer: `ids_to_text(self, ids, lang=None)` (accept optional `lang` arg used by ctc decoding).

Files changed and where to find them:
- The patch is applied to the **installed** NeMo `conv_asr.py` (e.g. in your venv `site-packages/nemo/collections/asr/modules/conv_asr.py`). `setup_env.sh` copies `patches/conv_asr_fixed.py` over it.
- Vendored snapshot (committed): `patches/conv_asr_fixed.py` (commit: d18f10e)

---

## Committed patches & how to use them 🗂️
- `patches/conv_asr_fixed.py`: A snapshot of the patched `conv_asr.py` that contains `_LanguageMaskList` & the safer `forward` implementation.
- The `setup_env.sh` script uses Python to locate the installed `conv_asr.py` and copies `patches/conv_asr_fixed.py` over it.

**GPU visibility note:** If `CUDA_VISIBLE_DEVICES` is defined but empty, GPU runtimes (PyTorch) will not see any devices. `setup_env.sh` will set `CUDA_VISIBLE_DEVICES=0` if it is empty and then validate PyTorch CUDA availability. If your environment uses multiple GPUs or a different device mapping, adjust `CUDA_VISIBLE_DEVICES` accordingly before running the script.


---

## Notes & caveats ⚠️
- The `setup_env.sh` script uses `apt-get` and installs PyTorch via pip wheel index; adjust the CUDA wheel selection if you need a different CUDA version.
- **Important (RunPod / fresh setup):** This project uses **Python 3.11** and **upstream** NVIDIA NeMo (`nemo_toolkit[all]`). **Do not** use the AI4Bharat NeMo fork for normal setup—it requires Python 3.9 and will fail on 3.11 (llvmlite dependency). Always set `USE_UPSTREAM_NEMO=1` before running `setup_env.sh`, or create a Python 3.11 venv and install with `pip install "nemo_toolkit[all]" pynini librosa`, then apply the conv_asr patch (see above). The script may try the AI4Bharat fork first if `USE_UPSTREAM_NEMO` is not set; on Python 3.11 that install will fail—use upstream instead.
- Running the script as non-root may fail the `apt-get` step.
- This process is specific to the AI4Bharat *models* (IndicConformer .nemo files) with **upstream** NeMo; the AI4Bharat *fork* of NeMo is only for Python 3.9 and is not used in this repo's standard workflow.

---

## Preflight checks (recommended)
To avoid regressions and wasted time during training, run the preflight checks before starting any training or inference work. These checks validate the runtime environment, tokenizer consistency, and key library availability.

Run the quick script:

```bash
python3 scripts/preflight_checks.py
```

What the preflight checks validate:
- Python version: **3.11** (recommended and tested). Do not use 3.9 for normal setup unless you explicitly use the AI4Bharat NeMo fork (see AI4BHARAT_SETUP_GUIDE.md).
- `ffmpeg` binary availability
- `torch` import and CUDA availability
- `nemo` import and whether the `conv_asr` runtime patch is applied (we look for our `_LanguageMaskList` fix)
- Local `tokenizer.model` encodes Devanagari for a canonical sample (prevents `<unk>` regressions)
- Disk free space (require >= 10 GB free by default)
- Presence of the `.nemo` file referenced by your training config

If any check fails, the script prints diagnostic info to guide remediation.

**Quick tokenizer sanity check (manual)**

```bash
python3 debug_tokenizer.py
# Expect: 'PASS' and a round-trip decode containing Devanagari characters
```

---

## Suggested next steps
- Add unit tests around `ConvASRDecoder.forward` verifying the handling of: scalar torch language ids, list of tensor masks, boolean masks, numpy arrays.
- Add the preflight checks to CI (run early in the pipeline) and add a small test that validates tokenizer encoding of a Devanagari sample.
- Optionally create an upstream PR to the NeMo fork for maintainability and to prevent regressions.

If anything here needs to be expanded (e.g., more detailed package versions, Dockerfile, or CI steps), tell me which area to prioritize and I will add it to this document and/or the repo.

---

## Full preflight & unit tests
To validate the environment and essential invariants before training, run the convenience script which runs the preflight checks and our unit tests (including the tokenizer <-> .nemo consistency test).

```bash
# From repo root, activated venv (recommended)
python3 -m venv venv_py311  # if not already created
source venv_py311/bin/activate
pip install -r requirements.txt  # or ensure python deps are present
./scripts/run_preflight_tests.sh
```

This exits non-zero on failure and prints diagnostic info to help repair issues quickly.

Optional micro-overfit check (opt-in) 🔬

- We provide a micro-overfit sanity test that runs training for 20 epochs on a single sample to verify the model can memorize a single sentence. This is intentionally opt-in because it runs training (but on a tiny dataset, so it completes reasonably fast).

  Run it like this:

  ```bash
  # runs preflight checks, unit tests, and the micro-overfit check (20 epochs)
  RUN_MICRO_OVERFIT=1 ./scripts/run_preflight_tests.sh
  ```

- The check is implemented in `scripts/run_micro_overfit.py` and will exit non-zero if the final test prediction does not contain Devanagari characters and a low WER (meaning it did not memorize the sample). The micro-overfit is intentionally opt-in for full training (`RUN_MICRO_OVERFIT=1`) but can be included in the preflight suite by setting `PREFLIGHT_RUN_MICRO=1`.

- For quick CI/fast validation we support a lightweight synthetic micro-overfit fallback (runs if real training fails) which checks that the PyTorch training loop and GPU are functional.

Quick CI / unit test (fast):

- To validate the acceptance logic without performing training (useful for CI), run the unit acceptance tests which simulate experiment outputs and skip preflight/training:

  ```bash
  SKIP_MICRO_PREFLIGHT=1 SKIP_MICRO_TRAIN=1 RUN_MICRO_OVERFIT=1 pytest tests/test_micro_overfit_acceptance.py -q
  ```

Cleanup behavior:

- When a micro-overfit is run as part of preflight you will be prompted interactively whether to remove experiment outputs after it finishes. In automated runs (CI or scripts) set `CLEANUP_AFTER_MICRO=1` to automatically delete `results/experiments/*` and `results/checkpoints/*` after a successful micro-overfit.

This test suite checks both the PASS and FAIL branches of the micro-overfit acceptance logic and is intentionally opt-in (`RUN_MICRO_OVERFIT=1`).

## RunPod persistent storage guidance
- When launching a RunPod instance, attach a persistent block storage volume and mount it somewhere stable (for example `/workspace` or `/workspace/storage`).
- Clone the repo onto the persistent volume so code, small datasets, and scripts are preserved across instance restarts:

```bash
cd /workspace
git clone https://github.com/<you>/amchi_asr.git
cd amchi_asr
sudo ./setup_env.sh  # sets up environment and runs non-failing preflight
./scripts/run_preflight_tests.sh  # run full checks and unit tests
```

- Keep experiment outputs on the attached volume by configuring `exp_manager.exp_dir` in your config or mounting `results/` to the persistent volume.

These steps let you spin up a new RunPod and be training-ready within minutes after preflight passes.

