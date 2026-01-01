# Amchi-ASR — Golden Environment Setup & Notes ✅

## Overview 🎯
This document captures the exact setup, fixes, and patches required to get the AI4Bharat + NeMo ASR experimentation environment working (the *state that took ~6 hours to reproduce*). Use `setup_env.sh` to reapply the environment setup & the critical NeMo patch.

> Note: This process is specifically for the AI4Bharat + NeMo approach used in this repo.

---

## Quick usage 🔧
1. Make the script executable: `chmod +x setup_env.sh`
2. Run it as root (it installs system packages): `sudo ./setup_env.sh`

The script will:
- Install `ffmpeg` to fix pydub warnings
- Reinstall PyTorch wheels for CUDA (cu118) to enable GPU
- Install required Python packages (nemo_toolkit and key libs)
- Copy our vendored `patches/conv_asr_fixed.py` over the installed NeMo `conv_asr.py`

Important environment variables set/used by the project:
- `APPLY_CONV_PATCH=1` (recommended): enables the runtime conv_asr patch that fixes language-id and mask handling.
- `LD_LIBRARY_PATH` should include your CUDA lib directory (for example `/usr/local/cuda/lib64`) if CUDA is installed in a non-standard location.

Data splitting policy (story-based):
- The `download_data_from_railway.py` script supports a `--use_story_split` flag that enforces deterministic, leakage-free splits: story_id 1/2/3 → train, story_id 5 → dev, story_id 4 → test. This is the canonical split used for experiments to avoid speaker/story leakage.

## Bootstrap / First-Time Host Notes 🐣
- **Hugging Face login (manual step):** The script cannot guess your Hugging Face token. On a new machine you'll need to run `huggingface-cli login` manually (or set `HF_TOKEN` in your environment) before attempting to download AI4Bharat models from the Hub.

- **Chicken & Egg (bootstrap command):** This repository and the setup script may not exist yet on a fresh host. Use this command to bootstrap a fresh machine in one step (replace YOUR_USER):

  ```bash
  git clone https://github.com/YOUR_USER/amchi_asr.git && cd amchi_asr && sudo ./setup_env.sh
  ```

- **Cython / build deps:** Some packages (e.g., `pynini`) require `Cython` and build tools to be installed before they can build. We install `Cython` early in the script (`pip install Cython`) and the script now installs `build-essential` via `apt-get` on fresh hosts so compilers and make are available for building wheels. If you prefer to pre-seed system build deps yourself, run `sudo apt-get install -y build-essential` before running `setup_env.sh`.

- **Hugging Face login reminder:** The `setup_env.sh` script will now print a final reminder to run `huggingface-cli login` (or set `HF_TOKEN`) after the setup finishes so you're ready to download model files.

- **Experiments / checkpointing / log location:** By default training configs point to a persistent experiments directory at `/workspace/amchi_asr/experiments`. The default checkpoint policy is to **save only the top 5** checkpoints (`save_top_k: 5`) to limit disk usage. You can change `exp_manager.exp_dir` in your config files if you prefer a different mount path.

---

## What we learned / environment details 📚
- Python version used: **3.11** (venv: `venv_py311`).
- Main NeMo workspace (fork used during debugging): `/workspace/NeMo_ai4bharat/` (local editable copy).
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
- Installed library file modified in-place: `/workspace/NeMo_ai4bharat/nemo/collections/asr/modules/conv_asr.py`
- Vendored snapshot (committed): `patches/conv_asr_fixed.py` (commit: d18f10e)

---

## Committed patches & how to use them 🗂️
- `patches/conv_asr_fixed.py`: A snapshot of the patched `conv_asr.py` that contains `_LanguageMaskList` & the safer `forward` implementation.
- The `setup_env.sh` script uses Python to locate the installed `conv_asr.py` and copies `patches/conv_asr_fixed.py` over it.

**GPU visibility note:** If `CUDA_VISIBLE_DEVICES` is defined but empty, GPU runtimes (PyTorch) will not see any devices. `setup_env.sh` will set `CUDA_VISIBLE_DEVICES=0` if it is empty and then validate PyTorch CUDA availability. If your environment uses multiple GPUs or a different device mapping, adjust `CUDA_VISIBLE_DEVICES` accordingly before running the script.


---

## Notes & caveats ⚠️
- The `setup_env.sh` script uses `apt-get` and installs PyTorch via pip wheel index; adjust the CUDA wheel selection if you need a different CUDA version.
- **Important**: `setup_env.sh` now prefers installing the **AI4Bharat NeMo fork** (the `multi-softmax` branch) because AI4Bharat models require fork-specific codepaths (e.g., `multisoftmax` in RNNT decoders). If for any reason you'd prefer the upstream NVIDIA NeMo package, set the environment variable `USE_UPSTREAM_NEMO=1` before running the script and it will install `nemo_toolkit[all]` instead of the fork.

  **Note**: The AI4Bharat fork is tested primarily with **Python 3.9** (see `AI4BHARAT_SETUP_GUIDE.md`). On newer Python versions you may run into dependency incompatibilities (e.g., `llvmlite`/`numba`); if that happens, either run the setup in a Python 3.9 venv or set `USE_UPSTREAM_NEMO=1` to install upstream NeMo instead.
- Running the script as non-root may fail the `apt-get` step.
- This process is specific to the AI4Bharat + NeMo setup used here and may not be appropriate for other workflows.

---

## Suggested next steps
- Add unit tests around `ConvASRDecoder.forward` verifying the handling of: scalar torch language ids, list of tensor masks, boolean masks, numpy arrays.
- Optionally create an upstream PR to the NeMo fork for maintainability and to prevent regressions.

---

If anything here needs to be expanded (e.g., more detailed package versions, Dockerfile, or CI steps), tell me which area to prioritize and I will add it to this document and/or the repo.
