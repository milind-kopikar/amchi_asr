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

---

## What we learned / environment details 📚
- Python version used: **3.11** (venv: `venv_py311`).
- Main NeMo workspace (fork used during debugging): `/workspace/NeMo_ai4bharat/` (local editable copy).
- Important packages & versions (examples):
  - torch / torchvision / torchaudio: installed from the official PyTorch CUDA 11.8 wheel (matching CUDA available in the runpod).
  - nemo_toolkit: `pip install "nemo_toolkit[all]"` (used for training & decoding features).
  - pynini, librosa: other libs used by preprocessing and tokenization.

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

---

## Notes & caveats ⚠️
- The `setup_env.sh` script uses `apt-get` and installs PyTorch via pip wheel index; adjust the CUDA wheel selection if you need a different CUDA version.
- Running the script as non-root may fail the `apt-get` step.
- This process is specific to the AI4Bharat + NeMo setup used here and may not be appropriate for other workflows.

---

## Suggested next steps
- Add unit tests around `ConvASRDecoder.forward` verifying the handling of: scalar torch language ids, list of tensor masks, boolean masks, numpy arrays.
- Optionally create an upstream PR to the NeMo fork for maintainability and to prevent regressions.

---

If anything here needs to be expanded (e.g., more detailed package versions, Dockerfile, or CI steps), tell me which area to prioritize and I will add it to this document and/or the repo.
