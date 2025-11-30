WSL + NeMo Smoke Test
=====================

Purpose
-------
This document explains how to verify WSL + VS Code + GitHub Copilot are working together and how to run a local smoke test that loads the AI4Bharat Marathi IndicConformer model and runs inference on the small dev manifest.

Overview
--------
- Use WSL2 (Ubuntu) and VS Code Remote‑WSL so NeMo runs in Linux.
- Keep GitHub Copilot signed in to your local Windows VS Code — Copilot will continue to suggest code while you work in the Remote‑WSL window.

Quick WSL install (PowerShell as Administrator)
----------------------------------------------
```powershell
# Install WSL with Ubuntu 22.04 (Windows 10/11)
wsl --install -d Ubuntu-22.04
# Ensure WSL2
wsl --update
wsl --set-default-version 2
```

Open VS Code in WSL
-------------------
1. Install the VS Code extension `Remote - WSL` and `GitHub Copilot` in your Windows VS Code.
2. In VS Code: F1 → `Remote-WSL: New Window`.
3. In the new Remote-WSL window: open your repo from the WSL filesystem (recommended) or mount the Windows path.

Setup Python env and dependencies (inside WSL terminal)
-----------------------------------------------------
```bash
cd ~/code
# If you want to use the Windows repo copy, you can, but cloning inside WSL is recommended
# git clone <your-repo-url> amchi_konkani
cd amchi_konkani/konkani_asr

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# CPU PyTorch for smoke test
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install nemo-toolkit[all]==2.5.3 huggingface_hub transformers librosa jiwer soundfile

# Persist your HF token (set this to your token)
echo 'export HF_TOKEN="hf_pb..."' >> ~/.bashrc
source ~/.bashrc
```

Run the smoke test
------------------
From the repo root (inside the WSL Remote window) run:
```bash
source .venv/bin/activate
python scripts/nemo_finetune_smoke.py
```

What this does
---------------
- Downloads/loads the AI4Bharat Marathi model via HF (if not already cached).
- Runs inference on the two dev samples in `data/manifests/dev_small.json`.
- Prints transcriptions and WER for the examples.

If you see correct transcriptions and no exceptions, the environment is ready for a GPU run on the cloud. If you hit missing-file or tokenizer errors, copy the full model repo artifacts (model_config.yaml and tokenizer files) from HuggingFace using `huggingface_hub.hf_hub_download` or let me fetch them for you.

Next steps after smoke test
--------------------------
- Fix any issues locally.
- Provision a GPU VM (T4/A100) and use VS Code Remote‑SSH to connect (Copilot continues to work locally).
- Run the GPU finetune script (I'll prepare `scripts/nemo_finetune_gpu.py`) on the remote GPU VM.
