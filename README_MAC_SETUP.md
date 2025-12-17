**macOS Setup for NeMo / AI4Bharat finetuning (Konkani)**

This document collects steps and recommendations to prepare a macOS machine for developing and smoke-testing the NeMo-based finetuning workflow in this repository. It focuses on practical steps that will allow you to: (A) run small smoke tests locally, (B) reliably reproduce the Linux environment using Docker as a fallback, and (C) prepare a cloud-ready Dockerfile for true GPU finetuning.

Notes up front
- NeMo (nemo-toolkit) and many production ASR training stacks are developed and tested primarily on Linux. On macOS you can do most development and small CPU smoke tests, but full GPU finetuning should be done on a Linux GPU instance (cloud or Linux machine).
- Apple Silicon (M1/M2) can run many Python packages, but some binary wheels (CUDA, some NeMo internals) are x86_64-only. If your Mac is Apple Silicon prefer Miniforge/conda for compatibility and consider using Linux (Docker or cloud) for heavy runs.

Table of contents
- Prerequisites & quick checks
- macOS (Intel) recommended steps (fastest path for smoke tests)
- macOS (Apple Silicon) recommended steps (Miniforge / compatibility notes)
- Common package fixes & troubleshooting (dill, antlr, jiwer, lightning)
- How to run the repo smoke test (commands)
- Recommended fallback: Docker (Linux) + example Dockerfile snippet
- Quick checklist before moving to cloud

1) Prerequisites & quick checks
- Open Terminal and check CPU architecture:
```
uname -m
```
- If output is `x86_64` it’s Intel mac; if `arm64` it’s Apple Silicon.
- Install Homebrew if missing: https://brew.sh/

2) macOS (Intel) — recommended quick setup (CPU smoke tests)
- Use system Python or a virtualenv for small, CPU-only smoke tests.
- Commands (copy & paste):
```bash
# create project venv
python3 -m venv .venv_mac
source .venv_mac/bin/activate

# upgrade pip
pip install --upgrade pip setuptools wheel

# Install CPU PyTorch (choose the latest cpu wheel URL if needed)
pip install "torch==2.9.1+cpu" --index-url https://download.pytorch.org/whl/cpu

# install NeMo (note: may fail for some mac setups)
pip install nemo-toolkit==2.5.3

# small deps used by scripts
pip install jiwer librosa soundfile huggingface_hub transformers omegaconf hydra-core antlr4-python3-runtime dill
```

Notes:
- If `pip install nemo-toolkit` fails on macOS, prefer the Docker fallback (below) or run smoke tests on a Linux VM.

3) macOS (Apple Silicon) — Miniforge / Conda recommended
- Install Miniforge (recommended) — it provides conda packages built for arm64.
  - https://github.com/conda-forge/miniforge
- Create a conda env and install packages (example):
```bash
# create conda env
conda create -n amchi_asr python=3.10 -y
conda activate amchi_asr

# Try installing torch for macOS (metal) via conda/conda-forge if available
conda install -c pytorch -c conda-forge pytorch cpuonly -y

# Install other deps via pip or conda-forge
pip install jiwer librosa soundfile huggingface_hub transformers omegaconf hydra-core antlr4-python3-runtime dill

# Try installing nemo-toolkit; if it fails, use Docker or a Linux VM for NeMo runs
pip install nemo-toolkit==2.5.3 || echo "nemo-toolkit may not be supported on Apple Silicon — use Docker or cloud Linux"
```

4) Common package fixes & troubleshooting notes
- jiwer: Some versions expose different symbols. If `from jiwer import wer` fails, use `pip install jiwer==2.5.1` or let the script use the builtin fallback.
- dill: If you see `AttributeError: module 'dill' has no attribute 'extend'`, pin `dill==0.3.6`.
- antlr/hydra/omegaconf: use `antlr4-python3-runtime==4.9.3`, `hydra-core==1.3.2`, `omegaconf==2.3.0` to match NeMo 2.5.3 compatibility.
- lightning (PyTorch Lightning): NeMo imports may require `lightning` (2.x) — install `pip install lightning`.

5) Run the repo smoke test on mac (CPU)
- In the project root (where `scripts/nemo_finetune_smoke.py` lives):
```bash
# activate venv or conda env
source .venv_mac/bin/activate   # or `conda activate amchi_asr`

# set HF token (replace with your token) and HF cache path (optional)
export HF_TOKEN="hf_xxx"
export HF_HOME="$HOME/.cache/huggingface"

# run the smoke test (may attempt to download model files)
python scripts/nemo_finetune_smoke.py
```

If the script fails to import NeMo or `ASRModel` isn't available, that means your installed `nemo-toolkit` is not compatible with macOS or the package layout differs — use Docker (below) or run the smoke test on a Linux VM.

6) Recommended fallback: Docker (Linux) — best for NeMo & reproducibility
- Because NeMo is Linux-first and finetuning requires many Linux binaries, Docker is often the simplest reproducible approach on macOS. Docker Desktop on mac can run Linux containers (note: GPU passthrough is not available on Mac; GPU training still requires a Linux GPU machine). Use Docker to validate runtime and reproduce the cloud environment.

Example Dockerfile (Linux, CUDA-enabled — use on cloud or Linux box with GPU):
```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-venv python3-pip ffmpeg git wget build-essential
RUN python3 -m pip install --upgrade pip
# Install PyTorch (CUDA) - choose matching torch+cu wheel or use pip index
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Install NeMo and deps
RUN pip install nemo-toolkit==2.5.3 hydra-core==1.3.2 omegaconf==2.3.0 antlr4-python3-runtime==4.9.3 dill==0.3.6 jiwer librosa soundfile huggingface_hub transformers
WORKDIR /workspace
COPY . /workspace
ENTRYPOINT ["/bin/bash"]
```

Notes:
- Build this Dockerfile on a Linux machine with an NVIDIA GPU driver and `nvidia-docker2` / `--gpus all` support. On macOS Docker Desktop this will run but without GPU acceleration (useful for quick validation only).

7) Quick checklist before cloud finetuning
- Ensure manifests (`data/manifests/train_small.json` and `dev_small.json`) are correct and accessible by the training script. Paths should be valid inside the runtime container or VM.
- Confirm HF access: `export HF_TOKEN=...` and `huggingface_hub` can download the AI4Bharat model.
- Verify sample transcriptions with `scripts/nemo_finetune_smoke.py` on a small subset.
- Prepare a Dockerfile and an entrypoint that saves checkpoints to a mounted volume or cloud storage (S3/GCS).
- Choose cloud GPU (e.g., AWS p3/p4, GCP A100, or similar) and matching CUDA toolkit for PyTorch.

8) If something breaks — what to share with a helper/agent
- The full traceback of the failure.
- Output of these quick checks from the environment where you ran them:
```bash
python -c "import nemo; print('nemo', getattr(nemo,'__version__', 'no-version'))"
python -c "import nemo.collections.asr as nasr; print([n for n in dir(nasr) if not n.startswith('__')])"
pip freeze | grep -E 'nemo|torch|lightning|jiwer|dill|antlr' || true
```

9) Notes about GitHub Copilot on Mac
- After installing Copilot in VS Code on mac, it will operate normally editing remote or local files. Copilot suggestions are not blocked by the OS. If you use Remote‑SSH later to connect to a Linux cloud VM, Copilot will continue to work inside the remote workspace.

10) Useful references
- NeMo docs: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/
- AI4Bharat model hub (example): https://huggingface.co/ai4bharat

If you want, I can also:
- Produce a ready-to-run `requirements.txt` tuned for a Mac Intel environment.
- Create the Dockerfile as a file in this repo with a `run_cloud.sh` helper that mounts data and checkpoints.

---
Last updated: Nov 29, 2025
