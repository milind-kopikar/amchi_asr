#!/usr/bin/env bash
set -euo pipefail

# Smoke run helper for MMS/other HF models
# Usage: ./scripts/smoke_run.sh [--hf-home /path/to/cache] [--gpu]

HF_HOME_ARG=""
USE_GPU=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-home) HF_HOME_ARG="$2"; shift 2;;
    --gpu) USE_GPU=1; shift;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -n "$HF_HOME_ARG" ]]; then
  echo "Using HF_HOME=$HF_HOME_ARG"
  export HF_HOME="$HF_HOME_ARG"
fi

# create venv
python3 -m venv .venv || true
. .venv/bin/activate
python -m pip install --upgrade pip

if [[ "$USE_GPU" -eq 1 ]]; then
  echo "--gpu selected: please install the correct torch wheel for your CUDA version manually if needed"
  echo "Installing common packages"
  python -m pip install --upgrade pip
  python -m pip install transformers datasets librosa soundfile scipy evaluate jiwer
else
  echo "Installing CPU PyTorch and minimal packages"
  python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
  python -m pip install transformers datasets librosa soundfile scipy evaluate jiwer
fi

# Download smoke data
python scripts/download_smoke_test_data.py --output_dir data_smoke
mkdir -p data/test_run
cp data_smoke/train/manifest.jsonl data/test_run/train.tsv
cp data_smoke/dev/manifest.jsonl data/test_run/dev.tsv
cp data_smoke/test/manifest.jsonl data/test_run/test.tsv

# Backup main config and point to MMS smoke config
cp configs/main_config.yaml configs/main_config.yaml.bak || true
python - <<'PY'
from pathlib import Path
p=Path('configs/main_config.yaml')
s=p.read_text()
s=s.replace('config_file: "configs/huggingface_config.smoke.yaml"','config_file: "configs/mms_smoke.yaml"')
p.write_text(s)
print('Patched main_config to use configs/mms_smoke.yaml')
PY

# Run training smoke (will download model if required)
mkdir -p results/smoke
python scripts/fine_tune_hf.py 2>&1 | tee results/smoke/train.log || true

# Run inference smoke
python scripts/smoke_infer.py --model facebook/mms-1b-all --manifest data_smoke/test/manifest.jsonl --limit 5

# Restore main_config
mv configs/main_config.yaml.bak configs/main_config.yaml || true

echo "Smoke run complete. Logs: results/smoke/train.log"
