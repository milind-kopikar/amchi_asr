#!/usr/bin/env bash
set -euo pipefail

# Idempotent environment provisioning script for the canonical training environment
# Usage: ./scripts/ensure_env.sh

echo "==> Ensuring canonical amchi_asr environment"

# 1) System packages
if ! command -v ffmpeg &>/dev/null; then
  echo "Installing system packages: ffmpeg build-essential (requires sudo)"
  sudo apt-get update && sudo apt-get install -y ffmpeg build-essential
else
  echo "ffmpeg already present"
fi

# 2) Python venv
PYTHON_BIN=${PYTHON_BIN:-python3}
VEVN_DIR=${VENV_DIR:-venv_py311}
if [ ! -d "$VEVN_DIR" ]; then
  echo "Creating venv: $VEVN_DIR (using $PYTHON_BIN)"
  $PYTHON_BIN -m venv $VEVN_DIR
fi

# Activate venv
source $VEVN_DIR/bin/activate

pip install --upgrade pip setuptools wheel

# 3) Core python deps
echo "Installing core requirements from requirements.txt (non-pinned)"
pip install -r requirements.txt

# 4) Critical packages: sentencepiece and nemo fork
if ! python -c "import sentencepiece" &>/dev/null; then
  echo "Installing sentencepiece"
  pip install sentencepiece
else
  echo "sentencepiece already installed"
fi

# Prefer AI4Bharat NeMo fork (editable install)
if ! python -c "import nemo" &>/dev/null; then
  echo "Installing AI4Bharat NeMo fork (editable)"
  pip install -e NeMo_ai4bharat || (echo "AI4Bharat NeMo install failed; installing upstream nemo_toolkit[all]" && pip install "nemo_toolkit[all]")
else
  echo "nemo already importable"
fi

# 5) Apply conv_asr patch if needed
python - <<'PY'
try:
    import nemo.collections.asr.modules.conv_asr as ca
    found = ca
    print('conv_asr module present at', ca.__file__)
except Exception as e:
    print('conv_asr not importable or missing:', e)
    raise SystemExit(0)
PY

# 6) Optional: download base model if requested (AUTO_DOWNLOAD_MODEL=1)
if [ "${AUTO_DOWNLOAD_MODEL:-0}" = "1" ]; then
  echo "AUTO_DOWNLOAD_MODEL=1 -> ensuring base .nemo model is present"
  python3 scripts/download_model_from_hf.py --repo ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large || echo "Model download failed; continue and let preflight report error"
fi

# 6) Safety: ensure env vars
export APPLY_CONV_PATCH=${APPLY_CONV_PATCH:-1}
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES=0
  echo "Set CUDA_VISIBLE_DEVICES=0"
fi

# 7) Final validation (print summary)
python3 scripts/dump_env_info.py

echo "Canonical environment ensured. If you want me to run the micro-overfit next, set RUN_MICRO_OVERFIT=1 and re-run ./scripts/run_preflight_tests.sh or tell me to proceed now."