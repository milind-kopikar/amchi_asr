#!/bin/bash
set -e  # Exit immediately if a command fails

echo "🚀 Starting Amchi-ASR Environment Setup..."

# 1. System Dependencies (Fixes pydub warning)
echo "📦 Installing system tools..."
apt-get update && apt-get install -y ffmpeg build-essential

# 2. GPU Enabler (The Critical Fix)
echo "🔥 Waking up the GPU (Reinstalling PyTorch for CUDA)..."
# Adjust the index URL if you need a different CUDA version
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Dependencies (From our manual installs)
echo "📚 Installing Python libraries..."
# Some build-time deps are required for packages such as pynini; install Cython first to avoid build failures
pip install --upgrade pip setuptools wheel
pip install Cython

# Allow explicit override to force upstream NeMo
if [ "${USE_UPSTREAM_NEMO:-0}" = "1" ]; then
  echo "ℹ️ USE_UPSTREAM_NEMO=1 set — installing upstream nemo_toolkit[all]"
  pip install "nemo_toolkit[all]" pynini librosa
else
  # Prefer the AI4Bharat NeMo fork (supports multi-softmax / hybrid RNNT)
  if pip install 'nemo_toolkit[asr] @ git+https://github.com/AI4Bharat/NeMo.git@multi-softmax'; then
    echo "✅ Installed AI4Bharat NeMo fork"
    pip install pynini librosa
  else
    echo "⚠️ AI4Bharat NeMo install failed; falling back to upstream nemo_toolkit[all]"
    pip install "nemo_toolkit[all]" pynini librosa
  fi
fi

# 4. The Magic Patch (The 6-hour fix)
echo "🔧 Patching NeMo Hybrid Engine..."
# Find where NeMo is installed
NEMO_FILE=$(python -c "import nemo.collections.asr.modules.conv_asr as m; print(m.__file__)" || true)
if [ -z "$NEMO_FILE" ]; then
  echo "⚠️ Could not find NeMo conv_asr location via modules.* path; trying parts.submodules path..."
  NEMO_FILE=$(python -c "import nemo.collections.asr.parts.submodules.conv_asr as m; print(m.__file__)" || true)
fi

if [ -z "$NEMO_FILE" ]; then
  echo "❌ Failed to locate installed NeMo conv_asr.py; aborting patch step."
  exit 1
fi

# Overwrite it with our vendored fix
cp patches/conv_asr_fixed.py "$NEMO_FILE"

# Ensure CUDA_VISIBLE_DEVICES is set (avoid empty value which hides GPUs)
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
  echo "⚠️ CUDA_VISIBLE_DEVICES is empty; setting to 0 to enable GPU visibility"
  export CUDA_VISIBLE_DEVICES=0
fi

# Quick validation of PyTorch CUDA availability
python - <<'PY'
import torch,sys
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
print('cuda device count', torch.cuda.device_count())
if torch.cuda.is_available() and torch.cuda.device_count()>0:
    print('device 0 name:', torch.cuda.get_device_name(0))
else:
    print('CUDA not available - please check GPU drivers or CUDA_VISIBLE_DEVICES')
    sys.exit(1)
PY

echo "✅ Environment Setup Complete! GPU and Patch should be active."
echo "🔐 Reminder: Run 'huggingface-cli login' (or set HF_TOKEN) before downloading AI4Bharat models from Hugging Face."
