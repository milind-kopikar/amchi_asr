#!/bin/bash
set -e  # Exit immediately if a command fails

echo "🚀 Starting Amchi-ASR Environment Setup..."

# 1. System Dependencies (Fixes pydub warning)
echo "📦 Installing system tools..."
apt-get update && apt-get install -y ffmpeg

# 2. GPU Enabler (The Critical Fix)
echo "🔥 Waking up the GPU (Reinstalling PyTorch for CUDA)..."
# Adjust the index URL if you need a different CUDA version
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Dependencies (From our manual installs)
echo "📚 Installing Python libraries..."
pip install "nemo_toolkit[all]" pynini librosa

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

echo "✅ Environment Setup Complete! GPU and Patch should be active."
