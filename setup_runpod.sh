#!/bin/bash
# Setup script for RunPod Linux environment
# Run this after cloning the repository

set -e  # Exit on error

echo "=== Konkani ASR RunPod Setup ==="

# Check if running on Linux
if [[ "$(uname)" != "Linux" ]]; then
    echo "ERROR: This script is for Linux only"
    exit 1
fi

# Check for GPU
if ! nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. GPU may not be available."
else
    echo "✓ GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install NeMo and ASR dependencies
echo "Installing NeMo toolkit..."
pip install nemo_toolkit[asr]

# Install other requirements
echo "Installing other dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directory structure..."
mkdir -p data/train
mkdir -p data/dev  
mkdir -p data/test
mkdir -p models
mkdir -p logs
mkdir -p results

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Download model: python scripts/download_model.py"
echo "3. Download data: python scripts/download_data_from_railway.py"
echo "4. Run training: python scripts/fine_tune.py --config configs/konkani_finetune.yaml"
