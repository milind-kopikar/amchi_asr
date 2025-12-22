#!/bin/bash
# Setup script for RunPod Linux environment
# Run this after cloning the repository
# 
# CRITICAL REQUIREMENTS:
# - Python 3.9 (AI4Bharat NeMo fork requires llvmlite==0.38.1)
# - CUDA 12.4 compatibility (numba>=0.57.0, llvmlite>=0.40.0)

set -e  # Exit on error

echo "=== Konkani ASR RunPod Setup ==="
echo "Last updated: December 22, 2025"
echo ""

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
    echo ""
fi

# Check Python version (must be 3.9)
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "Python version: $PYTHON_VERSION"

if [[ "$PYTHON_MAJOR" -eq 3 ]] && [[ "$PYTHON_MINOR" -eq 9 ]]; then
    echo "✓ Python 3.9 detected (required for AI4Bharat NeMo fork)"
    PYTHON_CMD=python3
elif command -v python3.9 &> /dev/null; then
    echo "✓ Found python3.9"
    PYTHON_CMD=python3.9
else
    echo "⚠️  Python 3.9 not found! Installing..."
    apt-get update
    apt-get install -y python3.9 python3.9-venv python3.9-dev
    PYTHON_CMD=python3.9
fi

echo ""

# Create virtual environment with Python 3.9
echo "Creating Python 3.9 virtual environment..."
$PYTHON_CMD -m venv venv
source venv/bin/activate

# Verify we're using Python 3.9
VENV_PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Virtual environment Python version: $VENV_PYTHON_VERSION"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install Cython first (required for youtokentome)
echo "Installing Cython (required for youtokentome)..."
pip install Cython
echo ""

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo ""

# Install youtokentome without build isolation
echo "Installing youtokentome..."
pip install --no-build-isolation youtokentome
echo ""

# Install AI4Bharat NeMo fork
echo "Installing AI4Bharat NeMo fork..."
pip install 'nemo_toolkit[asr] @ git+https://github.com/AI4Bharat/NeMo.git@multi-softmax'
echo ""

# CRITICAL: Upgrade numba and llvmlite for CUDA 12.4 compatibility
echo "Upgrading numba and llvmlite for CUDA 12.4 compatibility..."
pip install --upgrade 'numba>=0.57.0,<0.58' 'llvmlite>=0.40.0,<0.41'
echo ""

# Install other requirements
if [ -f requirements.txt ]; then
    echo "Installing additional requirements from requirements.txt..."
    pip install -r requirements.txt
    echo ""
fi

# Create necessary directories
echo "Creating directory structure..."
mkdir -p data/train
mkdir -p data/dev  
mkdir -p data/test
mkdir -p models/indicconformer_mr
mkdir -p models/indicconformer_kok
mkdir -p logs
mkdir -p results
echo ""

# Verify installations
echo "Verifying installation..."
python -c "import nemo; print(f'✓ NeMo version: {nemo.__version__}')"
python -c "import torch; print(f'✓ PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"
python -c "import numba; print(f'✓ Numba version: {numba.__version__}')"
python -c "import llvmlite; print(f'✓ llvmlite version: {llvmlite.__version__}')"
echo ""

echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo ""
echo "2. Download Konkani base model:"
echo "   python scripts/download_model.py --model konkani"
echo ""
echo "3. Download training data from Railway:"
echo "   python scripts/download_data_from_railway.py \\"
echo "     --base_url https://konkanicollector-production.up.railway.app \\"
echo "     --output_dir data/train \\"
echo "     --train_split 0.8"
echo ""
echo "4. Train model:"
echo "   python scripts/nemo_train.py \\"
echo "     --config configs/konkani_finetune.yaml \\"
echo "     --model konkani \\"
echo "     --output_dir results/konkani_full \\"
echo "     --max_epochs 50"
echo ""
echo "📚 For more details, see: RUNPOD_QUICK_START.md"
echo ""
