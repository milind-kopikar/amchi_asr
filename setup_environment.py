#!/usr/bin/env python3
"""
Environment setup script for Konkani ASR fine-tuning
Checks system requirements and installs dependencies
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(cmd, description=""):
    """Run shell command and return success status"""
    try:
        print(f"🔧 {description}")
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_cuda_availability():
    """Check CUDA availability for GPU acceleration"""
    print("🎮 Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            print(f"✅ CUDA available - {device_count} device(s)")
            print(f"   Current device: {device_name}")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU (slower)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed - CUDA check skipped")
        return False

def check_system_requirements():
    """Check basic system requirements"""
    print("💻 Checking system requirements...")

    system = platform.system().lower()
    print(f"   OS: {platform.system()} {platform.release()}")

    # Check FFmpeg
    ffmpeg_ok = run_command("ffmpeg -version", "Checking FFmpeg installation")
    if not ffmpeg_ok:
        print("   ⚠️  FFmpeg not found - required for audio processing")
        print("   Install from: https://ffmpeg.org/download.html")
        print("   Or use: conda install ffmpeg")

    return True

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Python dependencies installation...")

    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    if not requirements_file.exists():
        print(f"❌ Requirements file not found: {requirements_file}")
        return False

    print("   To install dependencies, run one of these commands:")
    print(f"   pip install -r {requirements_file}")
    print(f"   python -m pip install -r {requirements_file}")
    print("   conda install --file requirements.txt")
    print("\n   Key packages needed:")
    with open(requirements_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                print(f"   - {line.strip()}")

    print("\n   Note: If pip is not available, install it first:")
    print("   python -m ensurepip --upgrade")
    print("   Or download get-pip.py from https://bootstrap.pypa.io/get-pip.py")

    # Try to install anyway
    success = run_command(f"python -m pip install --upgrade pip", "Upgrading pip")
    if not success:
        print("   ⚠️  Could not upgrade pip - you may need to install it manually")

    success = run_command(f"python -m pip install -r {requirements_file}", "Installing dependencies")
    if success:
        print("✅ Dependencies installed successfully")
        return True
    else:
        print("❌ Failed to install dependencies automatically")
        print("   Please install manually using the commands above")
        return False

def setup_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")

    directories = [
        "data/audio",
        "data/transcripts",
        "models",
        "results/checkpoints",
        "results/logs",
        "logs"
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {dir_path}")

    print("✅ Directories created successfully")
    return True

def verify_installation():
    """Verify that key packages are installed correctly"""
    print("🔍 Verifying installation...")

    packages_to_check = [
        ("torch", "PyTorch"),
        ("nemo", "NVIDIA NeMo"),
        ("librosa", "Librosa"),
        ("soundfile", "SoundFile"),
        ("omegaconf", "OmegaConf"),
        ("jiwer", "JIWER")
    ]

    all_ok = True
    for package, name in packages_to_check:
        try:
            __import__(package)
            print(f"   ✅ {name} - OK")
        except ImportError:
            print(f"   ❌ {name} - Missing")
            all_ok = False

    if all_ok:
        print("✅ All key packages verified successfully")
    else:
        print("❌ Some packages are missing. Run 'pip install -r requirements.txt' again")

    return all_ok

def main():
    """Main setup function"""
    print("🚀 Konkani ASR Fine-tuning Environment Setup")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        print("❌ Setup failed - Python version incompatible")
        sys.exit(1)

    # Check system requirements
    check_system_requirements()

    # Check CUDA
    check_cuda_availability()

    # Setup directories
    if not setup_directories():
        print("❌ Setup failed - Could not create directories")
        sys.exit(1)

    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed - Could not install dependencies")
        sys.exit(1)

    # Verify installation
    if not verify_installation():
        print("❌ Setup completed with warnings - Some packages may be missing")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("🎉 Environment setup completed successfully!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Place your Konkani audio files in data/audio/")
    print("2. Place corresponding transcripts in data/transcripts/")
    print("3. Run: python scripts/download_model.py")
    print("4. Run: python scripts/prepare_data.py --audio_dir data/audio --transcript_dir data/transcripts --output_dir data")
    print("5. Run: python scripts/fine_tune.py --config configs/konkani_finetune.yaml")
    print("\nFor detailed instructions, see README.md")

if __name__ == "__main__":
    main()