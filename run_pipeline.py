#!/usr/bin/env python3
"""
Quick start script for Konkani ASR fine-tuning
Handles the complete pipeline from setup to evaluation
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_command(cmd, description=""):
    """Run command and return success status"""
    print(f"\n🔧 {description}")
    print(f"Command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed (exit code: {e.returncode})")
        return False

def check_requirements():
    """Check if basic requirements are met"""
    print("🔍 Checking requirements...")

    # Check Python
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False

    # Check if in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if not in_venv:
        print("⚠️  Not in virtual environment. Consider using: python -m venv venv && venv\\Scripts\\activate")

    print("✅ Basic requirements check passed")
    return True

def setup_environment():
    """Setup the environment"""
    if not run_command("python scripts/setup_environment.py", "Setting up environment"):
        return False
    return True

def download_model():
    """Download the base model"""
    if not run_command("python scripts/download_model.py", "Downloading base model"):
        return False
    return True

def prepare_data(audio_dir, transcript_dir):
    """Prepare data for training"""
    cmd = f"python scripts/prepare_data.py --audio_dir {audio_dir} --transcript_dir {transcript_dir} --output_dir data"
    if not run_command(cmd, "Preparing data"):
        return False
    return True

def fine_tune_model(config_path):
    """Fine-tune the model"""
    cmd = f"python scripts/fine_tune.py --config {config_path}"
    if not run_command(cmd, "Fine-tuning model"):
        return False
    return True

def evaluate_model(model_path, test_manifest):
    """Evaluate the fine-tuned model"""
    cmd = f"python scripts/evaluate.py --model_path {model_path} --test_manifest {test_manifest}"
    if not run_command(cmd, "Evaluating model"):
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Quick start Konkani ASR fine-tuning")
    parser.add_argument("--audio_dir", help="Directory containing audio files")
    parser.add_argument("--transcript_dir", help="Directory containing transcript files")
    parser.add_argument("--config", default="configs/konkani_finetune.yaml", help="Configuration file")
    parser.add_argument("--skip_setup", action="store_true", help="Skip environment setup")
    parser.add_argument("--skip_download", action="store_true", help="Skip model download")
    parser.add_argument("--skip_training", action="store_true", help="Skip fine-tuning")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip evaluation")

    args = parser.parse_args()

    print("🚀 Konkani ASR Fine-tuning Pipeline")
    print("=" * 50)

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Setup environment
    if not args.skip_setup:
        if not setup_environment():
            print("❌ Environment setup failed")
            sys.exit(1)

    # Download model
    if not args.skip_download:
        if not download_model():
            print("❌ Model download failed")
            sys.exit(1)

    # Prepare data (if directories provided)
    if args.audio_dir and args.transcript_dir:
        if not prepare_data(args.audio_dir, args.transcript_dir):
            print("❌ Data preparation failed")
            sys.exit(1)
    elif not args.skip_training:
        print("⚠️  No audio/transcript directories provided. Skipping data preparation.")
        print("   To prepare data, run: python run_pipeline.py --audio_dir /path/to/audio --transcript_dir /path/to/transcripts")

    # Fine-tune model
    if not args.skip_training:
        if not os.path.exists("data/train.tsv"):
            print("❌ Training manifest not found. Run data preparation first.")
            sys.exit(1)

        if not fine_tune_model(args.config):
            print("❌ Fine-tuning failed")
            sys.exit(1)

    # Evaluate model
    if not args.skip_evaluation:
        model_path = "results/konkani_asr_final.nemo"
        test_manifest = "data/test.tsv"

        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            sys.exit(1)

        if not os.path.exists(test_manifest):
            print(f"❌ Test manifest not found: {test_manifest}")
            sys.exit(1)

        if not evaluate_model(model_path, test_manifest):
            print("❌ Evaluation failed")
            sys.exit(1)

    print("\n" + "=" * 50)
    print("🎉 Pipeline completed successfully!")
    print("=" * 50)

    print("\n📊 Results Summary:")
    print("- Environment: Set up")
    print("- Model: Downloaded" if not args.skip_download else "- Model: Skipped")
    print("- Data: Prepared" if args.audio_dir else "- Data: Skipped")
    print("- Training: Completed" if not args.skip_training else "- Training: Skipped")
    print("- Evaluation: Completed" if not args.skip_evaluation else "- Evaluation: Skipped")

    if not args.skip_evaluation:
        print("\n📈 Check evaluation_results.json for detailed metrics")

    print("\n📚 Next steps:")
    print("1. Review training logs in results/logs/")
    print("2. Check evaluation metrics in evaluation_results.json")
    print("3. Fine-tune hyperparameters if needed")
    print("4. Deploy model for inference")

if __name__ == "__main__":
    main()