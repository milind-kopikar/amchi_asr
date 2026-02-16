#!/usr/bin/env python3
"""
Download AI4Bharat IndicConformer Marathi ASR model from Hugging Face
"""

import os
import argparse
import logging
from pathlib import Path
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_model(model_name: str, output_path: str, token: str = None):
    """
    Download model from Hugging Face Hub

    Args:
        model_name: Hugging Face model name
        output_path: Local path to save the model
        token: Hugging Face authentication token
    """
    logger.info(f"Downloading model: {model_name}")
    logger.info(f"Output path: {output_path}")

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    try:
        # Download the model
        downloaded_path = snapshot_download(
            repo_id=model_name,
            local_dir=output_path,
            token=token
        )

        logger.info(f"Model downloaded successfully to: {downloaded_path}")

        # List downloaded files
        model_files = list(Path(output_path).rglob("*"))
        logger.info(f"Downloaded {len(model_files)} files")

        # Check for key files
        key_files = ["model.nemo", "tokenizer.model", "tokenizer.vocab"]
        for key_file in key_files:
            matches = list(Path(output_path).rglob(f"**/{key_file}"))
            if matches:
                logger.info(f"Found {key_file}: {matches[0]}")
            else:
                logger.warning(f"{key_file} not found in downloaded files")

        return downloaded_path

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Download AI4Bharat IndicConformer model",
        epilog="""
Examples:
  # Download Marathi model
  python download_model.py --model marathi
  
  # Download Konkani model  
  python download_model.py --model konkani
  
  # Custom model
  python download_model.py --model_name ai4bharat/some_other_model --output_path models/custom
        """
    )
    
    # Shorthand for common models
    parser.add_argument(
        "--model",
        type=str,
        choices=['marathi', 'konkani'],
        help="Shorthand: 'marathi' or 'konkani' (auto-sets model_name and output_path)"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Hugging Face model name (e.g., ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output directory for downloaded model"
    )
    parser.add_argument(
        "--auth_token",
        type=str,
        default=None,
        help="Hugging Face authentication token (if not set, uses HF_TOKEN from .env or huggingface-cli login)"
    )

    args = parser.parse_args()
    
    # Handle shorthand model names
    if args.model == 'marathi':
        model_name = "ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large"
        output_path = "models/indicconformer_mr"
    elif args.model == 'konkani':
        model_name = "ai4bharat/indicconformer_stt_kok_hybrid_ctc_rnnt_large"
        output_path = "models/indicconformer_kok"
    else:
        model_name = args.model_name
        output_path = args.output_path
    
    # Validate arguments
    if not model_name:
        logger.error("Please provide --model marathi/konkani OR --model_name <huggingface_model>")
        parser.print_help()
        exit(1)
    
    if not output_path:
        logger.error("Please provide --model marathi/konkani OR --output_path <directory>")
        parser.print_help()
        exit(1)

    # Use token from args or environment
    token = args.auth_token or os.getenv('HF_TOKEN')

    if not token:
        logger.warning("⚠️  No HF_TOKEN found - will use huggingface-cli login credentials")
        logger.info("💡 If download fails, run: huggingface-cli login")
        token = None  # huggingface_hub will use CLI credentials

        token = None  # huggingface_hub will use CLI credentials

    try:
        download_model(model_name, output_path, token)
        logger.info("="*80)
        logger.info("✅ Model download completed successfully!")
        logger.info(f"📂 Model saved to: {output_path}")
        logger.info("="*80)
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Verify model files:")
        logger.info(f"     ls -lh {output_path}/*.nemo")
        logger.info("  2. Run smoke tests:")
        logger.info("     python scripts/download_smoke_test_data.py")
        logger.info("     python tests/test_e2e_pipeline.py --test all")
        logger.info("  3. If tests pass, train on full data:")
        logger.info("     python scripts/download_data_from_railway.py")
        logger.info("     python scripts/nemo_train.py --config configs/konkani_finetune.yaml")
    except Exception as e:
        logger.error(f"❌ Model download failed: {e}")
        logger.info("")
        logger.info("Troubleshooting:")
        logger.info("  1. Accept model conditions on Hugging Face:")
        logger.info(f"     https://huggingface.co/{model_name}")
        logger.info("  2. Authenticate:")
        logger.info("     huggingface-cli login")
        logger.info("  3. Check your internet connection")
        exit(1)

if __name__ == "__main__":
    main()