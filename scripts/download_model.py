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
    parser = argparse.ArgumentParser(description="Download AI4Bharat IndicConformer model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/wav2vec2-large-xlsr-53",
        help="Hugging Face model name"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="models/indicconformer_mr",
        help="Output directory for downloaded model"
    )
    parser.add_argument(
        "--auth_token",
        type=str,
        default=None,
        help="Hugging Face authentication token (if not set, uses HF_TOKEN from .env)"
    )

    args = parser.parse_args()

    # Use token from args or environment
    token = args.auth_token or os.getenv('HF_TOKEN')

    if not token:
        logger.error("No authentication token provided. Set HF_TOKEN in .env file or pass --auth_token")
        exit(1)

    try:
        download_model(args.model_name, args.output_path, token)
        logger.info("Model download completed successfully!")
    except Exception as e:
        logger.error(f"Model download failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()