#!/usr/bin/env python3
"""
Quick Smoke Test Data Downloader
Download minimal samples for rapid pipeline testing
"""

import os
import sys
import argparse
import logging
import json
import random
from pathlib import Path
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_file(url: str, output_path: str):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=os.path.basename(output_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

def fetch_recordings_list(base_url: str):
    """Fetch list of approved recordings from Railway API"""
    api_url = f"{base_url}/api/recordings"
    logger.info(f"Fetching recordings from {api_url}")
    
    response = requests.get(api_url)
    response.raise_for_status()
    
    recordings = response.json()
    
    # Filter for approved recordings only
    approved = [r for r in recordings if r.get('status') == 'approved']
    
    logger.info(f"Found {len(approved)} approved recordings out of {len(recordings)} total")
    return approved

def create_manifest(recordings: list, output_path: str, split: str = 'train'):
    """Create NeMo-compatible manifest file"""
    logger.info(f"Creating {split} manifest...")
    
    manifest_lines = []
    for rec in recordings:
        # NeMo manifest format
        manifest_entry = {
            "audio_filepath": f"data_smoke/{split}/audio/{rec['id']}.wav",
            "text": rec['sentence_text'],  # Devanagari text
            "duration": rec.get('duration', 0),
        }
        manifest_lines.append(json.dumps(manifest_entry))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(manifest_lines))
    
    logger.info(f"Created manifest with {len(manifest_lines)} entries: {output_path}")

def download_recordings(base_url: str, recordings: list, output_dir: str):
    """Download all audio files"""
    audio_dir = Path(output_dir) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {len(recordings)} audio files...")
    
    for rec in tqdm(recordings, desc="Downloading audio"):
        audio_url = f"{base_url}/api/recordings/{rec['id']}/audio"
        output_path = audio_dir / f"{rec['id']}.wav"
        
        try:
            download_file(audio_url, str(output_path))
        except Exception as e:
            logger.error(f"Failed to download recording {rec['id']}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download minimal data for smoke testing")
    parser.add_argument(
        "--base_url",
        type=str,
        default=os.getenv("RAILWAY_URL", "https://konkanicollector-production.up.railway.app"),
        help="Base URL of Railway deployment"
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=3,
        help="Number of training samples (default: 3)"
    )
    parser.add_argument(
        "--n_dev",
        type=int,
        default=1,
        help="Number of validation samples (default: 1)"
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=1,
        help="Number of test samples (default: 1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_smoke",
        help="Output directory for downloaded data (default: data_smoke)"
    )
    
    args = parser.parse_args()
    
    try:
        # Fetch recordings
        recordings = fetch_recordings_list(args.base_url)
        
        if not recordings:
            logger.error("No approved recordings found")
            return 1
        
        # Shuffle and sample
        random.seed(args.seed)
        random.shuffle(recordings)
        
        total_needed = args.n_train + args.n_dev + args.n_test
        if len(recordings) < total_needed:
            logger.error(f"Not enough recordings. Need {total_needed}, have {len(recordings)}")
            return 1
        
        # Sample subsets
        train_recordings = recordings[:args.n_train]
        dev_recordings = recordings[args.n_train:args.n_train + args.n_dev]
        test_recordings = recordings[args.n_train + args.n_dev:args.n_train + args.n_dev + args.n_test]
        
        logger.info("="*80)
        logger.info(f"🧪 SMOKE TEST DATA (seed={args.seed})")
        logger.info("="*80)
        logger.info(f"  Train: {len(train_recordings)} samples")
        logger.info(f"  Dev:   {len(dev_recordings)} samples")
        logger.info(f"  Test:  {len(test_recordings)} samples")
        logger.info("="*80)
        
        base_dir = Path(args.output_dir)
        
        # Download train data
        if train_recordings:
            train_dir = base_dir / "train"
            download_recordings(args.base_url, train_recordings, str(train_dir))
            create_manifest(train_recordings, str(train_dir / "manifest.jsonl"), "train")
        
        # Download dev data
        if dev_recordings:
            dev_dir = base_dir / "dev"
            download_recordings(args.base_url, dev_recordings, str(dev_dir))
            create_manifest(dev_recordings, str(dev_dir / "manifest.jsonl"), "dev")
        
        # Download test data
        if test_recordings:
            test_dir = base_dir / "test"
            download_recordings(args.base_url, test_recordings, str(test_dir))
            create_manifest(test_recordings, str(test_dir / "manifest.jsonl"), "test")
        
        logger.info("="*80)
        logger.info("✅ Smoke test data ready!")
        logger.info("="*80)
        logger.info(f"📁 {args.output_dir}/train/manifest.jsonl ({len(train_recordings)} samples)")
        logger.info(f"📁 {args.output_dir}/dev/manifest.jsonl ({len(dev_recordings)} samples)")
        logger.info(f"📁 {args.output_dir}/test/manifest.jsonl ({len(test_recordings)} samples)")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to download smoke test data: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
