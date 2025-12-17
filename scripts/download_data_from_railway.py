#!/usr/bin/env python3
"""
Download audio recordings and manifests from Railway/R2 storage
For use with konkani_collector data
"""

import os
import sys
import argparse
import logging
import json
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
            "audio_filepath": f"data/{split}/audio/{rec['id']}.wav",
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
    parser = argparse.ArgumentParser(description="Download data from konkani_collector Railway deployment")
    parser.add_argument(
        "--base_url",
        type=str,
        default=os.getenv("RAILWAY_URL", "https://konkanicollector-production.up.railway.app"),
        help="Base URL of Railway deployment"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/train",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Fraction of data for training (rest goes to dev)"
    )
    
    args = parser.parse_args()
    
    try:
        # Fetch recordings
        recordings = fetch_recordings_list(args.base_url)
        
        if not recordings:
            logger.error("No approved recordings found")
            return 1
        
        # Split into train/dev
        split_idx = int(len(recordings) * args.train_split)
        train_recordings = recordings[:split_idx]
        dev_recordings = recordings[split_idx:]
        
        logger.info(f"Split: {len(train_recordings)} train, {len(dev_recordings)} dev")
        
        # Download train data
        train_dir = Path(args.output_dir)
        logger.info(f"Downloading training data to {train_dir}...")
        download_recordings(args.base_url, train_recordings, str(train_dir))
        create_manifest(train_recordings, str(train_dir / "manifest.jsonl"), "train")
        
        # Download dev data
        if dev_recordings:
            dev_dir = Path(args.output_dir).parent / "dev"
            logger.info(f"Downloading dev data to {dev_dir}...")
            download_recordings(args.base_url, dev_recordings, str(dev_dir))
            create_manifest(dev_recordings, str(dev_dir / "manifest.jsonl"), "dev")
        
        logger.info("✓ Data download complete!")
        logger.info(f"Train manifest: {train_dir / 'manifest.jsonl'}")
        if dev_recordings:
            logger.info(f"Dev manifest: {dev_dir / 'manifest.jsonl'}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
