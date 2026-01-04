#!/usr/bin/env python3
"""
Download deaf speech recordings and manifests from Railway
Filters for a specific user and splits into train/dev/test.
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
        leave=False
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

def fetch_recordings_list(base_url: str, user_prefix: str):
    """Fetch list of approved recordings from Railway API and filter by user"""
    api_url = f"{base_url}/api/recordings"
    logger.info(f"Fetching recordings from {api_url}")
    
    response = requests.get(api_url)
    response.raise_for_status()
    
    recordings = response.json()
    
    # Filter for approved recordings and specific user
    # User email starts with t***y@gmail.com
    filtered = [
        r for r in recordings 
        if r.get('status') == 'approved' and r.get('user_id', '').startswith(user_prefix)
    ]
    
    logger.info(f"Found {len(filtered)} approved recordings for user {user_prefix}* out of {len(recordings)} total")
    return filtered

def create_manifest(recordings: list, output_path: str, split: str):
    """Create NeMo-compatible manifest file"""
    logger.info(f"Creating {split} manifest...")
    
    manifest_lines = []
    for idx, rec in enumerate(recordings):
        manifest_entry = {
            "audio_filepath": f"data/{split}/audio/{rec['id']}.wav",
            "text": rec['sentence_text'],
            "duration": rec.get('duration', 0),
            "lang": "mr",
            "sample_id": f"{split}_{idx:04d}"
        }
        manifest_lines.append(json.dumps(manifest_entry, ensure_ascii=False))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(manifest_lines))
    
    logger.info(f"Created manifest with {len(manifest_lines)} entries: {output_path}")

def download_recordings(base_url: str, recordings: list, output_dir: str):
    """Download all audio files"""
    audio_dir = Path(output_dir) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {len(recordings)} audio files to {audio_dir}...")
    
    for rec in tqdm(recordings, desc="Downloading audio"):
        audio_url = f"{base_url}/api/recordings/{rec['id']}/audio"
        output_path = audio_dir / f"{rec['id']}.wav"
        
        if output_path.exists():
            continue
            
        try:
            download_file(audio_url, str(output_path))
        except Exception as e:
            logger.error(f"Failed to download recording {rec['id']}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download deaf speech data from Railway")
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://deafspeechcollector-production.up.railway.app",
        help="Base URL of Railway deployment"
    )
    parser.add_argument(
        "--user_prefix",
        type=str,
        required=True,
        help="Prefix of the user email (e.g., 't')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for data"
    )
    
    args = parser.parse_args()
    
    # 1. Fetch and filter
    recordings = fetch_recordings_list(args.base_url, args.user_prefix)
    
    if not recordings:
        logger.error("No recordings found for the specified user.")
        return

    # 2. Shuffle and Split (80/10/10)
    random.seed(42)
    random.shuffle(recordings)
    
    n = len(recordings)
    train_end = int(n * 0.8)
    dev_end = train_end + int(n * 0.1)
    
    train_recs = recordings[:train_end]
    dev_recs = recordings[train_end:dev_end]
    test_recs = recordings[dev_end:]
    
    logger.info(f"Split: Train={len(train_recs)}, Dev={len(dev_recs)}, Test={len(test_recs)}")
    
    # 3. Download and Create Manifests
    splits = [
        ('train', train_recs),
        ('dev', dev_recs),
        ('test', test_recs)
    ]
    
    for split_name, split_recs in splits:
        split_dir = Path(args.output_dir) / split_name
        create_manifest(split_recs, str(split_dir / "manifest.jsonl"), split_name)
        download_recordings(args.base_url, split_recs, str(split_dir))

if __name__ == "__main__":
    main()
