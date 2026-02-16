#!/usr/bin/env python3
"""
Download audio recordings and manifests from Railway/R2 storage
For use with konkani_collector data.

Story split convention (--use_story_split):
  - Stories 1, 2, 3 -> train
  - Story 4 -> dev (validation during finetuning). Do not use for final test.
  - Story 5 -> test (held-out for evaluation). Always use Story 5 for testing.
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

def create_manifest(recordings: list, output_path: str, split: str = 'train', path_prefix: str = "data"):
    """Create NeMo-compatible manifest file with AI4Bharat required fields.
    path_prefix: base path for audio_filepath in manifest (e.g. 'data/amchi' or 'data/deaf')."""
    logger.info(f"Creating {split} manifest...")
    
    manifest_lines = []
    for idx, rec in enumerate(recordings):
        # NeMo manifest format with AI4Bharat multilingual fields
        manifest_entry = {
            "audio_filepath": f"{path_prefix}/{split}/audio/{rec['id']}.wav",
            "text": rec['sentence_text'],  # Devanagari text
            "duration": rec.get('duration', 0),
            "lang": "mr",  # Use "mr" (Marathi) for Konkani (AI4Bharat convention)
            "sample_id": f"{split}_{idx:04d}"  # Unique ID: train_0000, dev_0000, etc.
        }
        manifest_lines.append(json.dumps(manifest_entry, ensure_ascii=False))
    
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
        default="data",
        help="Base output directory (will create train/dev/test subdirs)"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.7,
        help="Fraction of data for training (default: 0.7 = 70%%)"
    )
    parser.add_argument(
        "--dev_split",
        type=float,
        default=0.15,
        help="Fraction of data for validation (default: 0.15 = 15%%)"
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.15,
        help="Fraction of data for testing (default: 0.15 = 15%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling (default: 42)"
    )
    parser.add_argument(
        "--use_story_split",
        action='store_true',
        help="If set, split data deterministically by story_id: stories 1,2,3 -> train; 5 -> dev; 4 -> test"
    )
    
    args = parser.parse_args()
    
    # Validate splits sum to 1.0
    total_split = args.train_split + args.dev_split + args.test_split
    if abs(total_split - 1.0) > 0.001:
        logger.error(f"Splits must sum to 1.0, got {total_split}")
        return 1
    
    try:
        # Fetch recordings
        recordings = fetch_recordings_list(args.base_url)
        
        if not recordings:
            logger.error("No recordings found")
            return 1

        # Story-based split (deterministic) if requested
        if args.use_story_split:
            logger.info("Using story-based splitting by story_id: {1,2,3}->train, 4->dev, 5->test")
            train_recordings = []
            dev_recordings = []
            test_recordings = []
            other = []
            for rec in recordings:
                sid = rec.get('story_id')
                try:
                    sid_int = int(sid)
                except Exception:
                    sid_int = None
                if sid_int in (1, 2, 3):
                    train_recordings.append(rec)
                elif sid_int == 4:
                    dev_recordings.append(rec)
                elif sid_int == 5:
                    test_recordings.append(rec)
                else:
                    other.append(rec)
            # If any recordings didn't match story ids, append them to train and warn
            if other:
                logger.warning(f"{len(other)} recordings had unknown or missing story_id; adding to train set")
                train_recordings.extend(other)
            logger.info(f"Split counts -> train: {len(train_recordings)}, dev: {len(dev_recordings)}, test: {len(test_recordings)} (total {len(train_recordings)+len(dev_recordings)+len(test_recordings)})")
        else:
            # IMPORTANT: Shuffle recordings randomly to mix speakers
            logger.info(f"🔀 Shuffling {len(recordings)} recordings (seed={args.seed})...")
            random.seed(args.seed)
            random.shuffle(recordings)
            logger.info("✓ Recordings shuffled to mix speakers across splits")
            # Calculate split indices
            n_train = int(len(recordings) * args.train_split)
            n_dev = int(len(recordings) * args.dev_split)
            # Remaining goes to test (handles rounding)
            train_recordings = recordings[:n_train]
            dev_recordings = recordings[n_train:n_train + n_dev]
            test_recordings = recordings[n_train + n_dev:]
        
        logger.info("="*80)
        logger.info(f"📊 Data Split (Total: {len(recordings)} samples)")
        logger.info("="*80)
        logger.info(f"  Train: {len(train_recordings)} samples ({len(train_recordings)/len(recordings)*100:.1f}%)")
        logger.info(f"  Dev:   {len(dev_recordings)} samples ({len(dev_recordings)/len(recordings)*100:.1f}%)")
        logger.info(f"  Test:  {len(test_recordings)} samples ({len(test_recordings)/len(recordings)*100:.1f}%)")
        logger.info("="*80)
        
        base_dir = Path(args.output_dir)
        
        # Path prefix for manifest audio_filepath (so paths work from repo root)
        path_prefix = str(base_dir)

        # Download train data
        if train_recordings:
            train_dir = base_dir / "train"
            logger.info(f"📥 Downloading training data to {train_dir}...")
            download_recordings(args.base_url, train_recordings, str(train_dir))
            create_manifest(train_recordings, str(train_dir / "manifest.jsonl"), "train", path_prefix)
        
        # Download dev data
        if dev_recordings:
            dev_dir = base_dir / "dev"
            logger.info(f"📥 Downloading validation data to {dev_dir}...")
            download_recordings(args.base_url, dev_recordings, str(dev_dir))
            create_manifest(dev_recordings, str(dev_dir / "manifest.jsonl"), "dev", path_prefix)
        
        # Download test data
        if test_recordings:
            test_dir = base_dir / "test"
            logger.info(f"📥 Downloading test data to {test_dir}...")
            download_recordings(args.base_url, test_recordings, str(test_dir))
            create_manifest(test_recordings, str(test_dir / "manifest.jsonl"), "test", path_prefix)
        
        logger.info("="*80)
        logger.info("✅ Data download complete!")
        logger.info("="*80)
        logger.info(f"📁 Train manifest: {train_dir / 'manifest.jsonl'} ({len(train_recordings)} samples)")
        logger.info(f"📁 Dev manifest:   {dev_dir / 'manifest.jsonl'} ({len(dev_recordings)} samples)")
        logger.info(f"📁 Test manifest:  {test_dir / 'manifest.jsonl'} ({len(test_recordings)} samples)")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
