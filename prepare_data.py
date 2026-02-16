#!/usr/bin/env python3
"""
Data preparation script for Konkani ASR fine-tuning
Converts audio files and transcripts into NeMo-compatible manifest format
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd

# Optional imports
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_audio_duration(audio_path: str) -> float:
    """
    Get duration of audio file in seconds

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds
    """
    try:
        # Try soundfile first (faster for some formats)
        if SOUNDFILE_AVAILABLE:
            info = sf.info(audio_path)
            return info.duration
        else:
            raise ImportError("soundfile not available")
    except:
        try:
            # Fallback to librosa
            if LIBROSA_AVAILABLE:
                y, sr = librosa.load(audio_path, sr=None)
                return len(y) / sr
            else:
                raise ImportError("librosa not available")
        except:
            # Fallback to file size calculation (assuming 16kHz mono 16-bit WAV)
            try:
                file_size = os.path.getsize(audio_path)
                # For 16kHz mono 16-bit: 32000 bytes per second
                return file_size / 32000.0
            except Exception as e:
                logger.warning(f"Could not get duration for {audio_path}: {e}")
                return 0.0

def validate_audio_file(audio_path: str) -> bool:
    """
    Validate audio file format and properties

    Args:
        audio_path: Path to audio file

    Returns:
        True if valid, False otherwise
    """
    try:
        if SOUNDFILE_AVAILABLE:
            info = sf.info(audio_path)
            # Check sample rate (should be 16kHz)
            if abs(info.samplerate - 16000) > 100:  # Allow some tolerance
                logger.warning(f"Sample rate {info.samplerate}Hz for {audio_path}, recommended 16000Hz")
            # Check channels (should be mono)
            if info.channels != 1:
                logger.warning(f"{info.channels} channels for {audio_path}, recommended mono")
            return True
        else:
            raise ImportError("soundfile not available")
    except:
        # If soundfile not available, assume valid based on file extension
        if audio_path.lower().endswith('.wav'):
            logger.info(f"Soundfile not available, assuming {audio_path} is valid WAV")
            return True
        else:
            logger.warning(f"Cannot validate {audio_path} without soundfile")
            return True

def create_manifest_entry(audio_path: str, transcript: str, base_dir: str) -> Dict:
    """
    Create a manifest entry for NeMo

    Args:
        audio_path: Path to audio file
        transcript: Text transcript
        base_dir: Base directory for relative paths

    Returns:
        Manifest entry dictionary
    """
    # Get relative path
    rel_path = os.path.relpath(audio_path, base_dir)

    # Get duration
    duration = get_audio_duration(audio_path)

    # Validate audio
    is_valid = validate_audio_file(audio_path)

    return {
        "audio_filepath": rel_path,
        "text": transcript.strip(),
        "duration": round(duration, 3),
        "valid": is_valid
    }

def process_transcript_file(transcript_path: str) -> str:
    """
    Process transcript file to extract clean text

    Args:
        transcript_path: Path to transcript file

    Returns:
        Cleaned transcript text
    """
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        # Basic text cleaning
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove common artifacts
        text = text.replace('\n', ' ').replace('\r', ' ')

        return text
    except Exception as e:
        logger.error(f"Error reading transcript {transcript_path}: {e}")
        return ""

def find_audio_transcript_pairs(audio_dir: str, transcript_dir: str) -> List[Tuple[str, str]]:
    """
    Find matching audio and transcript file pairs

    Args:
        audio_dir: Directory containing audio files
        transcript_dir: Directory containing transcript files

    Returns:
        List of (audio_path, transcript_path) tuples
    """
    pairs = []

    # Get all audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(Path(audio_dir).rglob(f"*{ext}"))

    logger.info(f"Found {len(audio_files)} audio files")

    for audio_path in audio_files:
        # Try different transcript file patterns
        transcript_patterns = [
            audio_path.with_suffix('.txt'),
            audio_path.with_suffix('.transcript'),
            Path(transcript_dir) / audio_path.with_suffix('.txt').name,
            Path(transcript_dir) / audio_path.with_suffix('.transcript').name,
            Path(transcript_dir) / f"{audio_path.stem}.txt",
            Path(transcript_dir) / f"{audio_path.stem}.transcript"
        ]

        transcript_path = None
        for pattern in transcript_patterns:
            if pattern.exists():
                transcript_path = pattern
                break

        if transcript_path:
            pairs.append((str(audio_path), str(transcript_path)))
        else:
            logger.warning(f"No transcript found for audio: {audio_path}")

    logger.info(f"Found {len(pairs)} audio-transcript pairs")
    return pairs

def create_manifest(pairs: List[Tuple[str, str]], output_file: str, base_dir: str):
    """
    Create NeMo manifest file from audio-transcript pairs

    Args:
        pairs: List of (audio_path, transcript_path) tuples
        output_file: Output manifest file path
        base_dir: Base directory for relative paths
    """
    manifest_entries = []

    for audio_path, transcript_path in pairs:
        # Get transcript text
        transcript = process_transcript_file(transcript_path)

        if not transcript:
            logger.warning(f"Skipping {audio_path} - empty transcript")
            continue

        # Create manifest entry
        entry = create_manifest_entry(audio_path, transcript, base_dir)
        manifest_entries.append(entry)

    # Write manifest file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    logger.info(f"Created manifest with {len(manifest_entries)} entries: {output_file}")

    # Summary statistics
    total_duration = sum(entry['duration'] for entry in manifest_entries)
    valid_entries = sum(1 for entry in manifest_entries if entry['valid'])

    logger.info(".2f")
    logger.info(f"Valid audio files: {valid_entries}/{len(manifest_entries)}")

def split_manifest(manifest_file: str, train_ratio: float = 0.8, dev_ratio: float = 0.1):
    """
    Split manifest into train/dev/test sets

    Args:
        manifest_file: Input manifest file
        train_ratio: Ratio for training set
        dev_ratio: Ratio for development set
    """
    # Read manifest
    entries = []
    with open(manifest_file, 'r', encoding='utf-8') as f:
        for line in f:
            entries.append(json.loads(line.strip()))

    # Shuffle entries
    import random
    random.seed(42)  # For reproducibility
    random.shuffle(entries)

    # Split
    n_total = len(entries)
    n_train = int(n_total * train_ratio)
    n_dev = int(n_total * dev_ratio)
    n_test = n_total - n_train - n_dev

    train_entries = entries[:n_train]
    dev_entries = entries[n_train:n_train + n_dev]
    test_entries = entries[n_train + n_dev:]

    # Write split manifests
    base_path = Path(manifest_file).parent

    splits = [
        (train_entries, base_path / "train.tsv"),
        (dev_entries, base_path / "dev.tsv"),
        (test_entries, base_path / "test.tsv")
    ]

    for split_entries, output_file in splits:
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in split_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        logger.info(f"Created {output_file.name} with {len(split_entries)} entries")

def main():
    parser = argparse.ArgumentParser(description="Prepare data for Konkani ASR fine-tuning")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--transcript_dir", required=True, help="Directory containing transcript files")
    parser.add_argument("--output_dir", required=True, help="Output directory for manifests")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--dev_ratio", type=float, default=0.1, help="Development set ratio")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find audio-transcript pairs
    pairs = find_audio_transcript_pairs(args.audio_dir, args.transcript_dir)

    if not pairs:
        logger.error("No audio-transcript pairs found!")
        exit(1)

    # Create combined manifest
    combined_manifest = os.path.join(args.output_dir, "combined.json")
    create_manifest(pairs, combined_manifest, args.audio_dir)

    # Split into train/dev/test
    split_manifest(combined_manifest, args.train_ratio, args.dev_ratio)

    logger.info("Data preparation completed successfully!")

if __name__ == "__main__":
    main()