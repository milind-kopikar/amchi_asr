#!/usr/bin/env python3
"""
ASR Manifest Preparation: Create NeMo-compatible manifest files from audio-text corpus
Prepares training data for NVIDIA NeMo ASR fine-tuning
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ASRManifestCreator:
    """Create NeMo ASR manifest files from audio-text pairs"""

    def __init__(self, corpus_dir: str, output_dir: str):
        """
        Initialize manifest creator

        Args:
            corpus_dir: Directory containing audio-text pairs
            output_dir: Directory to save manifest files
        """
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_audio_text_pairs(self) -> List[Dict[str, str]]:
        """
        Find all audio-text pairs in the corpus directory

        Returns:
            List of dictionaries with 'audio' and 'text' keys
        """
        pairs = []

        # Find all WAV files
        wav_files = list(self.corpus_dir.glob("*.wav"))

        for wav_file in sorted(wav_files):
            # Find corresponding text file
            txt_file = wav_file.with_suffix('.txt')

            if txt_file.exists():
                pairs.append({
                    'audio': str(wav_file),
                    'text': str(txt_file)
                })
            else:
                logger.warning(f"No text file found for: {wav_file}")

        logger.info(f"Found {len(pairs)} audio-text pairs")
        return pairs

    def create_manifest_entry(self, audio_path: str, text_path: str):
        """
        Create a single manifest entry for NeMo

        Args:
            audio_path: Path to audio file
            text_path: Path to text file

        Returns:
            Manifest entry dictionary
        """
        # Read text
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        # Get audio duration (you could use librosa or ffprobe here)
        # For now, we'll set duration to None and let NeMo calculate it
        duration = None

        # Create manifest entry
        entry = {
            "audio_filepath": audio_path,
            "text": text,
            "duration": duration
        }

        return entry

    def create_train_val_split(self, pairs: List[Dict[str, str]], val_ratio: float = 0.2) -> tuple:
        """
        Split pairs into training and validation sets

        Args:
            pairs: List of audio-text pairs
            val_ratio: Ratio of data to use for validation

        Returns:
            Tuple of (train_pairs, val_pairs)
        """
        # For small datasets, ensure minimum validation samples
        min_val_samples = max(1, int(len(pairs) * val_ratio))

        # Simple split: use last min_val_samples for validation
        train_pairs = pairs[:-min_val_samples] if len(pairs) > min_val_samples else pairs
        val_pairs = pairs[-min_val_samples:] if len(pairs) > min_val_samples else pairs[:1]  # At least 1 for val

        logger.info(f"Split: {len(train_pairs)} train, {len(val_pairs)} validation")
        return train_pairs, val_pairs

    def create_manifest_file(self, pairs: List[Dict[str, str]], output_path: str) -> int:
        """
        Create a manifest file from audio-text pairs

        Args:
            pairs: List of audio-text pairs
            output_path: Path to output manifest file

        Returns:
            Number of entries created
        """
        manifest_entries = []

        for pair in pairs:
            try:
                entry = self.create_manifest_entry(pair['audio'], pair['text'])
                manifest_entries.append(entry)
            except Exception as e:
                logger.error(f"Failed to create entry for {pair['audio']}: {str(e)}")

        # Write manifest file (one JSON per line)
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in manifest_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        logger.info(f"Created manifest: {output_path} ({len(manifest_entries)} entries)")
        return len(manifest_entries)

    def create_all_manifests(self, val_ratio: float = 0.2) -> Dict[str, int]:
        """
        Create all manifest files (train, val, and combined)

        Args:
            val_ratio: Ratio of data for validation

        Returns:
            Dictionary with manifest file counts
        """
        # Find all pairs
        pairs = self.find_audio_text_pairs()

        if not pairs:
            logger.error("No audio-text pairs found!")
            return {}

        # Create train/val split
        train_pairs, val_pairs = self.create_train_val_split(pairs, val_ratio)

        # Create manifest files
        results = {}

        # Training manifest
        train_manifest = self.output_dir / "train_manifest.json"
        results['train'] = self.create_manifest_file(train_pairs, str(train_manifest))

        # Validation manifest
        val_manifest = self.output_dir / "val_manifest.json"
        results['val'] = self.create_manifest_file(val_pairs, str(val_manifest))

        # Combined manifest (for testing)
        all_manifest = self.output_dir / "all_manifest.json"
        results['all'] = self.create_manifest_file(pairs, str(all_manifest))

        return results

    def validate_manifests(self) -> Dict[str, Any]:
        """
        Validate created manifest files

        Returns:
            Validation results
        """
        validation_results = {}

        manifest_files = ['train_manifest.json', 'val_manifest.json', 'all_manifest.json']

        for manifest_file in manifest_files:
            manifest_path = self.output_dir / manifest_file

            if not manifest_path.exists():
                validation_results[manifest_file] = {'exists': False}
                continue

            try:
                # Count entries
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    entries = [json.loads(line.strip()) for line in f if line.strip()]

                # Check each entry
                valid_entries = 0
                total_duration = 0

                for entry in entries:
                    if 'audio_filepath' in entry and 'text' in entry:
                        valid_entries += 1
                        # Could check if audio file exists here

                validation_results[manifest_file] = {
                    'exists': True,
                    'entries': len(entries),
                    'valid_entries': valid_entries
                }

            except Exception as e:
                validation_results[manifest_file] = {
                    'exists': True,
                    'error': str(e)
                }

        return validation_results

def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Create NeMo ASR manifest files from audio-text corpus")
    parser.add_argument("corpus_dir", help="Directory containing audio-text pairs")
    parser.add_argument("output_dir", help="Directory to save manifest files")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation data ratio (default: 0.2)")
    parser.add_argument("--validate", action="store_true", help="Validate created manifests")

    args = parser.parse_args()

    print("🎵 ASR Manifest Creator")
    print("=" * 40)
    print(f"Corpus:   {args.corpus_dir}")
    print(f"Output:   {args.output_dir}")
    print(f"Val ratio: {args.val_ratio}")
    print()

    # Create manifests
    creator = ASRManifestCreator(args.corpus_dir, args.output_dir)
    results = creator.create_all_manifests(args.val_ratio)

    print("\n" + "=" * 40)
    print("📊 MANIFEST CREATION RESULTS")
    print("=" * 40)

    if results:
        print(f"Training manifest:   {results.get('train', 0)} entries")
        print(f"Validation manifest: {results.get('val', 0)} entries")
        print(f"Combined manifest:   {results.get('all', 0)} entries")
    else:
        print("❌ No manifests created")

    # Validate if requested
    if args.validate and results:
        print("\n🔍 Validating manifests...")
        validation = creator.validate_manifests()

        for manifest, result in validation.items():
            if result.get('exists'):
                if 'error' in result:
                    print(f"❌ {manifest}: {result['error']}")
                else:
                    print(f"✅ {manifest}: {result['valid_entries']}/{result['entries']} valid")
            else:
                print(f"❌ {manifest}: File not found")

    print("\n✅ Manifest creation complete!")
    print(f"📂 Check your manifests in: {args.output_dir}")

    # Show usage example
    print("\n🚀 Usage example:")
    print(f"python scripts/real_fine_tune.py --train_manifest {args.output_dir}/train_manifest.json --val_manifest {args.output_dir}/val_manifest.json")

if __name__ == "__main__":
    main()