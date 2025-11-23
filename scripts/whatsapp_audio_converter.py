#!/usr/bin/env python3
"""
WhatsApp Audio Converter: Convert .opus files to .wav for ASR training
Converts crowdsourced WhatsApp audio recordings to WAV format
"""

import os
import sys
import glob
import logging
from pathlib import Path
from typing import List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WhatsAppAudioConverter:
    """Convert WhatsApp .opus audio files to .wav format for ASR training"""

    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize converter

        Args:
            input_dir: Directory containing .opus files
            output_dir: Directory to save .wav files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check if required libraries are available
        self._check_dependencies()

    def _check_dependencies(self):
        """Check if required audio processing libraries are installed"""
        # Set ffmpeg path for pydub (Windows compatibility)
        ffmpeg_path = r"c:\Users\Milind Kopikare\Code\amchi_konkani\konkani_asr\tools\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
        os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ.get("PATH", "")
        
        try:
            import pydub
            from pydub import AudioSegment
            # Set ffmpeg converter explicitly
            AudioSegment.converter = ffmpeg_path
            logger.info("✓ pydub library available")
            logger.info(f"✓ ffmpeg configured: {ffmpeg_path}")
            self.AudioSegment = AudioSegment
        except ImportError:
            logger.error("❌ pydub library not found. Install with: pip install pydub")
            logger.error("Also install ffmpeg: conda install ffmpeg or apt-get install ffmpeg")
            sys.exit(1)

    def find_opus_files(self) -> List[Path]:
        """Find all .opus files in input directory"""
        opus_pattern = "*.opus"
        opus_files = list(self.input_dir.glob(opus_pattern))

        # Also check subdirectories
        for subdir in self.input_dir.rglob("**/"):
            if subdir != self.input_dir:
                opus_files.extend(subdir.glob(opus_pattern))

        return sorted(opus_files)

    def convert_single_file(self, opus_file: Path) -> Tuple[bool, str]:
        """
        Convert a single .opus file to .wav

        Args:
            opus_file: Path to .opus file

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Generate output filename
            relative_path = opus_file.relative_to(self.input_dir)
            wav_filename = relative_path.with_suffix('.wav')
            output_file = self.output_dir / wav_filename

            # Create output subdirectory if needed
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Load and convert audio (let pydub auto-detect format)
            logger.info(f"Converting: {opus_file} → {output_file}")
            audio = self.AudioSegment.from_file(str(opus_file))

            # Export as WAV with 16kHz sample rate (good for ASR)
            audio.export(str(output_file), format="wav", parameters=["-ar", "16000", "-ac", "1"])

            # Verify conversion
            if output_file.exists():
                file_size = output_file.stat().st_size
                duration_ms = len(audio)
                duration_sec = duration_ms / 1000

                logger.info(".1f")
                return True, f"Converted {duration_sec:.1f}s audio"
            else:
                return False, "Output file not created"

        except Exception as e:
            error_msg = f"Failed to convert {opus_file}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

    def convert_all_files(self) -> Tuple[int, int, List[str]]:
        """
        Convert all .opus files in input directory

        Returns:
            Tuple of (successful_conversions, total_files, error_messages)
        """
        opus_files = self.find_opus_files()

        if not opus_files:
            logger.warning(f"❌ No .opus files found in {self.input_dir}")
            return 0, 0, ["No .opus files found"]

        logger.info(f"📁 Found {len(opus_files)} .opus files to convert")
        logger.info(f"📂 Output directory: {self.output_dir}")

        successful = 0
        errors = []

        for i, opus_file in enumerate(opus_files, 1):
            logger.info(f"🔄 Converting {i}/{len(opus_files)}: {opus_file.name}")

            success, message = self.convert_single_file(opus_file)

            if success:
                successful += 1
                logger.info(f"✅ {message}")
            else:
                errors.append(message)

        return successful, len(opus_files), errors

    def validate_conversions(self) -> Tuple[int, int, List[str]]:
        """
        Validate that converted files are valid audio files

        Returns:
            Tuple of (valid_files, total_converted, validation_errors)
        """
        logger.info("🔍 Validating converted audio files...")

        try:
            import librosa
        except ImportError:
            logger.warning("⚠️  librosa not available for validation")
            return 0, 0, ["librosa not available for validation"]

        wav_files = list(self.output_dir.rglob("*.wav"))
        valid_files = 0
        validation_errors = []

        for wav_file in wav_files:
            try:
                # Try to load the audio file
                audio, sr = librosa.load(str(wav_file), sr=None, duration=1.0)  # Load first second

                if len(audio) > 0 and sr > 0:
                    valid_files += 1
                else:
                    validation_errors.append(f"Empty or invalid audio: {wav_file}")

            except Exception as e:
                validation_errors.append(f"Failed to load {wav_file}: {str(e)}")

        return valid_files, len(wav_files), validation_errors

def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Convert WhatsApp .opus files to .wav for ASR training")
    parser.add_argument("input_dir", help="Directory containing .opus files")
    parser.add_argument("output_dir", help="Directory to save .wav files")
    parser.add_argument("--validate", action="store_true", help="Validate converted files")

    args = parser.parse_args()

    print("🎵 WhatsApp Audio Converter")
    print("=" * 40)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print()

    # Initialize converter
    converter = WhatsAppAudioConverter(args.input_dir, args.output_dir)

    # Convert files
    successful, total, errors = converter.convert_all_files()

    print("\n" + "=" * 40)
    print("📊 CONVERSION RESULTS")
    print("=" * 40)
    print(f"Total files: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(errors)}")

    if errors:
        print("\n❌ ERRORS:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  • {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    # Validate if requested
    if args.validate and successful > 0:
        print("\n🔍 Validating conversions...")
        valid, total_converted, val_errors = converter.validate_conversions()

        print(f"Valid files: {valid}/{total_converted}")
        if val_errors:
            print("Validation errors:")
            for error in val_errors[:3]:
                print(f"  • {error}")

    print("\n✅ Conversion complete!")
    print(f"📂 Check your .wav files in: {args.output_dir}")

if __name__ == "__main__":
    main()