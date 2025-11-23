#!/usr/bin/env python3
"""
Audio-Text Corpus Creator: Split story audio and text into sentence-level pairs
Creates training data for ASR fine-tuning from long-form audio recordings
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Tuple, Dict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioTextCorpusCreator:
    """Create sentence-level audio-text pairs from story recordings"""

    def __init__(self, audio_file: str, text_file: str, output_dir: str):
        """
        Initialize corpus creator

        Args:
            audio_file: Path to the main WAV audio file
            text_file: Path to the corresponding text file
            output_dir: Directory to save sentence pairs
        """
        self.audio_file = Path(audio_file)
        self.text_file = Path(text_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check dependencies
        self._check_dependencies()

    def _check_dependencies(self):
        """Check if required libraries are available"""
        # Set ffmpeg path for pydub (Windows compatibility)
        ffmpeg_path = r"c:\Users\Milind Kopikare\Code\amchi_konkani\konkani_asr\tools\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
        os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ.get("PATH", "")

        try:
            import pydub
            from pydub import AudioSegment
            from pydub.silence import split_on_silence
            # Set ffmpeg converter explicitly
            AudioSegment.converter = ffmpeg_path
            logger.info("✓ pydub library available")
            logger.info(f"✓ ffmpeg configured: {ffmpeg_path}")
            self.AudioSegment = AudioSegment
            self.split_on_silence = split_on_silence
        except ImportError:
            logger.error("❌ pydub library not found. Install with: pip install pydub")
            exit(1)

    def split_text_into_sentences(self) -> List[str]:
        """
        Split the text file into individual sentences

        Returns:
            List of sentence strings
        """
        logger.info(f"Reading text file: {self.text_file}")

        with open(self.text_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # Split on Devanagari danda (।) and exclamation marks
        # Also handle line breaks as potential sentence boundaries
        sentences = re.split(r'[।!\n]+', content)

        # Clean up sentences (remove extra whitespace, empty strings)
        sentences = [s.strip() for s in sentences if s.strip()]

        logger.info(f"Found {len(sentences)} sentences in text")
        for i, sentence in enumerate(sentences, 1):
            logger.info(f"  {i}: {sentence[:50]}...")

        return sentences

    def split_audio_into_segments(self, min_silence_len: int = 500, silence_thresh: int = -40):
        """
        Split audio into segments based on silence detection

        Args:
            min_silence_len: Minimum silence length in ms to consider as split point
            silence_thresh: Silence threshold in dBFS

        Returns:
            List of tuples: (audio_segment, duration_seconds)
        """
        logger.info(f"Loading audio file: {self.audio_file}")
        audio = self.AudioSegment.from_wav(str(self.audio_file))

        logger.info(f"Audio duration: {len(audio)/1000:.1f}s")
        logger.info(f"Detecting silence with min_silence={min_silence_len}ms, threshold={silence_thresh}dBFS")

        # Split on silence
        segments = self.split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=200  # Keep 200ms of silence at the edges
        )

        logger.info(f"Found {len(segments)} audio segments")

        # Filter out very short segments (likely noise)
        filtered_segments = []
        for i, segment in enumerate(segments):
            duration = len(segment) / 1000
            if duration >= 1.0:  # Keep segments 1+ seconds
                filtered_segments.append((segment, duration))
                logger.info(f"  Segment {i+1}: {duration:.1f}s")
            else:
                logger.info(f"  Segment {i+1}: {duration:.1f}s (too short, skipped)")

        return filtered_segments

    def create_corpus(self, min_silence_len: int = 500, silence_thresh: int = -40) -> Tuple[int, List[str]]:
        """
        Create the complete audio-text corpus

        Args:
            min_silence_len: Minimum silence length for audio splitting
            silence_thresh: Silence threshold for audio splitting

        Returns:
            Tuple of (successful_pairs, error_messages)
        """
        # Split text into sentences
        sentences = self.split_text_into_sentences()

        # Split audio into segments
        audio_segments = self.split_audio_into_segments(min_silence_len, silence_thresh)

        # Check if we have matching counts
        if len(sentences) != len(audio_segments):
            warning_msg = f"Mismatch: {len(sentences)} text sentences vs {len(audio_segments)} audio segments"
            logger.warning(f"⚠️  {warning_msg}")

            # Try to match them up anyway (take minimum)
            num_pairs = min(len(sentences), len(audio_segments))
            logger.info(f"Creating {num_pairs} pairs (minimum of text/audio)")
        else:
            num_pairs = len(sentences)
            logger.info(f"✅ Perfect match: {num_pairs} sentence-audio pairs")

        successful = 0
        errors = []

        # Create audio-text pairs
        for i in range(num_pairs):
            try:
                sentence = sentences[i]
                audio_segment, duration = audio_segments[i]

                # Create filenames
                base_name = f"story0_milind_sentence{i+1}"
                audio_filename = f"{base_name}.wav"
                text_filename = f"{base_name}.txt"

                audio_path = self.output_dir / audio_filename
                text_path = self.output_dir / text_filename

                # Save audio segment
                audio_segment.export(str(audio_path), format="wav", parameters=["-ar", "16000", "-ac", "1"])

                # Save text sentence
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(sentence)

                logger.info(f"✅ Created pair {i+1}: {duration:.1f}s audio + text")
                successful += 1

            except Exception as e:
                error_msg = f"Failed to create pair {i+1}: {str(e)}"
                logger.error(f"❌ {error_msg}")
                errors.append(error_msg)

        return successful, errors

    def validate_corpus(self) -> Tuple[int, int, List[str]]:
        """
        Validate the created corpus files

        Returns:
            Tuple of (valid_pairs, total_pairs, validation_errors)
        """
        logger.info("🔍 Validating created corpus...")

        wav_files = list(self.output_dir.glob("story0_milind_sentence*.wav"))
        txt_files = list(self.output_dir.glob("story0_milind_sentence*.txt"))

        total_pairs = len(wav_files)
        valid_pairs = 0
        validation_errors = []

        # Check each pair
        for i in range(1, total_pairs + 1):
            wav_path = self.output_dir / f"story0_milind_sentence{i}.wav"
            txt_path = self.output_dir / f"story0_milind_sentence{i}.txt"

            if not wav_path.exists():
                validation_errors.append(f"Missing audio: {wav_path.name}")
                continue

            if not txt_path.exists():
                validation_errors.append(f"Missing text: {txt_path.name}")
                continue

            # Check audio file
            try:
                audio = self.AudioSegment.from_wav(str(wav_path))
                if len(audio) < 500:  # Less than 0.5 seconds
                    validation_errors.append(f"Audio too short: {wav_path.name}")
                    continue
            except Exception as e:
                validation_errors.append(f"Invalid audio file {wav_path.name}: {str(e)}")
                continue

            # Check text file
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if not text:
                        validation_errors.append(f"Empty text file: {txt_path.name}")
                        continue
            except Exception as e:
                validation_errors.append(f"Invalid text file {txt_path.name}: {str(e)}")
                continue

            valid_pairs += 1

        return valid_pairs, total_pairs, validation_errors

def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Create sentence-level audio-text corpus from story recordings")
    parser.add_argument("audio_file", help="Path to the WAV audio file")
    parser.add_argument("text_file", help="Path to the corresponding text file")
    parser.add_argument("output_dir", help="Directory to save sentence pairs")
    parser.add_argument("--min-silence", type=int, default=500, help="Minimum silence length in ms (default: 500)")
    parser.add_argument("--silence-thresh", type=int, default=-40, help="Silence threshold in dBFS (default: -40)")
    parser.add_argument("--validate", action="store_true", help="Validate created corpus")

    args = parser.parse_args()

    print("🎵 Audio-Text Corpus Creator")
    print("=" * 50)
    print(f"Audio:     {args.audio_file}")
    print(f"Text:      {args.text_file}")
    print(f"Output:    {args.output_dir}")
    print(f"Silence:   {args.min_silence}ms @ {args.silence_thresh}dBFS")
    print()

    # Create corpus
    creator = AudioTextCorpusCreator(args.audio_file, args.text_file, args.output_dir)

    successful, errors = creator.create_corpus(args.min_silence, args.silence_thresh)

    print("\n" + "=" * 50)
    print("📊 CORPUS CREATION RESULTS")
    print("=" * 50)
    print(f"Successful pairs: {successful}")

    if errors:
        print("\n❌ ERRORS:")
        for error in errors[:5]:
            print(f"  • {error}")

    # Validate if requested
    if args.validate and successful > 0:
        print("\n🔍 Validating corpus...")
        valid, total, val_errors = creator.validate_corpus()

        print(f"Valid pairs: {valid}/{total}")
        if val_errors:
            print("Validation errors:")
            for error in val_errors[:3]:
                print(f"  • {error}")

    print("\n✅ Corpus creation complete!")
    print(f"📂 Check your audio-text pairs in: {args.output_dir}")

if __name__ == "__main__":
    main()