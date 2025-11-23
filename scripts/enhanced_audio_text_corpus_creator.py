#!/usr/bin/env python3
"""
Enhanced Audio-Text Corpus Creator with Manual Verification
Implements the improved algorithm for accurate sentence-level splitting
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedAudioTextCorpusCreator:
    """Enhanced corpus creator with manual verification and iterative refinement"""

    def __init__(self, audio_file: str, text_file: str, output_dir: str):
        """
        Initialize enhanced corpus creator

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
        try:
            import pydub
            from pydub import AudioSegment
            from pydub.silence import split_on_silence
            # Set ffmpeg path
            ffmpeg_path = r"c:\Users\Milind Kopikare\Code\amchi_konkani\konkani_asr\tools\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
            os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ.get("PATH", "")
            AudioSegment.converter = ffmpeg_path
            logger.info("✓ pydub library available")
            self.AudioSegment = AudioSegment
            self.split_on_silence = split_on_silence
        except ImportError:
            logger.error("❌ pydub library not found. Install with: pip install pydub")
            exit(1)

    def split_text_into_sentences(self) -> List[str]:
        """
        Split the text file into individual sentences using Devanagari danda (।)

        Returns:
            List of sentence strings
        """
        logger.info(f"Reading and splitting text file: {self.text_file}")

        with open(self.text_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # Split on Devanagari danda (।) and exclamation marks, but preserve line breaks
        # First, split on line breaks to get paragraphs
        paragraphs = content.split('\n')

        sentences = []
        for paragraph in paragraphs:
            if paragraph.strip():
                # Split each paragraph on danda and exclamation
                para_sentences = re.split(r'[।!]+', paragraph.strip())
                # Clean up and filter empty strings
                para_sentences = [s.strip() for s in para_sentences if s.strip()]
                sentences.extend(para_sentences)

        logger.info(f"Found {len(sentences)} sentences in text:")
        for i, sentence in enumerate(sentences, 1):
            logger.info(f"  {i}: {sentence}")
            # Create individual text files
            text_filename = f"story0_sentence{i}.txt"
            text_path = self.output_dir / text_filename
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(sentence)

        return sentences

    def split_audio_into_segments(self, min_silence_len: int = 1500, silence_thresh: int = -35) -> List[Tuple['AudioSegment', float]]:
        """
        Split audio into segments based on silence detection with longer pauses

        Args:
            min_silence_len: Minimum silence length in ms (increased for sentence pauses)
            silence_thresh: Silence threshold in dBFS

        Returns:
            List of tuples: (audio_segment, duration_seconds)
        """
        logger.info(f"Loading audio file: {self.audio_file}")
        audio = self.AudioSegment.from_wav(str(self.audio_file))

        logger.info(f"Audio duration: {len(audio)/1000:.1f}s")
        logger.info(f"Detecting sentence boundaries with min_silence={min_silence_len}ms, threshold={silence_thresh}dBFS")

        # Split on longer silence (sentence pauses are typically 1.5+ seconds)
        segments = self.split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=300  # Keep 300ms of silence at edges
        )

        logger.info(f"Found {len(segments)} potential sentence segments")

        # Filter out very short segments (likely noise or fragments)
        filtered_segments = []
        for i, segment in enumerate(segments):
            duration = len(segment) / 1000
            if duration >= 2.0:  # Keep segments 2+ seconds (reasonable sentence length)
                filtered_segments.append((segment, duration))
                logger.info(f"  Segment {i+1}: {duration:.1f}s")
            else:
                logger.info(f"  Segment {i+1}: {duration:.1f}s (too short, skipped)")

        return filtered_segments

    def create_initial_corpus(self, sentences: List[str], audio_segments: List[Tuple['AudioSegment', float]]) -> List[Dict[str, Any]]:
        """
        Create initial corpus with automatic alignment

        Returns:
            List of corpus entries with alignment info
        """
        corpus_entries = []

        # Try to match sentences to audio segments
        num_pairs = min(len(sentences), len(audio_segments))

        logger.info(f"Creating initial alignment for {num_pairs} pairs")

        for i in range(num_pairs):
            sentence = sentences[i]
            audio_segment, duration = audio_segments[i]

            entry = {
                'sentence_id': i + 1,
                'text': sentence,
                'audio_duration': duration,
                'text_word_count': len(sentence.split()),
                'estimated_wpm': len(sentence.split()) / (duration / 60),  # words per minute
                'audio_segment': audio_segment,
                'confidence': 'auto'  # Mark as automatically aligned
            }

            corpus_entries.append(entry)
            logger.info(f"  Pair {i+1}: {len(sentence.split())} words, {duration:.1f}s audio")

        return corpus_entries

    def save_audio_segments(self, corpus_entries: List[Dict[str, Any]], speaker_name: str = "milind"):
        """
        Save audio segments to individual WAV files

        Args:
            corpus_entries: List of corpus entries
            speaker_name: Name of the speaker
        """
        logger.info("Saving audio segments...")

        for entry in corpus_entries:
            sentence_id = entry['sentence_id']
            audio_segment = entry['audio_segment']

            # Create filename: story0_milind_sentence1.wav
            audio_filename = f"story0_{speaker_name}_sentence{sentence_id}.wav"
            audio_path = self.output_dir / audio_filename

            # Export as WAV
            audio_segment.export(str(audio_path), format="wav", parameters=["-ar", "16000", "-ac", "1"])
            logger.info(f"  Saved: {audio_filename} ({entry['audio_duration']:.1f}s)")

    def generate_alignment_report(self, corpus_entries: List[Dict[str, Any]]) -> str:
        """
        Generate a detailed alignment report for manual verification

        Returns:
            Report as string
        """
        report = []
        report.append("🎵 AUDIO-TEXT ALIGNMENT REPORT")
        report.append("=" * 50)
        report.append("")

        for entry in corpus_entries:
            sentence_id = entry['sentence_id']
            text = entry['text']
            duration = entry['audio_duration']
            word_count = entry['text_word_count']
            wpm = entry['estimated_wpm']

            report.append(f"📝 Sentence {sentence_id}")
            report.append(f"   Text: {text}")
            report.append(f"   Audio: {duration:.1f}s")
            report.append(f"   Words: {word_count}")
            report.append(f"   Speaking rate: {wpm:.1f} WPM")
            report.append("")

            # Flag potential issues
            if duration < 3.0:
                report.append("   ⚠️  WARNING: Very short audio for sentence")
            if duration > 15.0:
                report.append("   ⚠️  WARNING: Very long audio for sentence")
            if wpm < 80:
                report.append("   ⚠️  WARNING: Very slow speaking rate")
            if wpm > 200:
                report.append("   ⚠️  WARNING: Very fast speaking rate")

            report.append("")

        return "\n".join(report)

    def save_alignment_report(self, corpus_entries: List[Dict[str, Any]], report_file: str = "alignment_report.txt"):
        """
        Save the alignment report to a file
        """
        report = self.generate_alignment_report(corpus_entries)
        report_path = self.output_dir / report_file

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Alignment report saved: {report_path}")

    def create_corpus_with_verification(self, min_silence_len: int = 1500, silence_thresh: int = -35) -> List[Dict[str, Any]]:
        """
        Create corpus using the enhanced algorithm with verification

        Returns:
            List of verified corpus entries
        """
        # Step 1: Split text into sentences
        sentences = self.split_text_into_sentences()

        # Step 2: Split audio into segments
        audio_segments = self.split_audio_into_segments(min_silence_len, silence_thresh)

        # Step 3: Create initial alignment
        corpus_entries = self.create_initial_corpus(sentences, audio_segments)

        # Step 4: Save audio segments
        self.save_audio_segments(corpus_entries)

        # Step 5: Generate verification report
        self.save_alignment_report(corpus_entries)

        return corpus_entries

def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced audio-text corpus creator with manual verification")
    parser.add_argument("audio_file", help="Path to the WAV audio file")
    parser.add_argument("text_file", help="Path to the corresponding text file")
    parser.add_argument("output_dir", help="Directory to save sentence pairs")
    parser.add_argument("--min-silence", type=int, default=1500, help="Minimum silence length in ms (default: 1500)")
    parser.add_argument("--silence-thresh", type=int, default=-35, help="Silence threshold in dBFS (default: -35)")
    parser.add_argument("--speaker", type=str, default="milind", help="Speaker name for filenames")

    args = parser.parse_args()

    print("🎵 Enhanced Audio-Text Corpus Creator")
    print("=" * 55)
    print(f"Audio:     {args.audio_file}")
    print(f"Text:      {args.text_file}")
    print(f"Output:    {args.output_dir}")
    print(f"Speaker:   {args.speaker}")
    print(f"Silence:   {args.min_silence}ms @ {args.silence_thresh}dBFS")
    print()

    # Create enhanced corpus
    creator = EnhancedAudioTextCorpusCreator(args.audio_file, args.text_file, args.output_dir)
    corpus_entries = creator.create_corpus_with_verification(args.min_silence, args.silence_thresh)

    print("\n" + "=" * 55)
    print("📊 CORPUS CREATION RESULTS")
    print("=" * 55)
    print(f"Text sentences: {len([s for s in corpus_entries if 'text' in s])}")
    print(f"Audio segments: {len([s for s in corpus_entries if 'audio_segment' in s])}")
    print(f"Aligned pairs:  {len(corpus_entries)}")

    print("\n✅ Enhanced corpus creation complete!")
    print(f"📂 Check your files in: {args.output_dir}")
    print("📋 Review the alignment_report.txt for verification")

if __name__ == "__main__":
    main()