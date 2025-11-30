#!/usr/bin/env python3
"""
Test Marathi ASR Model on Konkani Data
Compare with our HuggingFace baseline
"""

import os
import sys
import torch
import librosa
import numpy as np
from transformers import pipeline
from jiwer import wer

def load_audio(file_path, target_sr=16000):
    """Load and preprocess audio file"""
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio, sr

def test_marathi_model():
    """Test the Marathi ASR model on Konkani data"""

    print("🤖 Testing Marathi ASR Model on Konkani Data...")

    # Initialize the Marathi ASR pipeline
    asr = pipeline(
        'automatic-speech-recognition',
        model='hriteshMaikap/marathi-asr-model',
        device='cpu'  # Use CPU since we don't have CUDA
    )

    print("✅ Marathi ASR model loaded successfully!")

    # Test on our Konkani data
    test_files = [
        ("data/audio/sentence_01.wav", "चल रॅ भोपळा टुनुक टुनुक"),
        ("data/audio/sentence_02.wav", "म्हाका देव भेटला"),
        ("data/audio/sentence_03.wav", "देवान काय दिलं"),
        ("data/audio/sentence_04.wav", "देवान दिलं सोन्याचं भोपळा"),
        ("data/audio/sentence_05.wav", "आनी ताका भोपळा घेवन")
    ]

    total_wer = 0
    num_tests = 0

    for audio_file, reference_text in test_files:
        if os.path.exists(audio_file):
            print(f"\n🎵 Testing: {audio_file}")

            # Load audio
            audio, sr = load_audio(audio_file)
            print(f"Audio loaded: {len(audio)/sr:.2f} seconds at {sr}Hz")

            # Transcribe
            result = asr(audio)
            transcription = result['text'].strip()
            print(f"Reference: {reference_text}")
            print(f"Transcription: {transcription}")

            # Calculate WER
            word_error_rate = wer(reference_text, transcription)
            print(f"WER: {word_error_rate:.2%}")

            total_wer += word_error_rate
            num_tests += 1
        else:
            print(f"⚠️  Audio file not found: {audio_file}")

    if num_tests > 0:
        avg_wer = total_wer / num_tests
        print(f"\n📊 Average WER on Konkani data: {avg_wer:.2%}")

        # Compare with our HuggingFace baseline (WER ~83%)
        baseline_wer = 0.83
        improvement = baseline_wer - avg_wer
        print(f"Baseline WER (HuggingFace): {baseline_wer:.2%}")
        print(f"Improvement: {improvement:.2%}")

        if improvement > 0:
            print("✅ Marathi model performs better on Konkani!")
        else:
            print("❌ Marathi model performs worse than baseline")
    else:
        print("❌ No test files found")

if __name__ == "__main__":
    test_marathi_model()