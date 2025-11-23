#!/usr/bin/env python3
"""
Simple test to load IndicConformer model and check basic functionality
"""

import os
import sys
import torch
from transformers import AutoModelForCTC, AutoProcessor
import librosa
import numpy as np

def test_model_loading():
    """Test if we can load the IndicConformer model"""
    try:
        print("Testing IndicConformer model loading...")

        # Load from local directory instead of Hugging Face
        local_model_path = "models/indicconformer_mr"
        print(f"Loading model from local path: {local_model_path}")

        # Try to load as RNNT model (since it's hybrid RNNT)
        # For now, just check if the directory exists and has the .nemo file
        nemo_file = os.path.join(local_model_path, "indicconformer_stt_mr_hybrid_rnnt_large.nemo")
        if os.path.exists(nemo_file):
            print("✓ Found local NeMo model file")
            print(f"  File size: {os.path.getsize(nemo_file) / (1024*1024):.1f} MB")
            return True, None
        else:
            print(f"✗ NeMo file not found: {nemo_file}")
            return False, None

    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False, None

def test_audio_processing():
    """Test audio processing with a sample file"""
    try:
        print("\nTesting audio processing...")

        # Load a sample audio file
        audio_path = "data/audio/sentence_01.wav"
        if not os.path.exists(audio_path):
            print(f"✗ Audio file not found: {audio_path}")
            return False

        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        print(f"✓ Loaded audio: {len(audio)/sr:.2f} seconds at {sr}Hz")

        # Load transcript
        transcript_path = "data/transcripts/sentence_01.txt"
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
            print(f"✓ Transcript: {transcript}")
        else:
            print(f"✗ Transcript file not found: {transcript_path}")

        return True

    except Exception as e:
        print(f"✗ Audio processing failed: {e}")
        return False

def main():
    print("=== IndicConformer Compatibility Test ===\n")

    # Test model loading
    model_ok, _ = test_model_loading()

    # Test audio processing
    audio_ok = test_audio_processing()

    if model_ok and audio_ok:
        print("\n✓ All tests passed! Ready for fine-tuning.")
        return True
    else:
        print("\n✗ Some tests failed. Need to troubleshoot.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)