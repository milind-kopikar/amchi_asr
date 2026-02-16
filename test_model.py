#!/usr/bin/env python3
"""
Test the fine-tuned Konkani ASR model
"""

import os
import torch
from transformers import Wav2Vec2BertProcessor, Wav2Vec2BertForCTC, pipeline
import librosa
import subprocess
import tempfile

def load_audio_with_fallback(audio_path, sample_rate=16000):
    """Load audio with FFmpeg fallback for M4A files"""
    try:
        audio_array, _ = librosa.load(audio_path, sr=sample_rate)
        return audio_array
    except Exception as e:
        print(f"Librosa failed, trying FFmpeg conversion...")
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_wav_path = temp_file.name

        # Convert M4A to WAV using ffmpeg
        try:
            subprocess.run([
                'c:/Users/Milind Kopikare/Code/amchi_konkani/konkani_asr/ffmpeg/ffmpeg-8.0.1-essentials_build/bin/ffmpeg.exe', '-i', audio_path,
                '-acodec', 'pcm_s16le', '-ar', str(sample_rate), temp_wav_path,
                '-y', '-loglevel', 'quiet'
            ], check=True)

            # Load the converted WAV file
            audio_array, _ = librosa.load(temp_wav_path, sr=sample_rate)

            # Clean up temp file
            os.unlink(temp_wav_path)
            return audio_array

        except subprocess.CalledProcessError as ffmpeg_error:
            print(f"FFmpeg conversion failed: {ffmpeg_error}")
            return None

def test_model():
    print("🧪 Testing Fine-tuned Konkani ASR Model")
    print("=" * 50)

    # Load the fine-tuned model
    model_path = "D:/konkani_asr_models/huggingface_konkani/checkpoint-5"
    print(f"Loading model from: {model_path}")

    try:
        processor = Wav2Vec2BertProcessor.from_pretrained(model_path)
        model = Wav2Vec2BertForCTC.from_pretrained(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Create ASR pipeline
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=-1  # CPU
    )

    # Test on the training sample
    test_audio = "data/audio/sentence_06.m4a"  # This was our training sample
    expected_text = "पाव वाट दाण्टुनु वत्ता म्हण्तना तिका एकु सिंहु मेऴ्ळो!"  # From the manifest

    print(f"\n🎵 Testing on: {test_audio}")
    print(f"Expected text: {expected_text}")

    try:
        # Load audio manually first
        audio_array = load_audio_with_fallback(test_audio)
        if audio_array is None:
            print("❌ Failed to load audio")
            return

        # Transcribe using audio array directly
        result = asr(audio_array)
        predicted_text = result["text"]

        print(f"Predicted text: {predicted_text}")
        print(f"Match: {'✅' if predicted_text.strip() == expected_text.strip() else '❌'}")

    except Exception as e:
        print(f"❌ Transcription failed: {e}")

    print("\n🎉 Pipeline test complete!")

if __name__ == "__main__":
    test_model()