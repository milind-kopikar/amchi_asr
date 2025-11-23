#!/usr/bin/env python3
"""
Inference script for Konkani ASR model
Transcribes audio files using the fine-tuned model
"""

import os
import sys
import argparse
import torch
from pathlib import Path
import librosa
import soundfile as sf
from nemo.collections.asr.models import EncDecHybridRNNTCTCModel
from nemo.utils import logging

def load_model(model_path):
    """Load the ASR model"""
    print(f"🔄 Loading model from {model_path}...")
    try:
        model = EncDecHybridRNNTCTCModel.restore_from(model_path)
        model.freeze()
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        print("✅ Model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def preprocess_audio(audio_path, target_sr=16000):
    """Preprocess audio file for inference"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=target_sr)

        # Normalize
        audio = librosa.util.normalize(audio)

        # Ensure minimum length (0.1 seconds)
        min_samples = int(0.1 * target_sr)
        if len(audio) < min_samples:
            # Pad with zeros
            padding = min_samples - len(audio)
            audio = torch.nn.functional.pad(torch.tensor(audio), (0, padding)).numpy()

        return audio, sr

    except Exception as e:
        print(f"❌ Failed to preprocess {audio_path}: {e}")
        return None, None

def transcribe_audio(model, audio_path):
    """Transcribe a single audio file"""
    # Preprocess audio
    audio, sr = preprocess_audio(audio_path)
    if audio is None:
        return None

    try:
        # Convert to tensor
        audio_tensor = torch.tensor(audio).unsqueeze(0)  # Add batch dimension

        # Move to device
        device = next(model.parameters()).device
        audio_tensor = audio_tensor.to(device)

        # Transcribe
        with torch.no_grad():
            transcriptions = model.transcribe([audio_tensor], batch_size=1)

        transcription = transcriptions[0][0] if transcriptions else ""
        return transcription

    except Exception as e:
        print(f"❌ Failed to transcribe {audio_path}: {e}")
        return None

def transcribe_batch(model, audio_files, output_file=None):
    """Transcribe multiple audio files"""
    results = []

    print(f"🎵 Transcribing {len(audio_files)} audio files...")

    for i, audio_path in enumerate(audio_files, 1):
        print(f"  [{i}/{len(audio_files)}] Processing: {Path(audio_path).name}")

        transcription = transcribe_audio(model, audio_path)

        if transcription is not None:
            result = {
                'audio_file': str(audio_path),
                'transcription': transcription
            }
            results.append(result)
            print(f"    📝 Transcription: {transcription}")
        else:
            print(f"    ❌ Failed to transcribe")

    # Save results if output file specified
    if output_file and results:
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 Results saved to {output_file}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using Konkani ASR model")
    parser.add_argument("--model_path", required=True, help="Path to the trained model (.nemo file)")
    parser.add_argument("--audio_file", help="Single audio file to transcribe")
    parser.add_argument("--audio_dir", help="Directory containing audio files to transcribe")
    parser.add_argument("--output_file", help="Output file for results (JSON format)")
    parser.add_argument("--audio_extensions", default="wav,flac,mp3,m4a", help="Audio file extensions (comma-separated)")

    args = parser.parse_args()

    # Validate arguments
    if not args.audio_file and not args.audio_dir:
        print("❌ Must specify either --audio_file or --audio_dir")
        sys.exit(1)

    if args.audio_file and args.audio_dir:
        print("❌ Cannot specify both --audio_file and --audio_dir")
        sys.exit(1)

    # Check model file
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        sys.exit(1)

    # Get audio files
    audio_files = []
    if args.audio_file:
        if not os.path.exists(args.audio_file):
            print(f"❌ Audio file not found: {args.audio_file}")
            sys.exit(1)
        audio_files = [args.audio_file]
    else:
        extensions = set(args.audio_extensions.lower().split(','))
        audio_dir = Path(args.audio_dir)

        if not audio_dir.exists():
            print(f"❌ Audio directory not found: {args.audio_dir}")
            sys.exit(1)

        for ext in extensions:
            audio_files.extend(audio_dir.glob(f"*.{ext}"))

        if not audio_files:
            print(f"❌ No audio files found in {args.audio_dir} with extensions: {args.audio_extensions}")
            sys.exit(1)

    print("🚀 Konkani ASR Inference")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Audio files: {len(audio_files)}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Load model
    model = load_model(args.model_path)
    if model is None:
        sys.exit(1)

    # Transcribe
    results = transcribe_batch(model, audio_files, args.output_file)

    print("\n" + "=" * 50)
    print("🎉 Inference completed!")
    print("=" * 50)

    print(f"\n📊 Summary:")
    print(f"- Files processed: {len(results)}")
    print(f"- Successful transcriptions: {len([r for r in results if r['transcription']])}")

    if args.output_file:
        print(f"- Results saved to: {args.output_file}")

    # Print sample results
    if results:
        print(f"\n📝 Sample transcriptions:")
        for i, result in enumerate(results[:3]):  # Show first 3
            audio_name = Path(result['audio_file']).name
            transcription = result['transcription'][:100] + "..." if len(result['transcription']) > 100 else result['transcription']
            print(f"  {i+1}. {audio_name}: {transcription}")

if __name__ == "__main__":
    # Set logging level
    logging.setLevel(logging.WARNING)

    main()