#!/usr/bin/env python3
"""
Check audio file properties for Konkani ASR requirements
Verifies sample rate, channels, bit depth, and duration
"""

import os
from pathlib import Path

def check_audio_properties():
    """Check WAV file properties using available libraries"""

    print("🎵 Checking WAV File Properties for ASR")
    print("=" * 45)

    audio_dir = Path("data/audio")
    wav_files = list(audio_dir.glob("*.wav"))

    if not wav_files:
        print("❌ No WAV files found in data/audio/")
        return False

    print(f"Found {len(wav_files)} WAV files:")
    for wav_file in wav_files:
        print(f"  - {wav_file.name}")

    # Try different methods to check audio properties
    success = False

    # Method 1: Try soundfile (most reliable)
    try:
        import soundfile as sf
        print("\n🔍 Using soundfile library:")
        print("-" * 30)
        success = check_with_soundfile(wav_files)
    except ImportError:
        print("⚠️ soundfile not available, trying alternatives...")

        # Method 2: Try librosa
        try:
            import librosa
            print("\n🔍 Using librosa library:")
            print("-" * 30)
            success = check_with_librosa(wav_files)
        except ImportError:
            print("⚠️ librosa not available, trying basic method...")

            # Method 3: Basic file size check
            success = check_basic_properties(wav_files)

    return success

def check_with_soundfile(wav_files):
    """Check audio properties using soundfile library"""

    all_good = True

    for wav_file in wav_files:
        try:
            import soundfile as sf
            info = sf.info(str(wav_file))

            # ASR requirements
            asr_requirements = {
                "Sample Rate": (info.samplerate, 16000, "Hz"),
                "Channels": (info.channels, 1, ""),
                "Bit Depth": (int(info.subtype.split('_')[1]) if '_' in info.subtype else 16, 16, "bit"),
                "Format": (info.format, "WAV", "")
            }

            print(f"\n📁 {wav_file.name}:")
            print(f"   Duration: {info.duration:.1f} seconds")
            print(f"   File size: {os.path.getsize(wav_file):,} bytes")

            file_ok = True
            for prop_name, (actual, required, unit) in asr_requirements.items():
                status = "✅" if actual == required else "❌"
                if actual != required:
                    file_ok = False
                    all_good = False
                print(f"   {status} {prop_name}: {actual}{unit} (required: {required}{unit})")

            if not file_ok:
                print("   ⚠️ This file needs conversion!")

        except Exception as e:
            print(f"❌ Error checking {wav_file.name}: {e}")
            all_good = False

    return all_good

def check_with_librosa(wav_files):
    """Check audio properties using librosa library"""

    all_good = True

    for wav_file in wav_files:
        try:
            import librosa
            import numpy as np

            # Load audio to get properties
            audio, sr = librosa.load(str(wav_file), sr=None)

            # Estimate bit depth from file size (rough approximation)
            file_size = os.path.getsize(wav_file)
            duration = len(audio) / sr
            estimated_bitrate = (file_size * 8) / duration  # bits per second
            estimated_bit_depth = int(estimated_bitrate / (sr * (1 if audio.ndim == 1 else 2)))

            print(f"\n📁 {wav_file.name}:")
            print(f"   Duration: {duration:.1f} seconds")
            print(f"   File size: {file_size:,} bytes")
            print(f"   Estimated bit depth: {estimated_bit_depth} bit")

            # Check requirements
            checks = [
                ("Sample Rate", sr, 16000, "Hz"),
                ("Channels", 1 if audio.ndim == 1 else audio.shape[1], 1, ""),
                ("Bit Depth", estimated_bit_depth, 16, "bit")
            ]

            file_ok = True
            for prop_name, actual, required, unit in checks:
                status = "✅" if actual == required else "❌"
                if actual != required:
                    file_ok = False
                    all_good = False
                print(f"   {status} {prop_name}: {actual}{unit} (required: {required}{unit})")

            if not file_ok:
                print("   ⚠️ This file needs conversion!")

        except Exception as e:
            print(f"❌ Error checking {wav_file.name}: {e}")
            all_good = False

    return all_good

def check_basic_properties(wav_files):
    """Basic file property check when audio libraries unavailable"""

    print("\n🔍 Basic file properties (limited info):")
    print("-" * 40)

    for wav_file in wav_files:
        file_size = os.path.getsize(wav_file)
        print(f"\n📁 {wav_file.name}:")
        print(f"   File size: {file_size:,} bytes")

        # Rough size estimates for 16kHz mono 16-bit WAV
        # 16kHz * 16-bit * 1 channel = 256kbps = ~32KB per second
        estimated_duration = file_size / (16000 * 2 * 1)  # rough estimate
        print(f"   Estimated duration: {estimated_duration:.1f} seconds (at 16kHz mono 16-bit)")

        # Flag suspicious sizes
        if file_size < 10000:  # Less than ~0.5 seconds
            print("   ⚠️ File seems very small - may be wrong format or too short")
        elif file_size > 1000000:  # More than ~30 seconds
            print("   ⚠️ File seems very large - check sample rate")

    print("\n⚠️ Cannot verify sample rate/channels without audio libraries")
    print("Install: pip install soundfile")
    print("Then re-run: python check_audio_properties.py")

    return False

def provide_recommendations():
    """Provide recommendations based on findings"""

    print("\n" + "=" * 50)
    print("🎯 ASR REQUIREMENTS REMINDER:")
    print("=" * 50)
    print("✅ Sample Rate: 16000 Hz")
    print("✅ Channels: Mono (1 channel)")
    print("✅ Bit Depth: 16-bit")
    print("✅ Format: WAV")

    print("\n💡 IF FILES NEED CONVERSION:")
    print("1. Go to: https://cloudconvert.com/wav-converter")
    print("2. Set: Sample Rate=16000Hz, Channels=Mono, Bit Depth=16-bit")
    print("3. Re-upload and convert your WAV files")

    print("\n🚀 IF FILES ARE GOOD:")
    print("Run: python scripts/prepare_data.py --audio_dir data/audio --transcript_dir data/transcripts --output_dir data/test_run")

if __name__ == "__main__":
    success = check_audio_properties()
    provide_recommendations()

    if success:
        print("\n🎉 All files meet ASR requirements! Ready for training.")
    else:
        print("\n⚠️ Some files may need conversion. Check the details above.")