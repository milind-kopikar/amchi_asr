#!/usr/bin/env python3
"""
Convert M4A audio files to WAV format for Konkani ASR
Converts to 16kHz mono WAV as required for speech recognition
"""

import os
import glob
from pathlib import Path

def convert_m4a_to_wav():
    """Convert all M4A files in data/audio to WAV format"""

    try:
        import librosa
        import soundfile as sf
        import numpy as np

        print("🎵 Converting M4A files to WAV format for ASR")
        print("=" * 50)

        # Find all M4A files
        audio_dir = Path("data/audio")
        m4a_files = list(audio_dir.glob("*.m4a"))

        if not m4a_files:
            print("❌ No M4A files found in data/audio/")
            return False

        print(f"Found {len(m4a_files)} M4A files to convert:")
        for m4a_file in m4a_files:
            print(f"  - {m4a_file.name}")

        converted_count = 0

        for m4a_file in m4a_files:
            wav_file = m4a_file.with_suffix('.wav')

            try:
                print(f"\n🔄 Converting: {m4a_file.name} → {wav_file.name}")

                # Load audio with librosa (handles M4A format)
                audio, sr = librosa.load(str(m4a_file), sr=None)  # Keep original sample rate first

                print(f"   Original: {sr}Hz, {len(audio)} samples, {len(audio)/sr:.1f}s")

                # Resample to 16kHz if needed
                target_sr = 16000
                if sr != target_sr:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                    print(f"   Resampled to: {target_sr}Hz")

                # Ensure mono (convert stereo to mono if needed)
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)  # Average channels
                    print("   Converted to mono")

                # Normalize audio levels
                audio = librosa.util.normalize(audio)
                print("   Audio normalized")

                # Save as WAV
                sf.write(str(wav_file), audio, target_sr, subtype='PCM_16')
                file_size = os.path.getsize(wav_file)
                print(f"   ✅ Saved: {wav_file.name} ({file_size} bytes)")

                converted_count += 1

            except Exception as e:
                print(f"   ❌ Failed to convert {m4a_file.name}: {e}")

        print(f"\n{'='*50}")
        print(f"🎉 Conversion completed: {converted_count}/{len(m4a_files)} files converted")

        if converted_count > 0:
            print("\n📋 Converted files:")
            for m4a_file in m4a_files:
                wav_file = m4a_file.with_suffix('.wav')
                if wav_file.exists():
                    print(f"  ✅ {m4a_file.name} → {wav_file.name}")

        return converted_count > 0

    except ImportError as e:
        print(f"❌ Missing required library: {e}")
        print("Please install: pip install librosa soundfile numpy")
        return False

def verify_conversions():
    """Verify that converted WAV files meet ASR requirements"""

    print("\n🔍 Verifying converted files...")
    print("-" * 30)

    audio_dir = Path("data/audio")
    wav_files = list(audio_dir.glob("*.wav"))

    if not wav_files:
        print("❌ No WAV files found")
        return False

    all_good = True

    for wav_file in wav_files:
        try:
            import soundfile as sf
            info = sf.info(str(wav_file))

            # Check requirements
            checks = {
                "Sample Rate": (info.samplerate == 16000, f"{info.samplerate}Hz (need 16000Hz)"),
                "Channels": (info.channels == 1, f"{info.channels} channels (need mono)"),
                "Format": (info.subtype == 'PCM_16', f"{info.subtype} (need PCM_16)")
            }

            print(f"\n📁 {wav_file.name}:")
            for check_name, (passed, details) in checks.items():
                status = "✅" if passed else "❌"
                print(f"   {status} {check_name}: {details}")
                if not passed:
                    all_good = False

        except Exception as e:
            print(f"❌ Error checking {wav_file.name}: {e}")
            all_good = False

    if all_good:
        print("\n🎉 All files meet ASR requirements!")
        return True
    else:
        print("\n⚠️ Some files may need manual conversion")
        return False

if __name__ == "__main__":
    print("🎤 M4A to WAV Converter for Konkani ASR")
    print("=" * 50)

    # Convert files
    success = convert_m4a_to_wav()

    if success:
        # Verify conversions
        verify_conversions()

        print("\n🚀 Next steps:")
        print("1. Check that all WAV files are in data/audio/")
        print("2. Run data preparation:")
        print("   python scripts/prepare_data.py --audio_dir data/audio --transcript_dir data/transcripts --output_dir data/test_run")
        print("3. Test the ASR pipeline!")

    else:
        print("\n💡 Alternative: Use online converter")
        print("1. Go to: https://cloudconvert.com/m4a-to-wav")
        print("2. Convert each M4A to WAV (16000 Hz, Mono, 16-bit)")
        print("3. Save as sentence_01.wav, sentence_02.wav, sentence_03.wav in data/audio/")
<parameter name="filePath">c:\Users\Milind Kopikare\Code\amchi_konkani\konkani_asr\convert_m4a_to_wav.py