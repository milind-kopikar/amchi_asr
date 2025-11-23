#!/usr/bin/env python3
"""
Simple audio recording script for Konkani ASR testing
Alternative to FFmpeg - uses Python libraries
"""

import os
import time
import numpy as np

def record_with_python():
    """Record using sounddevice + soundfile (if available)"""
    try:
        import sounddevice as sd
        import soundfile as sf

        def record_sentence(filename, text, duration=5):
            print(f"🎤 Recording: {text}")
            print(f"File: {filename}")
            print(f"Duration: {duration} seconds")
            print("Press Enter to start...")
            input()

            print("🔴 Recording... Speak now!")

            # Record audio (16kHz, mono)
            sample_rate = 16000
            audio = sd.rec(int(duration * sample_rate),
                         samplerate=sample_rate,
                         channels=1)
            sd.wait()

            # Save as WAV
            sf.write(filename, audio, sample_rate)
            file_size = os.path.getsize(filename)
            print(f"✅ Saved: {filename} ({file_size} bytes)")

        # Create audio directory
        os.makedirs("data/audio", exist_ok=True)

        # Record sentences
        sentences = [
            ("data/audio/sentence_01.wav", "चल रे भोपळा टुनुक टुनुक", 5),
            ("data/audio/sentence_02.wav", "एकी गोम्टी काणी आय्कयाति!", 5),
            ("data/audio/sentence_03.wav", "एक घरांतु एकी आज्जी एक्ऴि राब्तालि।", 8)
        ]

        for filename, text, duration in sentences:
            record_sentence(filename, text, duration)
            print()

        print("🎉 All recordings completed!")
        return True

    except ImportError:
        print("❌ sounddevice or soundfile not available")
        print("Install with: pip install sounddevice soundfile")
        return False

def record_with_pyaudio():
    """Record using pyaudio (alternative)"""
    try:
        import pyaudio
        import wave

        def record_sentence(filename, text, duration=5):
            print(f"🎤 Recording: {text}")
            print(f"File: {filename}")
            print("Press Enter to start...")
            input()

            print("🔴 Recording... Speak now!")

            # Audio settings
            chunk = 1024
            sample_format = pyaudio.paInt16
            channels = 1
            fs = 16000
            seconds = duration

            p = pyaudio.PyAudio()

            stream = p.open(format=sample_format,
                          channels=channels,
                          rate=fs,
                          frames_per_buffer=chunk,
                          input=True)

            frames = []

            for i in range(0, int(fs / chunk * seconds)):
                data = stream.read(chunk)
                frames.append(data)

            stream.stop_stream()
            stream.close()
            p.terminate()

            # Save as WAV
            wf = wave.open(filename, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(sample_format))
            wf.setframerate(fs)
            wf.writeframes(b''.join(frames))
            wf.close()

            file_size = os.path.getsize(filename)
            print(f"✅ Saved: {filename} ({file_size} bytes)")

        # Create audio directory
        os.makedirs("data/audio", exist_ok=True)

        # Record sentences
        sentences = [
            ("data/audio/sentence_01.wav", "चल रे भोपळा टुनुक टुनुक", 5),
            ("data/audio/sentence_02.wav", "एकी गोम्टी काणी आय्कयाति!", 5),
            ("data/audio/sentence_03.wav", "एक घरांतु एकी आज्जी एक्ऴि राब्तालि।", 8)
        ]

        for filename, text, duration in sentences:
            record_sentence(filename, text, duration)
            print()

        print("🎉 All recordings completed!")
        return True

    except ImportError:
        print("❌ pyaudio not available")
        print("Install with: pip install pyaudio")
        return False

if __name__ == "__main__":
    print("🎤 Konkani ASR Recording Script")
    print("=" * 40)

    # Try sounddevice first (better)
    if not record_with_python():
        # Fallback to pyaudio
        if not record_with_pyaudio():
            print("❌ No recording libraries available")
            print("Please install:")
            print("  pip install sounddevice soundfile")
            print("  OR")
            print("  pip install pyaudio")
            print("  OR")
            print("Use Windows Voice Recorder + online converter")
<parameter name="filePath">c:\Users\Milind Kopikare\Code\amchi_konkani\konkani_asr\record_sentences.py