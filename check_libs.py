try:
    import librosa
    import soundfile
    print("✅ Audio libraries available")
except ImportError as e:
    print(f"❌ Missing library: {e}")
    print("Install with: pip install librosa soundfile")