# Audio Format Requirements for Konkani ASR

## 🎵 Required Audio Format Specifications

### ✅ **Supported File Formats**
The system accepts these audio formats:
- **WAV** (`.wav`) - **Recommended**
- **MP3** (`.mp3`)
- **FLAC** (`.flac`)
- **OGG** (`.ogg`)
- **M4A** (`.m4a`)

### 🎯 **Critical Requirements**

#### **Sample Rate: 16kHz (Required)**
```python
# The system expects 16kHz audio
# If your audio is different, it will be automatically resampled
target_sample_rate = 16000  # Hz
```

**Why 16kHz?**
- Standard for speech recognition models
- IndicConformer was trained on 16kHz data
- Optimal balance of quality vs. processing speed

#### **Channels: Mono (Required)**
```python
# Must be single channel (mono), not stereo
required_channels = 1
```

**Why Mono?**
- Speech is typically recorded in mono
- Reduces processing requirements
- ASR models expect single-channel input

#### **Bit Depth: 16-bit (Recommended)**
- **16-bit PCM** for WAV files
- Higher bit depths (24-bit, 32-bit) also work
- Compressed formats (MP3) automatically handled

### 📊 **Audio Quality Guidelines**

#### **Recommended Settings for Recording**
```bash
# Using FFmpeg to record audio:
ffmpeg -f dshow -i audio="Your Microphone" \
       -t 30 \                    # 30 seconds
       -acodec pcm_s16le \        # 16-bit PCM
       -ar 16000 \               # 16kHz sample rate
       -ac 1 \                   # Mono (1 channel)
       "output.wav"
```

#### **File Size Estimates**
For your **10 minutes of audio**:
- **WAV (16kHz, 16-bit, mono)**: ~9.6 MB
- **MP3 (128kbps)**: ~7.6 MB
- **FLAC (compressed)**: ~4.8 MB

### 🔄 **Automatic Processing**

#### **What Happens to Your Audio**
1. **Format Detection**: System identifies your file type
2. **Sample Rate Check**: If not 16kHz → **Automatic resampling**
3. **Channel Check**: If stereo → **Automatic conversion to mono**
4. **Normalization**: Audio levels adjusted for consistency
5. **Validation**: Quality checks and warnings

#### **Automatic Conversions**
```python
# Example: Your 44kHz stereo MP3 → 16kHz mono WAV
original: "story.mp3" (44kHz, stereo, MP3)
processed: internally converted to 16kHz mono for ASR
```

### ⚠️ **Common Issues & Solutions**

#### **Issue 1: Wrong Sample Rate**
```
❌ Your audio: 44kHz (CD quality)
✅ System will: Automatically resample to 16kHz
⚠️ Warning: "Sample rate 44100Hz, recommended 16000Hz"
```

#### **Issue 2: Stereo Audio**
```
❌ Your audio: Stereo (2 channels)
✅ System will: Convert to mono (average both channels)
⚠️ Warning: "2 channels, recommended mono"
```

#### **Issue 3: Unsupported Format**
```
❌ Your audio: WMA, AAC, etc.
✅ Solution: Convert to WAV/MP3/FLAC first
```

### 🛠️ **How to Prepare Your Audio**

#### **Option 1: Use the Recording Script (Easiest)**
```bash
# The included script records in correct format
python scripts/prepare_recording.py --duration 600  # 10 minutes
```

#### **Option 2: Convert Existing Audio**
```bash
# Convert any audio to required format
ffmpeg -i your_audio.mp3 \
       -ar 16000 \        # Set sample rate to 16kHz
       -ac 1 \           # Convert to mono
       -c:a pcm_s16le \  # 16-bit PCM
       output.wav
```

#### **Option 3: Batch Convert Multiple Files**
```bash
# Convert all MP3 files in a directory
for file in *.mp3; do
    ffmpeg -i "$file" -ar 16000 -ac 1 "${file%.mp3}.wav"
done
```

### 📱 **Mobile/Recording Device Settings**

#### **Phone Recording Apps**
- **Sample Rate**: Set to 16kHz or 48kHz (will be resampled)
- **Channels**: Mono
- **Format**: WAV or M4A

#### **Microphone Settings**
- **USB Microphone**: Set to 16kHz, mono in device settings
- **Built-in Mic**: Usually mono by default
- **External Recorder**: 16kHz WAV format

### 🔍 **Validation Checklist**

Before using your audio, verify:

```python
import soundfile as sf

# Check your audio file
info = sf.info("your_audio.wav")
print(f"Sample Rate: {info.samplerate} Hz")  # Should be ~16000
print(f"Channels: {info.channels}")          # Should be 1
print(f"Duration: {info.duration:.1f}s")     # Your 10 minutes
```

### 📋 **Quick Reference**

| Requirement | Value | Notes |
|-------------|-------|-------|
| **File Formats** | WAV, MP3, FLAC, OGG, M4A | All supported |
| **Sample Rate** | 16kHz | Automatically resampled if different |
| **Channels** | Mono (1) | Automatically converted from stereo |
| **Bit Depth** | 16-bit | Recommended for WAV |
| **Duration** | Any length | 10 minutes is perfect |

### 🎯 **For Your 10-Minute Konkani Story**

**✅ Your current audio will work!**

- If it's already 16kHz mono: Perfect, use as-is
- If it's different: System will automatically convert
- If it's compressed: MP3/FLAC work fine

**The system is designed to be flexible with audio formats while ensuring optimal ASR performance!** 🚀

*Note: The automatic preprocessing handles most format issues, so don't worry if your audio isn't perfect - the system will fix it.*</content>
<parameter name="filePath">C:\Users\Milind Kopikare\Code\amchi_konkani\konkani_asr\AUDIO_FORMAT_GUIDE.md