# Simple Recording Solution for Konkani ASR

## 🎯 **Problem: FFmpeg Not Available**

The recording script failed because FFmpeg is not installed. Here are **3 easy solutions**:

## ✅ **Solution 1: Install FFmpeg (Recommended)**

### **Step 1: Download FFmpeg**
1. Go to: https://ffmpeg.org/download.html
2. Download the **Windows build** (not the source)
3. Extract to a folder (e.g., `C:\ffmpeg`)

### **Step 2: Add to PATH**
```powershell
# Add FFmpeg to your PATH (run in PowerShell as Administrator)
$ffmpegPath = "C:\ffmpeg\bin"
$env:Path += ";$ffmpegPath"
[Environment]::SetEnvironmentVariable("Path", $env:Path, "User")
```

### **Step 3: Test**
```powershell
ffmpeg -version
```

## ✅ **Solution 2: Use Windows Voice Recorder (Easiest)**

### **Step 1: Record with Windows Voice Recorder**
1. Open **Windows Voice Recorder** app
2. Click record button
3. Speak: **"चल रे भोपळा टुनुक टुनुक"**
4. Stop recording
5. Save as: `sentence_01.m4a` (default format)

### **Step 2: Convert to WAV**
```powershell
# If you have FFmpeg later, convert:
ffmpeg -i sentence_01.m4a -ar 16000 -ac 1 data/audio/sentence_01.wav
```

### **Step 3: Or Use Online Converter**
- Upload your M4A file to: https://cloudconvert.com/m4a-to-wav
- Convert to: **WAV, 16kHz, Mono**
- Download and save as `data/audio/sentence_01.wav`

## ✅ **Solution 3: Use Python Recording (Alternative)**

If you have Python, we can create a simple recording script:

```python
import sounddevice as sd
import soundfile as sf
import numpy as np

def record_sentence(filename, duration=5, sample_rate=16000):
    print(f"🎤 Recording: {filename}")
    print("Speak now...")

    # Record audio
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()

    # Save as WAV
    sf.write(filename, audio, sample_rate)
    print(f"✅ Saved: {filename}")

# Record your sentences
record_sentence("data/audio/sentence_01.wav", duration=5)  # चल रे भोपळा टुनुक टुनुक
record_sentence("data/audio/sentence_02.wav", duration=5)  # एकी गोम्टी काणी आय्कयाति!
record_sentence("data/audio/sentence_03.wav", duration=8)  # एक घरांतु एकी आज्जी एक्ऴि राब्तालि।
```

## 📋 **Your Recording Checklist**

### **For Each Sentence:**

1. **Prepare to record:**
   - Find a quiet room
   - Test your microphone volume
   - Have the text ready: "चल रे भोपळा टुनुक टुनुक"

2. **Record clearly:**
   - Speak at normal speed
   - Pronounce words clearly
   - No background noise
   - 5-8 seconds duration

3. **Save correctly:**
   - Format: WAV (or M4A then convert)
   - Sample Rate: 16kHz
   - Channels: Mono
   - Location: `data/audio/sentence_01.wav`

## 🎯 **Quick Test Without Recording**

For testing the pipeline, you can use any short audio file. Let's create a simple test:

```bash
# Create a test audio file (beep sound)
ffmpeg -f lavfi -i "sine=frequency=1000:duration=3" -ar 16000 -ac 1 data/audio/sentence_01.wav
```

## 🚀 **Recommended Approach**

**Start with Solution 2 (Windows Voice Recorder):**
1. Record with Windows app
2. Convert online to WAV 16kHz mono
3. Test the ASR pipeline

**Then install FFmpeg for future recordings.**

---

**Which solution would you like to try first?** The Windows Voice Recorder is probably the easiest to get started immediately! 🎤</content>
<parameter name="filePath">c:\Users\Milind Kopikare\Code\amchi_konkani\konkani_asr\RECORDING_TROUBLESHOOT.md