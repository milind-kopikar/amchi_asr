# Correct WAV Conversion Settings for CloudConvert

## 🎯 **CloudConvert WAV Settings**

You're right - CloudConvert doesn't show "Bit Depth" directly. Here's the **correct settings**:

### ✅ **Required Settings for ASR:**

| Setting | Value | Why |
|---------|-------|-----|
| **Audio Codec** | `PCM` | 16-bit WAV format |
| **Sample Rate** | `16000 Hz` | ASR standard |
| **Channels** | `Mono` | Single channel |
| **Audio Bitrate** | `256 kbps` | Auto-set by PCM |

### 📋 **Step-by-Step:**

1. **Upload** your WAV file
2. **Select format:** WAV
3. **Click "Options"** or **"Show advanced options"**
4. **Set:**
   - Audio Codec: **PCM**
   - Sample Rate: **16000 Hz**
   - Channels: **1 (Mono)**
5. **Convert**

### 🎵 **What PCM Means:**
- **PCM** = Pulse Code Modulation
- **For WAV** = 16-bit by default
- **Perfect for ASR** = Clean, uncompressed audio

### 📊 **Expected Results:**
After proper conversion:
- ✅ File size: ~80-128KB per 5-8 second sentence
- ✅ Sample rate: 16000 Hz confirmed
- ✅ Channels: Mono (1)
- ✅ Bit depth: 16-bit (via PCM codec)

### 🔍 **Verify After Conversion:**
```bash
python check_audio_properties.py
```

**Look for:**
- Sample Rate: 16000 Hz ✅
- Channels: 1 ✅
- File size: Much smaller ✅

---

**Use PCM codec with 16000 Hz sample rate - that gives you perfect 16-bit mono WAV files!** 🎵</content>
<parameter name="filePath">c:\Users\Milind Kopikare\Code\amchi_konkani\konkani_asr\CLOUDCONVERT_SETTINGS.md