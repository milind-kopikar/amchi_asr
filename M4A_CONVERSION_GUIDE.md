# Convert M4A to WAV for Konkani ASR

## 🎯 **Your M4A Files Need Conversion**

You have 3 M4A files that need to be converted to WAV format for ASR:

```
data/audio/
├── sentence_01.m4a ✅ (needs conversion)
├── sentence_02.m4a ✅ (needs conversion)
└── sentence_03.m4a ✅ (needs conversion)
```

## ✅ **Easiest Method: Online Converter**

### **Step 1: Go to Converter**
Visit: https://cloudconvert.com/m4a-to-wav

### **Step 2: Convert Each File**
For each M4A file:
1. Click **"Select File"** and upload `sentence_01.m4a`
2. Change settings:
   - **Sample Rate**: 16000 Hz
   - **Channels**: Mono
   - **Bit Depth**: 16 bit
3. Click **"Convert"**
4. Download as `sentence_01.wav`
5. Save in `data/audio/` folder

### **Step 3: Repeat for All Files**
- `sentence_01.m4a` → `sentence_01.wav`
- `sentence_02.m4a` → `sentence_02.wav`
- `sentence_03.m4a` → `sentence_03.wav`

## ✅ **Alternative: Install Python Libraries**

If you want to automate this later:

```bash
# Install required libraries
pip install librosa soundfile numpy

# Then run the conversion script
python convert_m4a_to_wav.py
```

## 📋 **After Conversion: Verify Files**

Once converted, check your files:

```bash
# List files
dir data\audio\

# Should show:
# sentence_01.m4a (original)
# sentence_01.wav (converted)
# sentence_02.m4a (original)
# sentence_02.wav (converted)
# sentence_03.m4a (original)
# sentence_03.wav (converted)
```

## 🚀 **Next Step: Test ASR Pipeline**

After conversion, run:

```bash
# Prepare data for training
python scripts/prepare_data.py ^
  --audio_dir data/audio ^
  --transcript_dir data/transcripts ^
  --output_dir data/test_run

# Check results
dir data\test_run\
```

This will create your manifest files and test Block 1 (Data Management)!

---

**Start with the online converter - it's the fastest way to get your WAV files!** 🎵

*Note: The online converter will give you perfect 16kHz mono WAV files for ASR.*</content>
<parameter name="filePath">c:\Users\Milind Kopikare\Code\amchi_konkani\konkani_asr\M4A_CONVERSION_GUIDE.md