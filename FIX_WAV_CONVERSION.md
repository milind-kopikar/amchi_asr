# Fix WAV Files for Konkani ASR

## ⚠️ **Problem: Files Are Wrong Sample Rate**

Your WAV files are **too large** and **too long**, indicating they're at high sample rate (probably 44.1kHz or 48kHz) instead of the required 16kHz.

### **Current Status:**
- ❌ `sentence_01.wav`: 827KB (26 seconds estimated)
- ❌ `sentence_02.wav`: 1.1MB (35 seconds estimated)  
- ❌ `sentence_03.wav`: 1.5MB (49 seconds estimated)

**Expected for 5-8 second sentences at 16kHz:**
- ✅ Should be: ~80-128KB each
- ✅ Should be: 5-8 seconds each

## ✅ **Solution: Re-convert with Correct Settings**

### **Step 1: Go to Converter**
Visit: https://cloudconvert.com/wav-converter

### **Step 2: Upload and Convert Each File**
For each WAV file:
1. **Upload** your current `sentence_01.wav`
2. **Change settings:**
   - **Sample Rate**: `16000 Hz` (not default!)
   - **Channels**: `Mono` (1 channel)
   - **Bit Depth**: `16 bit`
3. **Convert** and **download**
4. **Replace** the file in `data/audio/` folder

### **Step 3: Verify After Conversion**
```bash
python check_audio_properties.py
```

**Expected results after proper conversion:**
- ✅ `sentence_01.wav`: ~80KB (5 seconds)
- ✅ `sentence_02.wav`: ~80KB (5 seconds)  
- ✅ `sentence_03.wav`: ~128KB (8 seconds)

## 🔧 **Alternative: Install Audio Libraries**

If you want to automate conversion:

```bash
# Install libraries
pip install soundfile

# Re-run property checker (will give detailed info)
python check_audio_properties.py
```

## 🎯 **Why This Matters**

- **16kHz**: Standard for speech recognition models
- **Mono**: Reduces file size and processing
- **16-bit**: Optimal quality without waste
- **Correct duration**: Ensures proper training batches

## 🚀 **After Proper Conversion**

Once files are correctly converted:

```bash
# Test data preparation
python scripts/prepare_data.py \
  --audio_dir data/audio \
  --transcript_dir data/transcripts \
  --output_dir data/test_run

# Check manifest files
dir data\test_run\
```

---

**Re-convert your WAV files with 16kHz sample rate - that's the key setting you missed!** 🎵

*Note: The default online converter settings keep the original high sample rate, which is why your files are so large.*</content>
<parameter name="filePath">c:\Users\Milind Kopikare\Code\amchi_konkani\konkani_asr\FIX_WAV_CONVERSION.md