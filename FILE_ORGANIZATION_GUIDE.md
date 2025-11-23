# File Organization for Konkani ASR Testing

## 📁 **Recommended Structure for Your 3-Sentence Test**

Create this folder structure in your project directory:

```
c:\Users\Milind Kopikare\Code\amchi_konkani\konkani_asr\
├── data\
│   ├── audio\           # Audio files go here
│   │   ├── sentence_01.wav
│   │   ├── sentence_02.wav
│   │   └── sentence_03.wav
│   └── transcripts\     # Text files go here
│       ├── sentence_01.txt
│       ├── sentence_02.txt
│       └── sentence_03.txt
└── story1.txt          # Your original story file
```

## 🎯 **Your First 3 Sentences from story1.txt**

### **Sentence 1:**
**Text:** चल रे भोपळा टुनुक टुनुक  
**Audio File:** `data/audio/sentence_01.wav`  
**Transcript File:** `data/transcripts/sentence_01.txt`

### **Sentence 2:**
**Text:** एकी गोम्टी काणी आय्कयाति!  
**Audio File:** `data/audio/sentence_02.wav`  
**Transcript File:** `data/transcripts/sentence_02.txt`

### **Sentence 3:**
**Text:** एक घरांतु एकी आज्जी एक्ऴि राब्तालि।  
**Audio File:** `data/audio/sentence_03.wav`  
**Transcript File:** `data/transcripts/sentence_03.txt`

## 📝 **Step-by-Step Recording Process**

### **Step 1: Create Directory Structure**
```bash
# Create the data directories
mkdir -p data/audio data/transcripts
```

### **Step 2: Create Transcript Files First**
```bash
# Create transcript files with the exact text
echo "चल रे भोपळा टुनुक टुनुक" > data/transcripts/sentence_01.txt
echo "एकी गोम्टी काणी आय्कयाति!" > data/transcripts/sentence_02.txt
echo "एक घरांतु एकी आज्जी एक्ऴि राब्तालि।" > data/transcripts/sentence_03.txt
```

### **Step 3: Record Audio Files**
For each sentence, record separately:

```bash
# Record Sentence 1 (speak: चल रे भोपळा टुनुक टुनुक)
ffmpeg -f dshow -i audio="Microphone (Realtek(R) Audio)" -t 5 -acodec pcm_s16le -ar 16000 -ac 1 data/audio/sentence_01.wav -y

# Record Sentence 2 (speak: एकी गोम्टी काणी आय्कयाति!)
ffmpeg -f dshow -i audio="Microphone (Realtek(R) Audio)" -t 5 -acodec pcm_s16le -ar 16000 -ac 1 data/audio/sentence_02.wav -y

# Record Sentence 3 (speak: एक घरांतु एकी आज्जी एक्ऴि राब्तालि।)
ffmpeg -f dshow -i audio="Microphone (Realtek(R) Audio)" -t 8 -acodec pcm_s16le -ar 16000 -ac 1 data/audio/sentence_03.wav -y
```

## 🎵 **Recording Tips**

- **Speak clearly and at normal pace**
- **Use a quiet environment**
- **Record each sentence separately**
- **Take your time - accuracy matters more than speed**
- **Use 16kHz mono WAV format** (automatically handled by FFmpeg)

## ✅ **Verification**

After recording, verify your files:

```bash
# Check file structure
ls -la data/audio/
ls -la data/transcripts/

# Verify audio format (should show 16kHz mono)
ffprobe data/audio/sentence_01.wav

# Check transcript content
cat data/transcripts/sentence_01.txt
```

## 🚀 **Next Step: Test the Pipeline**

Once you have the 3 audio + transcript pairs, run:

```bash
# Prepare data for ASR training
python scripts/prepare_data.py \
  --audio_dir data/audio \
  --transcript_dir data/transcripts \
  --output_dir data/test_run

# This will create:
# - data/test_run/train.tsv
# - data/test_run/val.tsv
# - data/test_run/test.tsv
```

## 📊 **Why This Structure Works**

1. **Matching Names**: Audio and transcript files have identical names (just different extensions)
2. **Clear Organization**: Separate directories for audio and text
3. **Scalable**: Easy to add more sentences later
4. **ASR Compatible**: Works perfectly with the prepare_data.py script
5. **Backup Safe**: Original story1.txt remains untouched

## 🎯 **Ready to Record?**

1. Create the directories
2. Create the transcript files
3. Record each sentence separately
4. Run the data preparation
5. Test the full ASR pipeline!

**This will give us 3 clean audio-transcript pairs to validate the entire system!** 🚀</content>
<parameter name="filePath">C:\Users\Milind Kopikare\Code\amchi_konkani\konkani_asr\FILE_ORGANIZATION_GUIDE.md