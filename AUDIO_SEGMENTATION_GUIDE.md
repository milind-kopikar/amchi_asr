# Audio Segmentation for Konkani ASR Training

## Your Question: Long Story vs. Single Sentences?

**Short Answer**: You can use either approach, but **segmented sentences are better for training quality**.

## Option 1: Long Audio File (100-line Story) ✅

### What You Can Do:
- **Keep your entire 10-minute story as one audio file**
- **Create one corresponding text file with all 100 lines**
- **The manifest generator will create one training example**

### Example:
```
Audio: story_10_minutes.wav (600 seconds)
Text: "तुमी कसो आसा माझे नाव मिलिंद आहे कोकणी भाषा छान आहे..." (entire story)
Manifest: One line with 600-second audio
```

### Pros ✅:
- **Simple**: No need to split audio
- **Natural speech**: Preserves storytelling flow
- **Easy setup**: One audio + one text file

### Cons ❌:
- **Poor training**: Model learns one long pattern
- **Memory issues**: 10-minute audio needs lots of RAM
- **Hard alignment**: Audio-text sync is difficult
- **Batch problems**: Can't mix with other examples

## Option 2: Segmented Sentences (Recommended) 🎯

### What You Do:
- **Split your 10-minute story into 2-5 second clips**
- **Each clip gets its own text transcript**
- **Creates 120-300 training examples**

### Example:
```
Audio: clip_001.wav (3.2s) → Text: "तुमी कसो आसा"
Audio: clip_002.wav (2.8s) → Text: "माझे नाव मिलिंद आहे"
Audio: clip_003.wav (3.5s) → Text: "कोकणी भाषा छान आहे"
...
Manifest: 200+ lines with short audio clips
```

### Pros ✅:
- **Better training**: Model learns sentence patterns
- **Efficient memory**: Small audio chunks
- **Easy batching**: Mix with other examples
- **Precise alignment**: Audio-text sync is accurate
- **More data**: 200 examples > 1 example

### Cons ❌:
- **More work**: Need to split audio and text
- **Time-consuming**: Manual segmentation

## Which Option for Your 10 Minutes?

### For Testing Manifest Generator: Either Works! ✅

**Both options will successfully:**
- Create manifest files
- Test the data preparation pipeline
- Verify NeMo can read your data

### For Best Training Results: Segmentation Wins 🏆

**Why segmentation is better:**
- **ASR models work best** with 2-10 second utterances
- **More training examples** from same audio
- **Better generalization** to new sentences
- **Industry standard** approach

## How to Segment Your Audio (If You Choose Option 2)

### Manual Segmentation:
1. **Listen to audio** and note timestamps
2. **Split audio** at sentence boundaries
3. **Extract corresponding text** for each clip

### Semi-Automatic Segmentation:
```python
import librosa
import soundfile as sf
from pydub import AudioSegment

def split_audio_by_silence(audio_path, output_dir, min_silence_len=500, silence_thresh=-40):
    """Split audio at silent gaps (sentence boundaries)"""

    # Load audio
    audio = AudioSegment.from_wav(audio_path)

    # Split on silence
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,  # 500ms silence
        silence_thresh=silence_thresh     # -40dB threshold
    )

    # Save chunks
    for i, chunk in enumerate(chunks):
        chunk.export(f"{output_dir}/chunk_{i:03d}.wav", format="wav")

    return len(chunks)

# Usage
num_chunks = split_audio_by_silence("your_story.wav", "audio_clips/")
print(f"Split into {num_chunks} segments")
```

### Text Segmentation:
```python
def split_text_by_sentences(text, audio_segments):
    """Split text to match audio segments"""

    # Split text into sentences
    sentences = text.split('।')  # Devanagari sentence marker

    # Create transcript files
    for i, sentence in enumerate(sentences):
        if sentence.strip():
            with open(f"transcripts/transcript_{i:03d}.txt", 'w', encoding='utf-8') as f:
                f.write(sentence.strip())

    return len(sentences)

# Usage
text = "तुमी कसो आसा। माझे नाव मिलिंद आहे। कोकणी भाषा छान आहे।"
num_texts = split_text_by_sentences(text, audio_segments)
```

## Recommendation for Your Project

### Start Simple, Upgrade Later:

**Phase 1: Test with Long Audio** (What you have now)
```
✅ Use your 10-minute story as-is
✅ Test manifest creation
✅ Verify pipeline works
✅ Get baseline results
```

**Phase 2: Segment for Better Training** (Future improvement)
```
🎯 Split into 2-5 second clips
🎯 Create 100+ training examples
🎯 Improve model accuracy
🎯 Follow ASR best practices
```

## Why Segmentation Matters for ASR

### Technical Reasons:

1. **Memory Efficiency**:
   - Long audio: 10MB per example
   - Short clips: 0.5MB per example
   - **Result**: Fit more examples in GPU memory

2. **Training Stability**:
   - Long audio: Unstable gradients
   - Short clips: Stable, consistent learning
   - **Result**: Faster convergence

3. **Batch Diversity**:
   - Long audio: All batches similar
   - Short clips: Varied sentence patterns
   - **Result**: Better generalization

### Real-World Example:

**Before Segmentation:**
- Training data: 1 example (10 minutes)
- Model learns: "This specific story pattern"
- Result: Works well on similar stories, poorly on new sentences

**After Segmentation:**
- Training data: 200 examples (10 minutes total)
- Model learns: "Konakani sentence patterns"
- Result: Works well on any Konkani speech

## Your Next Steps

### Option A: Test with Long Audio (Easiest)
```bash
# Place your files
cp your_10_minute_story.wav data/audio/
echo "your entire story text" > data/transcripts/story.txt

# Generate manifest
python scripts/prepare_data.py \
  --audio_dir data/audio \
  --transcript_dir data/transcripts \
  --output_dir data
```

### Option B: Segment First (Better Results)
```bash
# Split your audio into clips (manual or script)
# Create corresponding transcripts
# Then run prepare_data.py
```

## Summary

**✅ You can absolutely use your 100-line story as one long audio file for testing!**

**🎯 For best results, segment into sentences when you're ready to seriously train the model.**

**Your 10 minutes will work either way - the manifest generator is flexible!** 🚀

*Note: Most ASR research starts with long audio for testing, then segments for production training.*</content>
<parameter name="filePath">C:\Users\Milind Kopikare\Code\amchi_konkani\konkani_asr\AUDIO_SEGMENTATION_GUIDE.md