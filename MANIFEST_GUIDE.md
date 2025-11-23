# Understanding NeMo Manifest Files

## What is a Manifest File?

A **manifest file** is NeMo's way of telling the ASR model "here's your training data". It's a TSV (Tab-Separated Values) file that acts like a "table of contents" for your audio-text pairs.

### Manifest File Structure

Each line in the manifest represents one training example:

```
audio_filepath	text	duration
/path/to/audio1.wav	तुमी कसो आसा	2.5
/path/to/audio2.wav	माझे नाव मिलिंद आहे	3.1
/path/to/audio3.wav	कोकणी भाषा छान आहे	2.8
```

**Columns:**
- `audio_filepath`: Absolute path to the audio file
- `text`: The correct transcription (ground truth)
- `duration`: Audio length in seconds (optional but recommended)

### Why TSV Format?

- **Simple**: Just text, no complex binary formats
- **Human-readable**: You can open it in Excel or any text editor
- **Streaming-friendly**: NeMo can read it line-by-line without loading everything into memory
- **Language-agnostic**: Works with any language's Unicode text

## How NeMo Uses Manifest Files

### Training Process

```
Manifest File → DataLoader → Audio Batch + Text Batch → Model
     ↓              ↓              ↓                    ↓
   Reads line     Groups into     Creates tensors      Learns patterns
   by line        batches         for GPU training     from audio→text
```

### During Training

1. **DataLoader reads manifest** line by line
2. **Groups into batches** (e.g., 8 audio files at once)
3. **Loads audio files** from `audio_filepath`
4. **Pairs with text** from `text` column
5. **Creates training examples**: `(audio_tensor, text_string)`

### Example Training Batch

```python
# What NeMo sees from manifest:
batch = {
    'audio': torch.tensor([...]),  # [batch_size, audio_length]
    'audio_lengths': torch.tensor([2.5, 3.1, 2.8]),
    'texts': ['तुमी कसो आसा', 'माझे नाव मिलिंद आहे', 'कोकणी भाषा छान आहे']
}
```

## Data Requirements for Testing

### Minimum Data for Manifest Testing

**✅ YES, 10 minutes of audio will work perfectly!**

#### Why 10 minutes is sufficient:
- **Manifest creation**: Works with 1 audio file
- **Format validation**: Just needs proper TSV structure
- **Data splitting**: Can create train/val/test splits

### Recommended Data Breakdown

For **10 minutes total audio**:

```
Total: 10 minutes
├── Training: 7 minutes (70%)
├── Validation: 2 minutes (20%)
├── Test: 1 minute (10%)
```

#### Per-file recommendations:
- **Duration**: 2-10 seconds each (NeMo works best with shorter clips)
- **Format**: WAV, 16kHz, mono
- **Count**: ~30-50 audio files from your 10 minutes

### Example with Your 10 Minutes

If you split 10 minutes into 3-second clips:
- **Total clips**: ~200 audio files
- **Training set**: 140 clips (7 minutes)
- **Validation set**: 40 clips (2 minutes)
- **Test set**: 20 clips (1 minute)

## Testing the Manifest Generator

### Unit Test for Manifest Creation

```python
def test_manifest_creation():
    """Test creating manifest from audio-text pairs"""

    # Your 10 minutes of data
    audio_files = ["audio_001.wav", "audio_002.wav", ..., "audio_200.wav"]
    transcripts = ["तुमी कसो आसा", "माझे नाव मिलिंद आहे", ...]
    durations = [3.0, 2.8, 3.2, ...]  # in seconds

    # Create manifest
    manifest_path = create_manifest(audio_files, transcripts, durations,
                                   output_file="data/train.tsv")

    # Verify manifest
    df = pd.read_csv(manifest_path, sep='\t')

    # Assertions
    assert len(df) == 200  # All files included
    assert list(df.columns) == ['audio_filepath', 'text', 'duration']
    assert df['duration'].sum() == 600.0  # 10 minutes total

    # Check sample entry
    first_row = df.iloc[0]
    assert first_row['audio_filepath'].endswith('audio_001.wav')
    assert first_row['text'] == "तुमी कसो आसा"
    assert first_row['duration'] == 3.0

    print("✅ Manifest creation test passed!")
```

### Integration Test with NeMo

```python
def test_manifest_with_nemo():
    """Test that NeMo can actually read your manifest"""

    from nemo.collections.asr.data.audio_to_text import AudioToTextDataset

    # Create manifest (from above)
    manifest_path = create_manifest(audio_files, transcripts, durations)

    # Try to create NeMo dataset
    try:
        dataset = AudioToTextDataset(
            manifest_filepath=manifest_path,
            sample_rate=16000,
            batch_size=8
        )

        # Test data loading
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=8)
        batch = next(iter(data_loader))

        # Verify batch structure
        assert 'audio' in batch
        assert 'text' in batch
        assert batch['audio'].shape[0] <= 8  # Batch size

        print("✅ NeMo can successfully read your manifest!")

    except Exception as e:
        print(f"❌ NeMo manifest reading failed: {e}")
        return False

    return True
```

## Preparing Your 10 Minutes of Data

### Step 1: Audio Preparation
```bash
# Split your 10-minute audio into clips
# Using ffmpeg or audio editor:
ffmpeg -i your_10_minute_audio.wav \
       -f segment -segment_time 3 \
       -c copy audio_%03d.wav
```

### Step 2: Transcript Creation
Create corresponding text files:
```
audio_001.txt: "तुमी कसो आसा"
audio_002.txt: "माझे नाव मिलिंद आहे"
audio_003.txt: "कोकणी भाषा छान आहे"
...
```

### Step 3: Manifest Generation
```python
# Run the manifest generator
python scripts/prepare_data.py \
  --audio_dir your_audio_clips/ \
  --transcript_dir your_transcripts/ \
  --output_dir data/
```

### Step 4: Verify Splits
```python
# Check the generated files
train_df = pd.read_csv('data/train.tsv', sep='\t')
val_df = pd.read_csv('data/val.tsv', sep='\t')
test_df = pd.read_csv('data/test.tsv', sep='\t')

print(f"Training: {len(train_df)} samples")
print(f"Validation: {len(val_df)} samples")
print(f"Test: {len(test_df)} samples")
print(f"Total duration: {train_df['duration'].sum() + val_df['duration'].sum() + test_df['duration'].sum()} seconds")
```

## Why Manifest Files Matter

### Debugging Benefits
- **Inspect data**: Open TSV in Excel to see what the model will train on
- **Find issues**: Spot corrupted audio files or wrong transcripts
- **Balance check**: Ensure train/val/test have similar audio lengths

### Performance Impact
- **Memory efficient**: NeMo streams data instead of loading everything
- **Flexible**: Easy to add/remove training examples
- **Reproducible**: Same manifest = same training results

### Real-world Example
Your manifest tells NeMo:
- "Play `audio_001.wav` and expect the text 'तुमी कसो आसा'"
- "Play `audio_002.wav` and expect the text 'माझे नाव मिलिंद आहे'"

The model learns these audio→text mappings!

## Summary

**✅ Your 10 minutes of audio is perfect for testing the manifest generator!**

- **Manifest creation**: Works with any amount of data
- **Data splitting**: Can create proper train/val/test splits
- **NeMo compatibility**: Standard TSV format that NeMo expects
- **Debugging**: Easy to inspect and verify data quality

The manifest file is essentially your "dataset recipe" - it tells NeMo exactly what audio to play and what text to expect. With your 10 minutes, you'll have plenty of data to test this critical component thoroughly! 🎯</content>
<parameter name="filePath">C:\Users\Milind Kopikare\Code\amchi_konkani\konkani_asr\MANIFEST_GUIDE.md