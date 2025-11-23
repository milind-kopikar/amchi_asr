# WhatsApp Audio Converter for ASR Data Collection

This tool converts WhatsApp .opus audio recordings to .wav format for ASR training.

## Installation

```bash
pip install pydub
```

Also install ffmpeg (required by pydub):
- **Windows**: `conda install ffmpeg` or download from https://ffmpeg.org/
- **Linux**: `sudo apt-get install ffmpeg`
- **macOS**: `brew install ffmpeg`

## Usage

### Basic Conversion

```bash
python scripts/whatsapp_audio_converter.py input_directory output_directory
```

### With Validation

```bash
python scripts/whatsapp_audio_converter.py input_directory output_directory --validate
```

## Crowdsourcing Workflow

### 1. Collect Audio on WhatsApp
- Ask participants to record Konkani sentences
- Have them send audio messages to you

### 2. Export Audio Files
- Save WhatsApp audio as .opus files
- Organize in folders by speaker/sentence

### 3. Batch Convert
```bash
# Convert all files in one go
python scripts/whatsapp_audio_converter.py whatsapp_recordings/ converted_audio/

# Validate conversions
python scripts/whatsapp_audio_converter.py whatsapp_recordings/ converted_audio/ --validate
```

### 4. Verify Quality
- Listen to converted .wav files
- Check file properties (16kHz, mono)
- Ensure audio is clear and complete

## File Organization

```
whatsapp_recordings/
├── speaker_01/
│   ├── sentence_001.opus
│   ├── sentence_002.opus
│   └── sentence_003.opus
├── speaker_02/
│   ├── sentence_001.opus
│   └── sentence_002.opus
└── speaker_03/
    └── sentence_001.opus

converted_audio/
├── speaker_01/
│   ├── sentence_001.wav  ← Converted files
│   ├── sentence_002.wav
│   └── sentence_003.wav
├── speaker_02/
│   ├── sentence_001.wav
│   └── sentence_002.wav
└── speaker_03/
    └── sentence_001.wav
```

## Technical Details

- **Input**: .opus files (WhatsApp audio format)
- **Output**: .wav files (16kHz, mono, PCM)
- **Quality**: Maintains original audio quality
- **Batch Processing**: Handles multiple files automatically
- **Validation**: Optional quality checking of converted files

## Troubleshooting

### "pydub not found"
```bash
pip install pydub
```

### "ffmpeg not found"
Install ffmpeg for your system.

### "No .opus files found"
- Check file extensions (should be .opus, not .oga)
- Verify input directory path
- Check file permissions

### Poor audio quality
- Original WhatsApp audio may be compressed
- Consider asking participants to record in quieter environments
- Use external microphones if possible

## Integration with ASR Pipeline

After conversion, use the .wav files with your existing ASR training:

```bash
# Add converted files to your data directory
cp converted_audio/* data/audio/

# Update manifests
python scripts/create_manifest.py converted_audio/ data/manifests/
```

This tool bridges the gap between WhatsApp crowdsourcing and your ASR training pipeline!