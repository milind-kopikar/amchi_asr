# Amchi Konkani ASR - Automatic Speech Recognition for Konkani Language

## 🎯 Project Overview

This project implements Automatic Speech Recognition (ASR) for Konkani language, specifically targeting the "Amchi Konkani" dialect. The system is designed for a science fair demonstration and supports multiple ASR frameworks for flexibility and performance comparison.

### Goals
- Build functional Konkani ASR using Marathi base models
- Support framework switching (HuggingFace, NeMo, AI4Bharat)
- Enable easy migration between Windows/Mac environments
- Demonstrate working transcription with acceptable accuracy
- Compare performance across different frameworks

## 🏗️ Architecture

### Framework Architecture
The system uses a configurable framework approach with three supported ASR backends:

1. **HuggingFace Transformers** (✅ Working on Windows)
   - Base: `hriteshMaikap/marathi-asr-model` (Wav2Vec2-BERT)
   - Status: Fully functional with training and testing
   - Best for: Windows development, quick prototyping

2. **NVIDIA NeMo** (🔄 Planned)
   - Status: Windows compatibility issues (os.uname, signal.SIGKILL)
   - Best for: Production deployment, advanced features

3. **AI4Bharat** (🔄 Planned)
   - Status: Requires Linux/Mac environment
   - Best for: Indian language specialization, academic research

### Code Structure
```
konkani_asr/
├── configs/                    # Configuration files
│   ├── main_config.yaml       # Framework selector
│   ├── huggingface_config.yaml # HF training params
│   ├── nemo_config.yaml       # NeMo config (placeholder)
│   └── ai4bharat_config.yaml  # AI4Bharat config (placeholder)
├── scripts/                   # Main execution scripts
│   ├── fine_tune_hf.py       # HuggingFace training
│   ├── test_model.py         # Model testing
│   └── switch_framework.py   # Framework switching utility
├── data/                     # Training data
│   ├── audio/               # Audio files (.m4a, .wav)
│   └── test_run/            # Manifests (train.tsv, dev.tsv, test.tsv)
├── models/                  # Local model storage (C: drive)
└── D:/konkani_asr_models/   # Model checkpoints (D: drive - more space)
```

## 📊 Data Format

### Audio Data
- **Format**: M4A (primary), WAV (fallback)
- **Sample Rate**: 16kHz
- **Channels**: Mono
- **Duration**: 1-30 seconds per sample
- **Location**: `data/audio/`

### Text Data
- **Language**: Konkani (Devanagari script)
- **Encoding**: UTF-8
- **Format**: TSV manifests with columns: `audio_filepath`, `text`

### Manifest Format
```
audio_filepath	text
sentence_01.m4a	पाव वाट दाण्टुनु वत्ता म्हण्तना तिका एकु सिंहु मेऴ्ळो!
sentence_02.m4a	[konkani text here]
```

## 🤖 Models & Configurations

### Current Working Model
- **Framework**: HuggingFace
- **Base Model**: `hriteshMaikap/marathi-asr-model`
- **Architecture**: Wav2Vec2-BERT
- **Model Path**: `D:/konkani_asr_models/huggingface_konkani/checkpoint-5`
- **Training Data**: 1 sample (for testing)
- **Performance**: WER ~83% (expected to improve with more data)

### Configuration Files
- **Main Config**: `configs/main_config.yaml`
  - Controls active framework
  - Defines source/target languages
  - Sets model directories

- **Framework Configs**:
  - `configs/huggingface_config.yaml`: Training parameters, model settings
  - `configs/nemo_config.yaml`: NeMo-specific settings (placeholder)
  - `configs/ai4bharat_config.yaml`: AI4Bharat settings (placeholder)

## 🚀 Setup & Installation

### Prerequisites
- Python 3.13+
- FFmpeg (for M4A audio processing)
- 8GB+ RAM, 10GB+ disk space

### Installation Steps
1. **Clone and navigate**:
   ```bash
   cd /path/to/konkani_asr
   ```

2. **Install dependencies**:
   ```bash
   pip install torch transformers datasets librosa soundfile scipy evaluate
   ```

3. **Verify FFmpeg**:
   - FFmpeg binary located at: `ffmpeg/ffmpeg-8.0.1-essentials_build/bin/ffmpeg.exe`
   - Used for M4A to WAV conversion

4. **Configure framework**:
   - Edit `configs/main_config.yaml` to set `framework: huggingface`
   - Adjust paths in config files as needed

## 🎯 Usage

### Framework Switching
```bash
python scripts/switch_framework.py list    # Show available frameworks
python scripts/switch_framework.py set huggingface  # Switch to HuggingFace
```

### Training
```bash
python scripts/fine_tune_hf.py
```
- Loads data from `data/test_run/`
- Trains model with parameters from `configs/huggingface_config.yaml`
- Saves checkpoint to `D:/konkani_asr_models/huggingface_konkani/`

### Testing
```bash
python scripts/test_model.py
```
- Loads model from `D:/konkani_asr_models/huggingface_konkani/checkpoint-5`
- Tests on `data/audio/sentence_06.m4a`
- Outputs transcription results

## 📈 Current Performance

### Metrics (1 training sample)
- **Training Loss**: 0.6335 → 0.2079 (5 steps)
- **Validation WER**: 83.3%
- **Test Transcription**: Good semantic understanding, minor character differences

### Sample Results
**Input Audio**: `sentence_06.m4a`  
**Expected Text**: `पाव वाट दाण्टुनु वत्ता म्हण्तना तिका एकु सिंहु मेऴ्ळो!`  
**Predicted Text**: `पाव वाट दाणटुनु वत्ता म्हणतना तिका एकु सिंहु मेळो`  
**Analysis**: Core words correct, understandable Konkani output

## 🔄 Framework Migration

### Windows → Mac/Linux Transition
1. **Copy project** to new environment
2. **Update paths** in config files
3. **Switch framework** in `main_config.yaml`
4. **Reinstall dependencies** (may need different versions)
5. **Retrain models** (framework-specific checkpoints)

### AI4Bharat Integration Plan
- **Environment**: Linux/Mac required
- **Authentication**: May need API tokens
- **Data Format**: Compatible with current TSV manifests
- **Expected Benefits**: Better Indian language support, potentially higher accuracy

## 🐛 Known Issues & Solutions

### Windows-Specific Issues
- **NeMo**: `os.uname` undefined, `signal.SIGKILL` missing
- **Solution**: Use WSL or migrate to Mac/Linux

### Audio Processing
- **M4A Support**: Requires FFmpeg for conversion
- **Solution**: FFmpeg binary included in project

### Disk Space
- **Issue**: Model saving fails with <4GB free space
- **Solution**: Save to D: drive (`D:/konkani_asr_models/`)

### Model Loading
- **Issue**: Missing `preprocessor_config.json`
- **Solution**: Copy from base model after training

## 🎯 Next Steps

### Immediate Tasks
1. **Scale training data** from 1 to full dataset (19 samples)
2. **AI4Bharat integration** on Mac/Linux
3. **Performance comparison** between frameworks
4. **Accuracy improvements** with more training data

### Long-term Goals
- Production deployment
- Real-time inference optimization
- Multi-speaker adaptation
- Extended Konkani dialect support

## 📚 Key Files for New Agents

When a new agent reviews this project, please have them examine:

1. **`configs/main_config.yaml`** - Framework selection and paths
2. **`scripts/fine_tune_hf.py`** - Main training logic
3. **`scripts/test_model.py`** - Testing and inference
4. **`data/test_run/train.tsv`** - Data format example
5. **This README** - Complete project overview

## 🤝 Contributing

For AI4Bharat integration or framework comparisons:
1. Review this README thoroughly
2. Test current HuggingFace pipeline
3. Set up target environment (Mac/Linux for AI4Bharat)
4. Follow framework-specific setup in config files
5. Compare performance metrics systematically

---

**Last Updated**: November 25, 2025  
**Status**: HuggingFace pipeline fully functional, AI4Bharat integration pending Mac environment setup