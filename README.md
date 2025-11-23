# 🗣️ Konkani ASR Fine-tuning Project

Fine-tune the AI4Bharat IndicConformer Marathi ASR model for spoken Konkani using NVIDIA NeMo framework.

## 🎯 Project Overview

This project adapts an existing Marathi Automatic Speech Recognition (ASR) model to recognize spoken Konkani by fine-tuning it with Konkani speech data. The base model is [AI4Bharat IndicConformer](https://huggingface.co/ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large).

## 📋 Prerequisites

- Python 3.8+
- NVIDIA GPU (recommended for training)
- CUDA 11.0+ (if using GPU)
- FFmpeg (for audio processing)
- Git LFS (for large model files)

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone this repository
git clone <repository-url>
cd konkani_asr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Place your Konkani speech data in the `data/` directory:

```
data/
├── audio/          # WAV/MP3 audio files
├── transcripts/    # Text transcripts
├── train.tsv       # Training manifest
├── dev.tsv         # Development manifest
└── test.tsv        # Test manifest
```

Each manifest file should be a TSV with columns:
- `audio_filepath` (relative path to audio file)
- `text` (transcription in Konkani script)
- `duration` (audio duration in seconds)

### 3. Download Base Model

```bash
# Download IndicConformer Marathi model
python scripts/download_model.py
```

### 4. Fine-tune the Model

```bash
# Start fine-tuning
python scripts/fine_tune.py --config configs/konkani_finetune.yaml
```

### 5. Evaluate Results

```bash
# Evaluate on test set
python scripts/evaluate.py --model_path results/konkani_asr.nemo --test_manifest data/test.tsv
```

## 📁 Project Structure

```
konkani_asr/
├── configs/            # NeMo configuration files
│   ├── base_config.yaml
│   └── konkani_finetune.yaml
├── data/              # Speech data and manifests
├── scripts/           # Python scripts
│   ├── download_model.py
│   ├── prepare_data.py
│   ├── fine_tune.py
│   └── evaluate.py
├── models/            # Downloaded/fine-tuned models
├── results/           # Training outputs and checkpoints
├── logs/              # Training logs and metrics
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## ⚙️ Configuration

### Fine-tuning Configuration

Key parameters in `configs/konkani_finetune.yaml`:

```yaml
# Model configuration
model:
  nemo_model: "models/indicconformer_mr.nemo"  # Base model path

# Training configuration
trainer:
  max_epochs: 50
  accelerator: "gpu"  # or "cpu"
  devices: 1

# Data configuration
data:
  train_manifest: "data/train.tsv"
  val_manifest: "data/dev.tsv"
  batch_size: 8
  num_workers: 4

# Optimizer configuration
optim:
  lr: 0.0001
  weight_decay: 0.001
```

## 🎵 Data Format Requirements

### Audio Files
- **Format**: WAV (16kHz, 16-bit, mono) or MP3
- **Sample Rate**: 16kHz recommended
- **Channels**: Mono
- **Quality**: Clean speech, minimal background noise

### Transcripts
- **Script**: Devanagari (कोंकणी) or Romanized Konkani
- **Format**: UTF-8 encoded text
- **Punctuation**: Minimal punctuation, focus on spoken language
- **Normalization**: Consistent spelling and formatting

### Manifest Format
```tsv
audio_filepath	text	duration
audio/train_001.wav	कोंकणी भाषा	3.45
audio/train_002.wav	Hello world	2.12
```

## 🏃 Training Process

### Stage 1: Data Preparation
```bash
python scripts/prepare_data.py \
  --audio_dir data/audio \
  --transcript_dir data/transcripts \
  --output_dir data
```

### Stage 2: Model Download
```bash
python scripts/download_model.py \
  --model_name ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large \
  --output_path models/
```

### Stage 3: Fine-tuning
```bash
python scripts/fine_tune.py \
  --config configs/konkani_finetune.yaml \
  --output_dir results/
```

### Stage 4: Evaluation
```bash
python scripts/evaluate.py \
  --model_path results/konkani_asr.nemo \
  --test_manifest data/test.tsv \
  --output_file results/evaluation_results.json
```

## 📊 Monitoring Training

### TensorBoard (Recommended)
```bash
# Install TensorBoard
pip install tensorboard

# Start TensorBoard
tensorboard --logdir logs/

# Open http://localhost:6006 in browser
```

### Training Logs
Monitor `logs/train.log` for:
- Loss values (CTC and RNNT)
- Word Error Rate (WER)
- Character Error Rate (CER)
- Learning rate changes

## 🔧 Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce batch size in config
- Use gradient accumulation
- Try CPU training for testing

**Poor Audio Quality**
- Check audio sample rate (should be 16kHz)
- Ensure mono channel audio
- Verify audio file formats

**High WER**
- Increase training data size
- Adjust learning rate
- Try different data augmentation

**Model Loading Errors**
- Check model file integrity
- Verify NeMo version compatibility
- Ensure sufficient disk space

## 📈 Expected Results

### Performance Metrics
- **Initial WER**: 50-70% (on Marathi model with Konkani data)
- **After Fine-tuning**: 15-30% WER (depending on data quality/quantity)
- **Target WER**: <10% for production use

### Training Time
- **GPU (RTX 3080)**: 2-4 hours for 50 epochs
- **CPU**: 8-12 hours for 50 epochs
- **Google Colab**: 4-8 hours (with T4 GPU)

## 🚀 Deployment

### Export for Inference
```bash
# Export to ONNX (optional)
python scripts/export_model.py \
  --model_path results/konkani_asr.nemo \
  --output_path models/konkani_asr.onnx
```

### Web API Deployment
```bash
# Start inference server
python scripts/inference_server.py \
  --model_path results/konkani_asr.nemo \
  --port 8000
```

## 🧪 Quick Test with Minimal Data

**Don't have much Konkani speech data yet?** Test the approach with just 1 minute:

```bash
# Record 1 minute of Konkani speech
# Then run:
python scripts/minimal_test.py \
  --audio_file your_1_minute_konkani.wav \
  --transcript "exact text you spoke in Konkani"

# This will:
# ✅ Create train/val/test splits from your 1 minute
# ✅ Fine-tune for 3 epochs (5-15 minutes)
# ✅ Test if the model learned your speech patterns
# ✅ Show transcription accuracy metrics
```

**Expected Results:**
- Training data: 90%+ accuracy (model memorizes)
- Unseen data: 50-80% accuracy (shows generalization)
- If successful: ✅ Proceed with collecting more data!

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📚 References

- [AI4Bharat IndicConformer Model](https://huggingface.co/ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large)
- [NVIDIA NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)
- [NeMo ASR Tutorials](https://github.com/NVIDIA/NeMo/tree/main/tutorials/asr)
- [Indic ASR Research](https://ai4bharat.org/)

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- AI4Bharat for the IndicConformer model
- NVIDIA for the NeMo framework
- Konkani language community

---

*Advancing Konkani language technology through speech recognition* 🗣️