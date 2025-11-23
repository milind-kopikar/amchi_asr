# Konkani ASR Fine-tuning Project

This project provides a complete pipeline for fine-tuning the AI4Bharat IndicConformer Marathi ASR model on Konkani speech data using NVIDIA NeMo.

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone or navigate to the project directory
cd konkani_asr

# Install dependencies
pip install -r requirements.txt

# Verify setup
python scripts/setup_environment.py
```

### 2. Download Base Model
```bash
python scripts/download_model.py
```

### 3. Prepare Your Data
```bash
# Place your audio files in a directory (e.g., my_audio/)
# Place corresponding transcripts in another directory (e.g., my_transcripts/)
# Audio files should be named: audio_001.wav, audio_002.wav, etc.
# Transcript files should be named: audio_001.txt, audio_002.txt, etc.

python run_pipeline.py --audio_dir my_audio --transcript_dir my_transcripts
```

### 4. Fine-tune the Model
```bash
python run_pipeline.py --skip_setup --skip_download
# Or run training directly:
python scripts/fine_tune.py --config configs/konkani_finetune.yaml
```

### 5. Evaluate the Model
```bash
python run_pipeline.py --skip_setup --skip_download --skip_training
# Or run evaluation directly:
python scripts/evaluate.py --model_path results/konkani_asr_final.nemo --test_manifest data/test.tsv
```

### 6. Use the Model for Inference
```bash
# Transcribe a single audio file
python scripts/infer.py --model_path results/konkani_asr_final.nemo --audio_file my_audio/test.wav

# Transcribe all files in a directory
python scripts/infer.py --model_path results/konkani_asr_final.nemo --audio_dir my_audio --output_file transcriptions.json
```

## 📁 Project Structure

```
konkani_asr/
├── configs/                 # NeMo configuration files
│   ├── base_config.yaml    # Base configuration
│   └── konkani_finetune.yaml # Konkani-specific config
├── data/                   # Training data and manifests
│   └── README.md          # Data format specifications
├── results/                # Model checkpoints and logs
├── scripts/                # Python scripts
│   ├── download_model.py  # Download base model
│   ├── prepare_data.py    # Data preparation
│   ├── fine_tune.py       # Model training
│   ├── evaluate.py        # Model evaluation
│   ├── infer.py          # Inference script
│   └── setup_environment.py # Environment setup
├── run_pipeline.py        # Main pipeline script
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .gitignore           # Git ignore rules
```

## 📋 Data Format

### Audio Files
- **Format**: WAV, FLAC, MP3, M4A
- **Sample Rate**: 16kHz (automatically resampled if different)
- **Channels**: Mono
- **Duration**: 1-30 seconds recommended

### Transcript Files
- **Format**: UTF-8 text files
- **Naming**: Same as audio files but with .txt extension
- **Content**: Plain text transcriptions in Konkani script
- **Cleaning**: Automatic normalization applied

### Manifest Format
The scripts generate TSV manifests with columns:
- `audio_filepath`: Path to audio file
- `text`: Transcription text
- `duration`: Audio duration in seconds

## ⚙️ Configuration

### Training Configuration
Key parameters in `configs/konkani_finetune.yaml`:
- `model`: Model architecture settings
- `trainer`: Training hyperparameters
- `exp_manager`: Experiment management
- `optim`: Optimizer settings

### Environment Variables
- `HF_TOKEN`: Hugging Face API token (for model download)
- `CUDA_VISIBLE_DEVICES`: GPU device selection

## 🔧 Customization

### Changing Model Architecture
Edit `configs/konkani_finetune.yaml`:
```yaml
model:
  encoder:
    n_layers: 12  # Adjust depth
  decoder:
    vocabulary: ["अ", "आ", "इ", ...]  # Konkani characters
```

### Training Hyperparameters
```yaml
trainer:
  max_epochs: 50
  accumulate_grad_batches: 4

optim:
  lr: 0.001
  weight_decay: 0.001
```

### Data Preparation Options
```bash
python scripts/prepare_data.py \
  --audio_dir my_audio \
  --transcript_dir my_transcripts \
  --output_dir data \
  --min_duration 1.0 \
  --max_duration 30.0 \
  --val_split 0.1 \
  --test_split 0.1
```

## 📊 Evaluation Metrics

The evaluation script calculates:
- **WER** (Word Error Rate): Word-level accuracy
- **CER** (Character Error Rate): Character-level accuracy
- **BLEU Score**: Translation quality metric

Results are saved to `evaluation_results.json`.

## 🚀 Deployment

### For Inference
```python
from scripts.infer import load_model, transcribe_audio

# Load model
model = load_model("results/konkani_asr_final.nemo")

# Transcribe audio
transcription = transcribe_audio(model, "audio.wav")
print(transcription)
```

### Model Export
The trained model is saved as a `.nemo` file which can be:
- Used directly with NeMo
- Deployed to NVIDIA Triton
- Converted to ONNX for other frameworks

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in config
   - Use gradient accumulation
   - Use smaller model

2. **Import Errors**
   - Install dependencies: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

3. **Model Download Issues**
   - Set HF_TOKEN environment variable
   - Check internet connection
   - Verify Hugging Face access

4. **Audio Processing Errors**
   - Check audio format compatibility
   - Verify sample rates
   - Ensure mono channel audio

### Getting Help

- Check the logs in `results/logs/`
- Review `evaluation_results.json` for metrics
- Validate data format with `data/README.md`

## 📈 Performance Tips

1. **GPU Usage**: Use CUDA-compatible GPU for faster training
2. **Data Quality**: Clean, diverse audio data improves results
3. **Model Size**: Balance between accuracy and inference speed
4. **Training Time**: 10-50 epochs typically sufficient for fine-tuning

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper documentation
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- AI4Bharat for the IndicConformer model
- NVIDIA for the NeMo toolkit
- The Konkani language community

## 🧪 Testing with Minimal Data

### Quick Proof-of-Concept Test

If you only have 1 minute of Konkani speech, you can still test the fine-tuning approach:

```bash
# 1. Record 1 minute of Konkani speech
# 2. Create test data
python scripts/minimal_test.py \
  --audio_file your_konkani_audio.wav \
  --transcript "exact transcript of what you spoke"

# 3. This will:
#    - Split your 1 minute into train/val/test
#    - Fine-tune for just 3 epochs
#    - Test transcription accuracy
```

### What to Expect

**With 1 minute of data:**
- **Training data transcription**: Should be 90%+ accurate (model memorizes)
- **Unseen data transcription**: May be 50-80% accurate (shows generalization)
- **Training time**: 5-15 minutes on GPU

**Success indicators:**
- ✅ Model can transcribe the exact training audio perfectly
- ✅ Some Konkani words/phrases are recognized in new audio
- ✅ WER improves from base Marathi model

### Progressive Validation

Track improvement during training:

```bash
# Validate model at different checkpoints
python scripts/validate_model.py \
  --checkpoints_dir results/checkpoints \
  --base_model models/base_indicconformer.nemo \
  --test_audio_dir data/audio \
  --transcript_dir data/transcripts
```

This creates comparison charts showing how accuracy improves with training.