# NeMo ASR Training Workflow Guide

Complete guide for training Konkani ASR models using AI4Bharat IndicConformer base models.

## 📋 Overview

This project supports fine-tuning two AI4Bharat base models:
- **Marathi ASR** (`ai4bharat/indicconformer_stt_mr_hybrid_rnnt_large`) - Default
- **Goan Konkani ASR** (`ai4bharat/indicconformer_stt_kok_hybrid_ctc_rnnt_large`) - May be closer to Amchi Konkani

## 🏗️ Modular Architecture

### Core Modules

1. **`nemo_train.py`** - Training module
   - Supports model switching (Marathi/Konkani)
   - Freeze encoder option for faster training
   - Comprehensive logging and checkpointing
   
2. **`nemo_validate.py`** - Validation module
   - Compute WER on validation sets
   - Transcribe audio files
   - Support for both .nemo and .ckpt formats
   
3. **`nemo_test.py`** - Testing module
   - Detailed WER calculation with error analysis
   - Character Error Rate (CER)
   - Save worst samples for inspection

4. **`download_data_from_railway.py`** - Data download
   - Fetch approved recordings from Railway
   - Auto-split train/dev sets
   - Create NeMo-compatible manifests

## 🚀 Complete Workflow

### Step 1: Download Data (Local - Windows OK)

```powershell
# Download from Railway (80/20 train/dev split)
python scripts/download_data_from_railway.py --output_dir data/train --train_split 0.8

# Verify manifests created
ls data/train/manifest.jsonl
ls data/dev/manifest.jsonl
ls data/train/audio/*.wav
```

**Expected Output:**
```
✓ Train manifest: data\train\manifest.jsonl (35 samples)
✓ Dev manifest: data\dev\manifest.jsonl (9 samples)
✓ Audio files: data\train\audio\*.wav, data\dev\audio\*.wav
```

### Step 2: Download Base Model (RunPod or Local Linux)

**On RunPod/Linux:**
```bash
# Download Marathi model (default)
python scripts/download_model.py --model marathi --output_dir models/indicconformer_mr

# OR download Konkani model
python scripts/download_model.py --model konkani --output_dir models/indicconformer_kok
```

### Step 3: Test Config Loading (Local - Windows OK)

```powershell
# Test config loads without errors
python -c "from omegaconf import OmegaConf; config = OmegaConf.load('configs/konkani_finetune.yaml'); print('Config OK:', config.trainer.accelerator)"
```

### Step 4: Train Model (RunPod - Linux + GPU Required)

#### Option A: Train with Marathi Base (Default)

```bash
# Full fine-tuning (all layers trainable)
python scripts/nemo_train.py \
  --config configs/konkani_finetune.yaml \
  --model marathi \
  --output_dir results/marathi_full

# Fast training (encoder frozen, recommended for <1hr data)
python scripts/nemo_train.py \
  --config configs/konkani_finetune.yaml \
  --model marathi \
  --freeze_encoder \
  --output_dir results/marathi_frozen
```

#### Option B: Train with Konkani Base

```bash
# Full fine-tuning
python scripts/nemo_train.py \
  --config configs/konkani_finetune.yaml \
  --model konkani \
  --output_dir results/konkani_full

# Fast training
python scripts/nemo_train.py \
  --config configs/konkani_finetune.yaml \
  --model konkani \
  --freeze_encoder \
  --output_dir results/konkani_frozen
```

**Training Output:**
```
🚀 Initializing NeMo ASR Trainer
📦 Selected Model: ai4bharat/indicconformer_stt_mr_hybrid_rnnt_large
✓ Train manifest: data/train/manifest.jsonl
✓ Validation manifest: data/dev/manifest.jsonl
🔧 Loading Pre-trained Model
📊 Model details:
   - Total parameters: 123,456,789
   - Trainable parameters: 123,456,789
⚙️  Setting up Trainer
✓ TensorBoard logging to: results/logs
🎯 Starting Training
Epoch 1/50: 100%|██████████| 5/5 [00:30<00:00, val_wer=0.45]
...
✅ Training Complete!
💾 Final model saved: results/marathi_asr_final.nemo
🏆 Best checkpoint: results/checkpoints/marathi_asr-epoch=12-val_wer=0.285.ckpt
📊 Best WER: 0.2850
```

### Step 5: Validate Model (RunPod or Local with .nemo file)

```bash
# Validate best checkpoint
python scripts/nemo_validate.py \
  --model results/marathi_asr_final.nemo \
  --manifest data/dev/manifest.jsonl \
  --model_type marathi \
  --batch_size 8
```

**Validation Output:**
```
📊 Validation Results
   test_wer: 0.2850
   test_loss: 1.2345
🎯 Word Error Rate (WER): 28.50%
```

### Step 6: Test and Calculate WER (RunPod or Local)

```bash
# Run comprehensive test with error analysis
python scripts/nemo_test.py \
  --model results/marathi_asr_final.nemo \
  --manifest data/dev/manifest.jsonl \
  --model_type marathi \
  --batch_size 8 \
  --output_dir test_results
```

**Test Output:**
```
📈 Test Results
   Total samples: 9
   Word Error Rate (WER): 28.50%
   Character Error Rate (CER): 12.34%
   
   Substitutions: 15
   Deletions: 8
   Insertions: 5
   Hits: 72

🔴 Top 10 Worst Samples (by WER):
1. Sample 3 - WER: 45.2%
   REF: एक घरांतु एकी आज्जी एक्ष्ी राब्ताली
   HYP: एक घरांतु एकी आजी एक राबताली

💾 Detailed results saved: test_results/test_results.json
💾 Worst 50 samples saved: test_results/worst_samples.json
```

### Step 7: Compare Models

Test both Marathi and Konkani base models to see which performs better:

```bash
# Test Marathi-based model
python scripts/nemo_test.py \
  --model results/marathi_full/marathi_asr_final.nemo \
  --manifest data/dev/manifest.jsonl \
  --model_type marathi \
  --output_dir test_results/marathi

# Test Konkani-based model
python scripts/nemo_test.py \
  --model results/konkani_full/konkani_asr_final.nemo \
  --manifest data/dev/manifest.jsonl \
  --model_type konkani \
  --output_dir test_results/konkani

# Compare WER results
echo "Marathi Base WER:"
cat test_results/marathi/test_results.json | grep '"wer"'
echo "Konkani Base WER:"
cat test_results/konkani/test_results.json | grep '"wer"'
```

## 📊 Monitoring Training

### TensorBoard (While training)

```bash
# On RunPod
tensorboard --logdir results/logs --host 0.0.0.0 --port 6006

# Access via RunPod's exposed port or SSH tunnel
```

### nvidia-smi (GPU monitoring)

```bash
# Watch GPU usage
watch -n 1 nvidia-smi
```

## 🔧 Configuration Options

### configs/konkani_finetune.yaml

Key parameters to tune:

```yaml
# Model path (auto-set by --model flag)
model:
  nemo_model: "models/indicconformer_mr/..." # or indicconformer_kok

# Training
trainer:
  max_epochs: 50          # Number of epochs
  accelerator: "gpu"      # GPU on RunPod, CPU on Windows
  devices: 1              # Number of GPUs

# Data
data:
  train_ds:
    batch_size: 8         # Adjust based on GPU memory
    max_duration: 16.7    # Max audio duration in seconds
    min_duration: 0.1     # Min audio duration

# Optimizer
optim:
  lr: 0.0001             # Learning rate
  weight_decay: 0.001     # Regularization
```

## 🐛 Debugging Tips

### Enable Debug Logging

All modules have comprehensive logging. Look for:
- ✓ Green checkmarks = success
- ⚠ Yellow warnings = non-critical issues
- ❌ Red X = errors
- 🎯 🚀 📊 Icons for key milestones

### Common Issues

**1. Model not found**
```
❌ Model file not found: models/indicconformer_mr/...
```
**Fix:** Run `python scripts/download_model.py --model marathi` first

**2. Manifest not found**
```
❌ Train manifest not found: data/train/manifest.jsonl
```
**Fix:** Run `python scripts/download_data_from_railway.py` first

**3. GPU out of memory**
```
RuntimeError: CUDA out of memory
```
**Fix:** Reduce `batch_size` in config (try 4 or 2)

**4. NeMo import error on Windows**
```
ImportError: Cannot import nemo on Windows
```
**Fix:** This is expected. NeMo only works on Linux. Use RunPod for training.

## 📁 Output Files

After training, you'll have:

```
results/
├── marathi_asr_final.nemo          # Final trained model (portable)
├── checkpoints/
│   ├── marathi_asr-epoch=12-val_wer=0.285.ckpt  # Best checkpoint
│   ├── last.ckpt                   # Latest checkpoint
│   └── ...
└── logs/
    └── tensorboard logs

test_results/
├── test_results.json               # All results
└── worst_samples.json              # Worst 50 samples for analysis
```

## 💾 Download Results from RunPod

```bash
# On your local machine (Windows PowerShell)
# Replace <runpod-host> and <port> with your RunPod SSH details

scp -P <port> root@<runpod-host>:/workspace/konkani_asr/results/*.nemo ./local_results/
scp -P <port> -r root@<runpod-host>:/workspace/konkani_asr/test_results ./local_results/
```

## 🎯 Recommended Workflow

1. **Test locally on Windows:**
   - ✅ Download data (works on Windows)
   - ✅ Test config loading (works on Windows)
   
2. **Deploy to RunPod:**
   - 🚀 Clone repo
   - 🚀 Run setup_runpod.sh
   - 🚀 Download model
   - 🚀 Train model
   
3. **Test both models:**
   - 📊 Train with Marathi base
   - 📊 Train with Konkani base
   - 📊 Compare WER results
   - 📊 Choose better model
   
4. **Download and iterate:**
   - 💾 Download best model
   - 🔍 Analyze worst samples
   - 📝 Collect more data for problematic cases
   - 🔁 Repeat training

## 🆚 Model Selection Guide

### When to use Marathi base:
- More training data available (>2 hours)
- Vocabulary similar to Marathi
- Larger model capacity needed

### When to use Konkani base:
- Limited training data (<1 hour)
- Goan Konkani dialect
- Phonetically closer to target domain

**Test both and compare WER!**

## ⏱️ Estimated Times (RunPod RTX 4090)

- Data download: ~5 minutes
- Model download: ~10 minutes
- Training (50 epochs, 35 samples): 2-4 hours
- Validation: <1 minute
- Testing: <1 minute

**Total RunPod cost: ~$2-3 for full experiment**

## 📚 Next Steps

1. Run both Marathi and Konkani experiments
2. Compare WER on dev set
3. Analyze error patterns in worst_samples.json
4. Collect more data for high-error cases
5. Iterate training with more data
6. Test on real-world audio

---

**Ready to start?** → Jump to [Step 1: Download Data](#step-1-download-data-local---windows-ok)
