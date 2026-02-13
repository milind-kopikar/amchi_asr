# Konkani ASR - RunPod Quick Start

**Last Updated**: December 22, 2025  
**Goal**: Fine-tune AI4Bharat Goan Konkani model for Amchi Konkani

---

## 🚀 Quick Command Reference

### 1️⃣ Initial Setup (Run Once)

```bash
# Clone repository
cd /workspace
git clone https://github.com/milind-kopikar/amchi_asr.git
cd amchi_asr

# Use Python 3.11 and upstream NeMo (see SETUP_ENV.md). Run setup with USE_UPSTREAM_NEMO=1.
USE_UPSTREAM_NEMO=1 bash setup_env.sh
# If using a venv (recommended):
# python3.11 -m venv venv_py311 && source venv_py311/bin/activate
# USE_UPSTREAM_NEMO=1 bash setup_env.sh
source venv_py311/bin/activate 2>/dev/null || true

# Verify GPU
nvidia-smi
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"
```

---

### 2️⃣ Download Konkani Base Model

```bash
# Option A: Using shorthand (RECOMMENDED)
python scripts/download_model.py --model konkani

# Option B: Explicit model name
python scripts/download_model.py \
    --model_name ai4bharat/indicconformer_stt_kok_hybrid_ctc_rnnt_large \
    --output_path models/indicconformer_kok

# Verify download (should be ~500MB)
ls -lh models/indicconformer_kok/*.nemo
```

**Expected file**: `indicconformer_stt_kok_hybrid_ctc_rnnt_large.nemo`

---

### 3️⃣ Download Training Data

```bash
# Download from Railway (70/15/15 train/dev/test split, randomized with seed=42)
python scripts/download_data_from_railway.py \
    --base_url https://konkanicollector-production.up.railway.app \
    --output_dir data \
    --train_split 0.7 \
    --dev_split 0.15 \
    --test_split 0.15 \
    --seed 42

# Verify
echo "Train samples: $(wc -l < data/train/manifest.jsonl)"
echo "Dev samples: $(wc -l < data/dev/manifest.jsonl)"
echo "Test samples: $(wc -l < data/test/manifest.jsonl)"
cat data/train/manifest.jsonl | head -2
```

**Expected output**: ~175+ train, ~37+ dev, ~37+ test samples (250+ total)

**Important**: The script randomizes recordings (seed=42) to **mix speaker voices** across splits, preventing speaker bias!

---

### 4️⃣ Train Model (MAIN TASK)

```bash
# Full fine-tuning (all layers trainable)
python scripts/nemo_train.py \
    --config configs/konkani_finetune.yaml \
    --model konkani \
    --output_dir results/konkani_full \
    --max_epochs 50

# OR: Fast training (encoder frozen, 30% faster)
python scripts/nemo_train.py \
    --config configs/konkani_finetune.yaml \
    --model konkani \
    --freeze_encoder \
    --output_dir results/konkani_frozen \
    --max_epochs 30
```

**Training time**: ~12-15 minutes on RTX 4090

---

### 5️⃣ Monitor Training (Optional - Run in Separate Terminal)

```bash
# Start TensorBoard
tensorboard --logdir results/ --port 6006

# Watch logs
tail -f results/konkani_full/nemo_log_globalrank-0_localrank-0.txt

# Or watch validation WER
watch -n 5 "tail -20 results/konkani_full/nemo_log_globalrank-0_localrank-0.txt | grep val_wer"
```

---

### 6️⃣ Evaluate Model

```bash
# Run comprehensive test
python scripts/nemo_test.py \
    --model results/konkani_full/konkani_asr_final.nemo \
    --manifest data/dev/manifest.jsonl \
    --model_type konkani \
    --batch_size 8 \
    --output_dir test_results/konkani

# View results
cat test_results/konkani/test_results.json | grep -E '"wer"|"cer"'
```

---

### 7️⃣ Compare with Marathi Baseline (Optional)

```bash
# Download Marathi model
python scripts/download_model.py --model marathi

# Train Marathi model
python scripts/nemo_train.py \
    --config configs/konkani_finetune.yaml \
    --model marathi \
    --output_dir results/marathi_full \
    --max_epochs 50

# Test Marathi model
python scripts/nemo_test.py \
    --model results/marathi_full/marathi_asr_final.nemo \
    --manifest data/dev/manifest.jsonl \
    --model_type marathi \
    --output_dir test_results/marathi

# Compare
echo "=== Marathi Base ==="
cat test_results/marathi/test_results.json | jq '.wer, .cer'
echo "=== Konkani Base ==="
cat test_results/konkani/test_results.json | jq '.wer, .cer'
```

---

## 🐛 Troubleshooting

### ❌ "Model file not found"

```bash
# Make sure you downloaded the model first
python scripts/download_model.py --model konkani
ls -lh models/indicconformer_kok/*.nemo
```

### ❌ "CUDA out of memory"

```bash
# Reduce batch size
python scripts/nemo_train.py \
    --config configs/konkani_finetune.yaml \
    --model konkani \
    --data.train_ds.batch_size 4 \
    --data.validation_ds.batch_size 4
```

### ❌ "llvmlite version error" / "Could not find llvmlite==0.38.1"

Use **Python 3.11** and **upstream** NeMo (not the AI4Bharat fork). If you already use 3.11 and see numba/llvmlite issues, ensure compatible versions:

```bash
pip install --upgrade 'numba>=0.57.0,<0.58' 'llvmlite>=0.40.0,<0.41'
pip install "nemo_toolkit[all]" pynini librosa
```
Then re-apply the conv_asr patch (see SETUP_ENV.md).

### ❌ "Hugging Face authentication error"

```bash
# Login to Hugging Face
huggingface-cli login
# Paste your token from https://huggingface.co/settings/tokens

# Then retry download
python scripts/download_model.py --model konkani
```

### ❌ "No samples in manifest"

```bash
# Check Railway URL is accessible
curl https://konkanicollector-production.up.railway.app/api/recordings/approved | jq 'length'

# Re-download data
rm -rf data/train data/dev
python scripts/download_data_from_railway.py \
    --base_url https://konkanicollector-production.up.railway.app \
    --output_dir data/train \
    --train_split 0.8
```

---

### SSH & VS Code Remote-SSH (Windows)

When connecting from **Windows** using VS Code Remote-SSH, add a Host entry to your local `~/.ssh/config` with a Windows-style IdentityFile path (use double quotes as shown):

```ssh
Host runpod-<alias>
  HostName <IP_ADDRESS>
  Port <PORT>
  User root
  IdentityFile "C:/Users/Milind Kopikare/.ssh/runpod_ed25519"
  IdentitiesOnly yes
```

Then in VS Code: **Remote-SSH → Connect to Host...** → select `runpod-<alias>`.

For more details see [RUNPOD_DEPLOYMENT_CHECKLIST.md](RUNPOD_DEPLOYMENT_CHECKLIST.md) and `runpod/README.dev.md`.

---

## 💾 Download Trained Model to Local Machine

```bash
# On RunPod, find best checkpoint
ls -lht results/konkani_full/checkpoints/ | head -5

# On your local Windows machine (PowerShell):
# Get SSH connection details from RunPod dashboard
scp -P <pod-ssh-port> root@<pod-ip>:/workspace/amchi_asr/konkani_asr/results/konkani_full/konkani_asr_final.nemo C:\Users\Milind~1\Downloads\

# Or download entire results folder
scp -r -P <pod-ssh-port> root@<pod-ip>:/workspace/amchi_asr/konkani_asr/results/konkani_full C:\Users\Milind~1\Downloads\
```

---

## 📊 Expected Results

### Baseline (No Fine-tuning)
- **Konkani model on Amchi Konkani**: Unknown (will be measured)

### After Fine-tuning (Target)
- **Validation WER**: < 65% (better than Marathi baseline)
- **Test WER**: < 70%
- **Test CER**: < 30%

### Previous Results (Marathi baseline - Dec 18)
- **Validation WER**: 65.38%
- **Test WER**: 87.36%

---

## ⏱️ Time & Cost Estimates

| Task | Time | Cost (RTX 4090 @$0.69/hr) |
|------|------|---------------------------|
| Setup | 10 min | $0.12 |
| Model download | 5 min | $0.06 |
| Data download | 2 min | $0.02 |
| **Training (50 epochs)** | **12-15 min** | **$0.14-0.17** |
| Evaluation | 2 min | $0.02 |
| **TOTAL** | **~30-35 min** | **~$0.35-0.40** |

**Plus**: Comparison with Marathi (~15 min more) = **Total: $0.50-0.60**

---

## 📝 Commands in Order (Copy-Paste Friendly)

```bash
# Full workflow start-to-finish (Python 3.11 + upstream NeMo)
cd /workspace && \
git clone https://github.com/milind-kopikar/amchi_asr.git && \
cd amchi_asr && \
USE_UPSTREAM_NEMO=1 bash setup_env.sh && \
source venv_py311/bin/activate && \
python scripts/download_model.py --model konkani && \
python scripts/download_data_from_railway.py \
  --base_url https://konkanicollector-production.up.railway.app \
  --output_dir data \
  --train_split 0.7 \
  --dev_split 0.15 \
  --test_split 0.15 \
  --seed 42 && \
python scripts/nemo_train.py \
  --config configs/konkani_finetune.yaml \
  --model konkani \
  --output_dir results/konkani_full \
  --max_epochs 50 && \
python scripts/nemo_test.py \
  --model results/konkani_full/konkani_asr_final.nemo \
  --manifest data/dev/manifest.jsonl \
  --model_type konkani \
  --output_dir test_results/konkani && \
cat test_results/konkani/test_results.json
```

---

## 🎯 Success Criteria

✅ Training completes without errors  
✅ Validation WER < 65%  
✅ Model generates Devanagari transcriptions  
✅ Best checkpoint saved in `results/konkani_full/checkpoints/`

---

**Next**: Document results in `TRAINING_RESULTS_KONKANI_2025-12-22.md`
