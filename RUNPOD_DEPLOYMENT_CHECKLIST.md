# 🚀 RunPod Deployment Checklist

**Pre-deployment verification completed on Windows** ✅

## ✅ Local Testing (Windows) - COMPLETE

- [x] Data downloaded from Railway (44 samples)
- [x] 3-way split created: train (30) / dev (6) / test (8)
- [x] Random shuffling applied (seed=42) - speakers mixed
- [x] Manifests verified with Devanagari text
- [x] Config loads successfully (OmegaConf)
- [x] All Python imports working (except NeMo - Linux only)
- [x] Modular scripts created:
  - [x] `nemo_train.py` - Training module
  - [x] `nemo_validate.py` - Validation module
  - [x] `nemo_test.py` - Testing with WER analysis
  - [x] `download_data_from_railway.py` - Data fetcher
- [x] All changes committed and pushed to GitHub (commit 680a035)

## 🎯 RunPod Deployment Steps

### 1. Deploy RunPod Pod

**Pod Configuration:**
- Template: PyTorch 2.0.1
- GPU: RTX 4090 (24GB VRAM) - $0.69/hour
- Storage: 50GB container disk
- Region: Choose closest/cheapest

**Expected cost: ~$2-4 for full experiment (2-4 hours training)**

### 2. Initial Setup (via SSH or Web Terminal)

```bash
# Connect via SSH
ssh root@<runpod-host> -p <port>

# Clone repository
cd /workspace
git clone https://github.com/moksh-kopikar/amchi_asr.git
cd amchi_asr

# Run setup (Python 3.11 + upstream NeMo). See SETUP_ENV.md.
USE_UPSTREAM_NEMO=1 sudo bash setup_env.sh
# If using venv: python3.11 -m venv venv_py311 && source venv_py311/bin/activate && USE_UPSTREAM_NEMO=1 bash setup_env.sh
```

**Note (Windows / VS Code users):** add a Host entry to your local `~/.ssh/config` and use that alias in VS Code Remote-SSH. Example (Windows path format is important):

```ssh
Host runpod-<alias>
  HostName <runpod-ip>
  Port <port>
  User root
  IdentityFile "C:/Users/Milind Kopikare/.ssh/runpod_ed25519"
  IdentitiesOnly yes
```

Then in VS Code: **Remote-SSH → Connect to Host...** → select `runpod-<alias>`.


**Expected output:**
```
✓ Python virtual environment created
✓ PyTorch with CUDA installed
✓ NeMo toolkit installed
✓ Data directories created
```

### 3. Download Models (Choose ONE)

#### Option A: Marathi Base Model (Default)
```bash
# Set HF_TOKEN or run: huggingface-cli login
python scripts/download_model_from_hf.py --repo ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large --outdir models
```

#### Option B: Konkani Base Model
```bash
python scripts/download_model.py --model konkani
# Or: python scripts/download_model_from_hf.py --repo ai4bharat/indicconformer_stt_kok_hybrid_ctc_rnnt_large --outdir models
```

**Tip: Test both models to compare WER!**

### 4. Download Training Data

```bash
# Download and create train/dev/test splits (with shuffling)
python scripts/download_data_from_railway.py \
  --output_dir data \
  --train_split 0.7 \
  --dev_split 0.15 \
  --test_split 0.15 \
  --seed 42
```

**Expected output:**
```
✓ 30 train samples downloaded
✓ 6 dev samples downloaded  
✓ 8 test samples downloaded
```

### 5. Verify Setup

```bash
# Check GPU
nvidia-smi

# Verify data
ls -lh data/train/audio/*.wav | head -5
cat data/train/manifest.jsonl | head -3

# Test config
python -c "from omegaconf import OmegaConf; config = OmegaConf.load('configs/konkani_finetune.yaml'); print('Config OK')"
```

### 6. Train Model

#### Quick Test (Marathi, Frozen Encoder - ~1-2 hours)
```bash
python scripts/nemo_train.py \
  --config configs/konkani_finetune.yaml \
  --model marathi \
  --freeze_encoder \
  --output_dir results/marathi_frozen
```

#### Full Training (Marathi - ~2-4 hours)
```bash
python scripts/nemo_train.py \
  --config configs/konkani_finetune.yaml \
  --model marathi \
  --output_dir results/marathi_full
```

#### Konkani Base Model
```bash
python scripts/nemo_train.py \
  --config configs/konkani_finetune.yaml \
  --model konkani \
  --output_dir results/konkani_full
```

**Monitor in separate terminal:**
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor TensorBoard (if needed)
tensorboard --logdir results/logs --host 0.0.0.0 --port 6006
```

### 7. Validate Model

```bash
# Validate on dev set
python scripts/nemo_validate.py \
  --model results/marathi_full/marathi_asr_final.nemo \
  --manifest data/dev/manifest.jsonl \
  --model_type marathi
```

### 8. Test and Calculate WER

```bash
# Comprehensive test with error analysis
python scripts/nemo_test.py \
  --model results/marathi_full/marathi_asr_final.nemo \
  --manifest data/test/manifest.jsonl \
  --model_type marathi \
  --output_dir test_results/marathi
```

**Expected output:**
```
📈 Test Results
   Total samples: 8
   Word Error Rate (WER): XX.XX%
   Character Error Rate (CER): XX.XX%
   
💾 Detailed results saved: test_results/test_results.json
💾 Worst samples saved: test_results/worst_samples.json
```

### 9. Download Results to Local Machine

**On your Windows machine:**
```powershell
# Create local results directory
mkdir local_results

# Download trained models
scp -P <port> root@<runpod-host>:/workspace/konkani_asr/results/marathi_full/*.nemo ./local_results/

# Download test results
scp -P <port> -r root@<runpod-host>:/workspace/konkani_asr/test_results ./local_results/

# Download checkpoints (optional)
scp -P <port> -r root@<runpod-host>:/workspace/konkani_asr/results/marathi_full/checkpoints ./local_results/
```

### 10. Stop RunPod Pod

**⚠️ IMPORTANT: Stop pod immediately after downloading to halt billing!**

```bash
# On RunPod dashboard
Click "Stop" on your pod
```

## 🔬 Recommended Experiment Plan

### Experiment 1: Marathi Base (Frozen Encoder)
- **Purpose:** Quick baseline, less likely to overfit
- **Time:** ~1-2 hours
- **Cost:** ~$1-2
- **Command:** 
  ```bash
  python scripts/nemo_train.py --config configs/konkani_finetune.yaml --model marathi --freeze_encoder --output_dir results/exp1_marathi_frozen
  ```

### Experiment 2: Marathi Base (Full Fine-tuning)
- **Purpose:** Maximum adaptation to Konkani
- **Time:** ~2-4 hours
- **Cost:** ~$2-3
- **Command:**
  ```bash
  python scripts/nemo_train.py --config configs/konkani_finetune.yaml --model marathi --output_dir results/exp2_marathi_full
  ```

### Experiment 3: Konkani Base (Full Fine-tuning)
- **Purpose:** Test if Konkani base is closer to target
- **Time:** ~2-4 hours
- **Cost:** ~$2-3
- **Command:**
  ```bash
  python scripts/nemo_train.py --config configs/konkani_finetune.yaml --model konkani --output_dir results/exp3_konkani_full
  ```

### Compare Results
```bash
# Test all three
for exp in exp1_marathi_frozen exp2_marathi_full exp3_konkani_full; do
  python scripts/nemo_test.py \
    --model results/$exp/*.nemo \
    --manifest data/test/manifest.jsonl \
    --output_dir test_results/$exp
done

# Compare WER
grep '"wer"' test_results/*/test_results.json
```

## 📊 Success Metrics

- **Baseline WER:** ~50-60% (untrained on Konkani)
- **Target WER:** <30% (good performance)
- **Excellent WER:** <20% (production-ready)

## 🐛 Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size in config
# Edit configs/konkani_finetune.yaml
batch_size: 4  # or even 2
```

### Model Download Fails
```bash
# Check internet connection
ping huggingface.co

# Retry with explicit model name
huggingface-cli download ai4bharat/indicconformer_stt_mr_hybrid_rnnt_large
```

### SSH Connection Lost
- RunPod keeps running even if SSH disconnects
- Reconnect using same SSH command
- Check training status: `tail -f results/logs/*/events.out.tfevents.*`

## 📝 Notes

- All scripts have comprehensive logging with emoji icons
- Look for ✓ (success), ⚠️ (warning), ❌ (error)
- Use `python <script> --help` for all options
- GitHub Copilot works via VS Code Remote SSH!

## ✅ Ready for Deployment!

All local testing complete. You can now:
1. Deploy RunPod pod
2. Follow steps 1-10 above
3. Get your trained Konkani ASR model!

**Estimated total time:** 3-5 hours  
**Estimated total cost:** $2-4 on RunPod

Good luck! 🚀
