# Konkani ASR Model Fine-tuning Plan

**Date**: December 22, 2025  
**Goal**: Fine-tune AI4Bharat **Goan Konkani** model for Amchi Konkani  
**Previous**: Successfully fine-tuned **Marathi** model (WER: 65.38% val, 87.36% test)

---

## 🎯 Objective

Compare performance between two base models:
1. **Marathi base** (`ai4bharat/indicconformer_stt_mr_hybrid_rnnt_large`) - Already tested
2. **Konkani base** (`ai4bharat/indicconformer_stt_kok_hybrid_ctc_rnnt_large`) - **NEW**

**Hypothesis**: Goan Konkani model may perform better since it's linguistically closer to Amchi Konkani.

---

## ✅ What's Already Done

### Code Infrastructure
- ✅ Modular training script ([nemo_train.py](scripts/nemo_train.py)) with `--model konkani` support
- ✅ Data download from Railway ([download_data_from_railway.py](scripts/download_data_from_railway.py))
- ✅ Model download script ([download_model.py](scripts/download_model.py))
- ✅ Configuration file ([konkani_finetune.yaml](configs/konkani_finetune.yaml))
- ✅ Validation ([nemo_validate.py](scripts/nemo_validate.py)) and testing ([nemo_test.py](scripts/nemo_test.py)) modules

### Environment Setup Documentation
- ✅ Python 3.9 requirement documented
- ✅ CUDA 12.4 compatibility fixes documented
- ✅ AI4Bharat NeMo fork installation guide ([AI4BHARAT_SETUP_GUIDE.md](AI4BHARAT_SETUP_GUIDE.md))
- ✅ RunPod deployment checklist ([RUNPOD_DEPLOYMENT_CHECKLIST.md](RUNPOD_DEPLOYMENT_CHECKLIST.md))

### Training Data
- ✅ **250+ Konkani recordings** available on Railway (6x more than previous!)
- ✅ Train/dev/test split strategy (70/15/15 with random shuffling seed=42)
- ✅ Manifest format compatible with AI4Bharat models
- ✅ Randomization ensures speaker diversity across splits

---

## 🚀 Deployment Steps (RunPod)

### Step 1: Start RunPod Pod

**Recommended Configuration:**
- **GPU**: RTX 4090 (24GB VRAM) - $0.69/hour
- **Template**: PyTorch 2.0.1 (pre-installed CUDA drivers)
- **Storage**: 50GB container disk minimum
- **Estimated Cost**: $2-4 for complete experiment

### Step 2: Clone Repository & Setup Environment

```bash
# SSH into RunPod
ssh root@<pod-ip> -p <port>

# Clone repo
cd /workspace
git clone https://github.com/milind-kopikar/amchi_asr.git
cd amchi_asr/konkani_asr

# Setup environment (automated script)
bash setup_runpod.sh
source venv/bin/activate

# Verify GPU
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Expected Output:**
```
✓ Python 3.9 virtual environment created
✓ PyTorch with CUDA installed
✓ NeMo toolkit installed
✓ Dependencies installed
CUDA: True
```

### Step 3: Download Konkani Base Model

```bash
# Download Goan Konkani model from Hugging Face
python scripts/download_model.py \
    --model konkani \
    --output_dir models/indicconformer_kok

# Verify download
ls -lh models/indicconformer_kok/*.nemo
```

**Expected Files:**
- `indicconformer_stt_kok_hybrid_ctc_rnnt_large.nemo` (~500MB)
- `tokenizer.model`
- `tokenizer.vocab`
- `config.yaml`

### Step 4: Download Training Data from Railway

```bash
# Fetch recordings from Railway PostgreSQL
python scripts/download_data_from_railway.py \
    --base_url https://konkanicollector-production.up.railway.app \
    --output_dir data/train \
    --train_split 0.8

# Verify manifests
echo "Train samples: $(wc -l < data/train/manifest.jsonl)"
echo "Dev samples: $(wc -l < data/dev/manifest.jsonl)"
```

**Expected Output:**
```
✓ Downloaded 44 recordings
✓ Train manifest: data/train/manifest.jsonl (35 samples)
✓ Dev manifest: data/dev/manifest.jsonl (9 samples)
```

### Step 5: Verify Configuration

```bash
# Test config loads without errors
python -c "from omegaconf import OmegaConf; config = OmegaConf.load('configs/konkani_finetune.yaml'); print('✓ Config OK')"

# Quick compatibility check
python scripts/test_compatibility.py
```

### Step 6: Train Konkani Model

```bash
# Full fine-tuning with Konkani base
python scripts/nemo_train.py \
    --config configs/konkani_finetune.yaml \
    --model konkani \
    --output_dir results/konkani_full \
    --max_epochs 50

# OR: Fast training (encoder frozen) - if time/cost constrained
python scripts/nemo_train.py \
    --config configs/konkani_finetune.yaml \
    --model konkani \
    --freeze_encoder \
    --output_dir results/konkani_frozen \
    --max_epochs 30
```

**Training Progress to Monitor:**
- Validation WER dropping below 70% by epoch 30-40
- No significant overfitting (train WER >> val WER)
- GPU utilization ~70-90%

**Expected Training Time:**
- RTX 4090: ~12-15 minutes (50 epochs)
- A100: ~8-10 minutes (50 epochs)

### Step 7: Evaluate Model

```bash
# Run comprehensive test with error analysis
python scripts/nemo_test.py \
    --model results/konkani_full/konkani_asr_final.nemo \
    --manifest data/dev/manifest.jsonl \
    --model_type konkani \
    --batch_size 8 \
    --output_dir test_results/konkani

# View results
cat test_results/konkani/test_results.json
```

**Key Metrics to Compare:**
- **Word Error Rate (WER)**: Target < 65% (better than Marathi)
- **Character Error Rate (CER)**: Target < 30%
- **Substitutions/Deletions/Insertions**: Error pattern analysis

### Step 8: Compare with Marathi Baseline

```bash
# Download Marathi model (if not already trained)
python scripts/download_model.py \
    --model marathi \
    --output_dir models/indicconformer_mr

# Train Marathi model for comparison
python scripts/nemo_train.py \
    --config configs/konkani_finetune.yaml \
    --model marathi \
    --output_dir results/marathi_full \
    --max_epochs 50

# Test both models on same data
python scripts/nemo_test.py \
    --model results/marathi_full/marathi_asr_final.nemo \
    --manifest data/dev/manifest.jsonl \
    --model_type marathi \
    --output_dir test_results/marathi

# Compare results
echo "=== Marathi Base ==="
cat test_results/marathi/test_results.json | grep '"wer"'
echo "=== Konkani Base ==="
cat test_results/konkani/test_results.json | grep '"wer"'
```

---

## 📊 Expected Results

### Baseline Performance (No Fine-tuning)
- **Marathi model on Konkani**: WER ~92% (from previous test with 44 samples)
- **Konkani model on Amchi Konkani**: **UNKNOWN** - to be measured

### After Fine-tuning (Prediction)
- **Marathi base (Dec 18, 44 samples)**: WER ~65-70% validation, ~87% test (proven)
- **Konkani base (Dec 22, 250+ samples)**: WER ~**40-55%** (hypothesis: better model + 6x more data)

### Impact of 6x More Data
With 250+ recordings vs 44:
- **More speaker diversity** → Better generalization
- **More sentence variety** → Better vocabulary coverage  
- **More training examples** → Lower overfitting, better convergence
- **Expected WER improvement**: 15-25 percentage points lower than previous

### Success Criteria
✅ Konkani base model achieves **WER < 55%** (significant improvement over Marathi baseline)  
✅ Training completes without errors  
✅ Model generates intelligible Konkani transcriptions  
✅ Error analysis shows improvement in Konkani-specific phonemes
✅ No significant overfitting (train/val WER gap < 10 points)

---

## 🐛 Known Issues & Solutions

### Issue 1: Python Version
**Problem**: `llvmlite==0.38.1` not available for Python 3.10+  
**Solution**: Use Python 3.9 (enforced in `setup_runpod.sh`)

### Issue 2: CUDA 12.4 Compatibility
**Problem**: Default `numba==0.55.2` incompatible with CUDA 12.4  
**Solution**: Upgrade to `numba==0.57.1` and `llvmlite==0.40.1`

```bash
pip install --upgrade 'numba>=0.57.0,<0.58' 'llvmlite>=0.40.0,<0.41'
```

### Issue 3: Manifest Format
**Problem**: Missing `lang` and `sample_id` fields  
**Solution**: `download_data_from_railway.py` automatically adds these fields

### Issue 4: Scheduler Name
**Problem**: Config uses `name: "cosine"` (invalid)  
**Solution**: Use `name: "CosineAnnealing"` (already in config)

### Issue 5: Out of Memory
**Problem**: Batch size too large for GPU  
**Solution**: Reduce `data.train_ds.batch_size` from 8 to 4 in config

```bash
# Quick fix
python scripts/nemo_train.py \
    --config configs/konkani_finetune.yaml \
    --model konkani \
    --data.train_ds.batch_size 4 \
    --data.validation_ds.batch_size 4
```

---

## 📝 Experiment Tracking

### Training Log Template

```markdown
## Konkani Base Model Training

**Date**: YYYY-MM-DD
**Environment**: RunPod RTX 4090, CUDA 12.4
**Base Model**: ai4bharat/indicconformer_stt_kok_hybrid_ctc_rnnt_large
**Training Data**: XX samples (train), XX (dev), XX (test)

### Configuration
- Optimizer: AdamW, LR 0.0001
- Scheduler: CosineAnnealing (warmup 1000 steps)
- Batch size: 8
- Max epochs: 50
- Frozen encoder: No

### Results
- Best validation WER: XX.XX% (epoch XX)
- Test WER: XX.XX%
- Test CER: XX.XX%
- Training time: XX minutes
- Total cost: $X.XX

### Comparison with Marathi Base
| Metric | Marathi Base | Konkani Base | Improvement |
|--------|--------------|--------------|-------------|
| Val WER | 65.38% | XX.XX% | ±XX.XX pp |
| Test WER | 87.36% | XX.XX% | ±XX.XX pp |
| Test CER | 12.34% | XX.XX% | ±XX.XX pp |

### Observations
- [Add qualitative observations here]
- [Error patterns]
- [Surprising results]
```

---

## 🎯 Next Steps After Training

1. **Document Results**: Update `TRAINING_RESULTS_KONKANI_2025-12-22.md`
2. **Download Model**: SCP best checkpoint to local machine
3. **Science Fair Demo**: Test on new recordings
4. **Compare Models**: Create comparison table (Marathi vs Konkani base)
5. **Deploy**: If Konkani performs better, use it for production

---

## 💡 Key Insights from Previous Training

### What Worked Well
- ✅ Random shuffling (seed=42) prevented speaker bias
- ✅ 80/20 train/dev split worked well for small dataset
- ✅ CosineAnnealing scheduler converged smoothly
- ✅ Validation WER as early stopping metric

### What to Watch
- ⚠️ With only 44 samples, model may overfit after 30-40 epochs
- ⚠️ Test WER (87%) much higher than val WER (65%) - may indicate test set difficulty
- ⚠️ Character error patterns: "सिंहु" → "शिव" (phonetically similar but wrong)

### Optimization Tips
- If budget-constrained: use `--freeze_encoder` flag (faster, cheaper)
- If WER plateaus: try reducing learning rate by 10x
- If overfitting: add more data from Railway or increase dropout

---

## 📚 References

- [AI4BHARAT_SETUP_GUIDE.md](AI4BHARAT_SETUP_GUIDE.md) - Environment setup
- [NEMO_WORKFLOW_GUIDE.md](NEMO_WORKFLOW_GUIDE.md) - Complete workflow
- [RUNPOD_SETUP.md](RUNPOD_SETUP.md) - RunPod deployment
- [TRAINING_RESULTS_2025-12-18.md](TRAINING_RESULTS_2025-12-18.md) - Previous Marathi results

---

**Status**: Ready to deploy on RunPod ✅  
**Estimated Time**: 1-2 hours (setup + training + evaluation)  
**Estimated Cost**: $2-4
