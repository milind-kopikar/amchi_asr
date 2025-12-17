# Local Testing Guide - Before RunPod Deployment

Before spending money on RunPod GPU, run comprehensive local tests to catch issues early.

## 📋 Two Testing Scripts

### 1️⃣ Pre-Flight Checks (Windows Compatible)

**File:** `tests/test_preflight.py`

**What it tests:**
- ✅ Hugging Face authentication
- ✅ Model downloads (Marathi & Konkani)
- ✅ Test data download from Railway
- ✅ Config file validation

**Run now on Windows:**
```bash
python tests/test_preflight.py
```

**What happens:**
1. Checks if you're authenticated with Hugging Face
2. Downloads both ASR models (~1GB total)
3. Downloads 3 test audio samples from Railway
4. Validates YAML config files
5. Prints comprehensive pre-flight summary

**Output:**
```
🔐 Hugging Face: Authenticated ✅
📦 Marathi Model: Downloaded (0.52 GB) ✅
📦 Konkani Model: Downloaded (0.51 GB) ✅
📊 Test Data: 3 samples downloaded ✅
⚙️  Config Files: Valid ✅

🎉 ALL PRE-FLIGHT CHECKS PASSED!
```

---

### 2️⃣ Model Transcription Tests (RunPod/Linux Only)

**File:** `tests/test_local_models.py`

**What it tests:**
- 🧪 Loads both Marathi & Konkani models
- 🎤 Transcribes 3 test audio samples
- 📊 Compares predictions with ground truth
- 📈 Calculates WER (Word Error Rate) and CER (Character Error Rate)
- 📝 Generates comparison report

**Run on RunPod:**
```bash
# Download models and run tests
python tests/test_local_models.py --download --test

# If models already downloaded
python tests/test_local_models.py --test

# Test only Marathi model
python tests/test_local_models.py --test --models marathi

# Test with 5 samples instead of 3
python tests/test_local_models.py --test --n_samples 5
```

**Output Files:**
```
test_local_results/
├── report.txt          # Human-readable comparison report
├── results.json        # Machine-readable JSON results
└── test_audio/         # Downloaded test samples
    └── test/
        ├── manifest.jsonl
        └── audio/
            ├── sample_001.wav
            ├── sample_002.wav
            └── sample_003.wav
```

**Example Report:**
```
================================================================================
LOCAL ASR MODEL TEST REPORT
================================================================================

SUMMARY
--------------------------------------------------------------------------------

MARATHI Model:
  Samples tested: 3
  Average WER: 65.4%
  Average CER: 42.3%

KONKANI Model:
  Samples tested: 3
  Average WER: 48.2%
  Average CER: 28.7%

================================================================================
DETAILED RESULTS
================================================================================

MARATHI MODEL
--------------------------------------------------------------------------------

Sample 1: recording_123.wav
  Ground Truth: हांव आज घरा वतां
  Prediction:   मी आज घरी जातो
  WER: 100.00% | CER: 78.57%

Sample 2: recording_456.wav
  Ground Truth: तुवें कितें करतात
  Prediction:   तुम्ही काय करत आहात
  WER: 66.67% | CER: 45.45%

Sample 3: recording_789.wav
  Ground Truth: म्हाका भुक लागल्या
  Prediction:   मला भूक लागली आहे
  WER: 50.00% | CER: 38.46%

KONKANI MODEL
--------------------------------------------------------------------------------

Sample 1: recording_123.wav
  Ground Truth: हांव आज घरा वतां
  Prediction:   हांव आज घरा वतां
  WER: 0.00% | CER: 0.00%

Sample 2: recording_456.wav
  Ground Truth: तुवें कितें करतात
  Prediction:   तुवें कितें करता
  WER: 33.33% | CER: 14.29%

Sample 3: recording_789.wav
  Ground Truth: म्हाका भुक लागल्या
  Prediction:   म्हाका भूक लागली
  WER: 33.33% | CER: 15.38%
```

---

## 🎯 Recommended Testing Workflow

### Phase 1: Windows (Local - $0)

```bash
# Step 1: Run pre-flight checks
python tests/test_preflight.py

# Expected output:
# ✅ All authentication, downloads, and configs verified
# ⏱️  Takes: 5-10 minutes (mostly download time)
# 💰 Cost: $0 (runs locally)
```

**If pre-flight fails:**
- Fix authentication: `huggingface-cli login`
- Accept model conditions on Hugging Face
- Check Railway API credentials
- Fix YAML syntax errors

---

### Phase 2: RunPod (Quick Tests - $0.10-0.50)

```bash
# Step 2: Deploy RunPod pod
# (Follow RUNPOD_DEPLOYMENT_CHECKLIST.md)

# Step 3: Clone repo and setup
git clone https://github.com/milind-kopikar/amchi_asr.git
cd amchi_asr
pip install -r requirements.txt

# Step 4: Run model transcription tests
python tests/test_local_models.py --download --test

# Expected output:
# ✅ Both models transcribe 3 samples
# 📊 WER/CER calculated for each model
# 📝 Comparison report generated
# ⏱️  Takes: 5-10 minutes
# 💰 Cost: $0.10-0.15
```

**Review the report:**
```bash
cat test_local_results/report.txt
```

**What to look for:**
- ✅ Models load successfully
- ✅ Transcriptions produce Devanagari text (not gibberish)
- ✅ WER for base models: 40-80% (untrained, expected)
- ✅ Konkani model likely performs better than Marathi

**If tests fail:**
- Check NeMo installation
- Verify GPU is available: `nvidia-smi`
- Check model files exist
- Review error messages in output

---

### Phase 3: RunPod (Smoke Tests - $0.35-0.40)

```bash
# Step 5: Run end-to-end smoke tests
python tests/test_e2e_pipeline.py --model marathi --test all

# Expected output:
# ✅ Test 1: Overfitting test passes
# ✅ Test 2: Before-After improvement shown
# ✅ Test 3: Empty data test validates
# ⏱️  Takes: 30 minutes
# 💰 Cost: $0.35-0.40
```

**If smoke tests fail:**
- Review test_e2e_results/test_summary.json
- Check specific test failures
- Fix issues before full training

---

### Phase 4: RunPod (Full Training - $2-2.50)

```bash
# Step 6: Download full dataset
python scripts/download_data_from_railway.py

# Step 7: Train on full data
python scripts/nemo_train.py \
    --config configs/konkani_finetune.yaml \
    --model marathi

# Expected output:
# ✅ Training completes 50 epochs
# ✅ Checkpoints saved
# ✅ Final WER: <20-30% (target)
# ⏱️  Takes: 3 hours
# 💰 Cost: $2.00-2.50
```

---

## 📊 Expected WER Results

### Base Models (Untrained on Your Data)

| Model | Expected WER | Reason |
|-------|--------------|--------|
| **Marathi** | 60-80% | Different language, similar script |
| **Konkani** | 40-60% | Same language, different speakers |

**This is NORMAL!** Base models aren't trained on your specific speakers/audio.

### After Fine-Tuning (Your Goal)

| Performance | WER Range | Interpretation |
|-------------|-----------|----------------|
| **Excellent** | <10% | Production-ready |
| **Good** | 10-20% | Usable with some errors |
| **Acceptable** | 20-30% | Needs more data/training |
| **Poor** | >30% | Investigate issues |

---

## ⏱️ Time Estimates

| Phase | Duration | Can Skip? |
|-------|----------|-----------|
| Pre-flight (Windows) | 5-10 min | ❌ Required |
| Model tests (RunPod) | 5-10 min | ⚠️  Recommended |
| Smoke tests (RunPod) | 30 min | ⚠️  Recommended |
| Full training (RunPod) | 3 hours | ❌ Required |
| **Total** | **~4 hours** | |

---

## 💰 Cost Breakdown

| Phase | Cost | Value |
|-------|------|-------|
| Pre-flight (Windows) | **$0.00** | Catches auth/config issues |
| Model tests (RunPod) | **$0.10-0.15** | Verifies models work |
| Smoke tests (RunPod) | **$0.35-0.40** | Validates full pipeline |
| Full training (RunPod) | **$2.00-2.50** | Actual fine-tuning |
| **Total (if all passes)** | **$2.50-3.00** | |
| **Savings from testing** | **$0-2.00** | Avoid failed 3-hour runs |

---

## 🚨 Common Issues & Solutions

### Issue 1: Model Download Fails

**Error:**
```
❌ Repository not found
```

**Solution:**
1. Accept model conditions:
   - Marathi: https://huggingface.co/ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large
   - Konkani: https://huggingface.co/ai4bharat/indicconformer_stt_kok_hybrid_ctc_rnnt_large
2. Authenticate: `huggingface-cli login`

---

### Issue 2: NeMo Import Fails on Windows

**Error:**
```
ImportError: No module named 'nemo'
```

**Solution:**
- ✅ This is EXPECTED on Windows
- 🐧 Run transcription tests on RunPod/Linux only
- ✅ Pre-flight checks work on Windows

---

### Issue 3: High WER After Training

**Possible causes:**
- 🎤 Audio quality issues
- 📝 Transcription errors in ground truth
- ⚙️  Hyperparameters need tuning
- 📊 Need more training data

**Solutions:**
1. Review worst samples: `test_results.json`
2. Check audio quality: `verify_data.py`
3. Increase epochs: Edit `configs/konkani_finetune.yaml`
4. Collect more recordings via `konkani_collector`

---

### Issue 4: Out of Memory on RunPod

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
1. Reduce batch size in config:
   ```yaml
   batch_size: 2  # Try 1 if still fails
   ```
2. Use smaller model (not recommended)
3. Use pod with more VRAM

---

## 📈 Interpreting Test Results

### Good Signs ✅

- Models load without errors
- Transcriptions are in Devanagari (not gibberish)
- Konkani model WER < Marathi model WER
- Some samples transcribe perfectly (0% WER)
- Training loss decreases over epochs
- Validation WER improves over epochs

### Warning Signs ⚠️

- Models fail to load
- Transcriptions are empty or ASCII gibberish
- WER >90% on all samples
- Training loss increases
- Validation WER gets worse
- Out of memory errors

---

## 🎯 Decision Points

### After Pre-Flight Checks:

✅ **All passed?** → Proceed to RunPod
❌ **Any failed?** → Fix issues before spending money

### After Model Tests:

✅ **WER 40-80%?** → Normal, proceed to smoke tests
⚠️  **WER >90%?** → Investigate models/data
❌ **Models don't load?** → Check NeMo installation

### After Smoke Tests:

✅ **All 3 tests pass?** → Proceed to full training
❌ **Any test fails?** → Review test details, fix pipeline

### After Full Training:

✅ **WER <30%?** → Success! Deploy model
⚠️  **WER 30-50%?** → Train longer or collect more data
❌ **WER >50%?** → Review data quality, hyperparameters

---

## 🚀 Quick Start Commands

### On Windows (Now):
```bash
# Install dependencies
pip install huggingface_hub omegaconf

# Run pre-flight checks
python tests/test_preflight.py

# Expected: Downloads models, validates everything
```

### On RunPod (Later):
```bash
# Setup
git clone https://github.com/milind-kopikar/amchi_asr.git
cd amchi_asr
pip install -r requirements.txt

# Quick model test
python tests/test_local_models.py --download --test

# Smoke tests
python tests/test_e2e_pipeline.py --test all

# Full training
python scripts/download_data_from_railway.py
python scripts/nemo_train.py --config configs/konkani_finetune.yaml
```

---

## 📚 Related Documentation

- **AI4BHARAT_MODEL_ACCESS.md** - Model authentication & access
- **MODEL_DOWNLOAD_GUIDE.md** - Model download details
- **TESTING_GUIDE.md** - Comprehensive testing documentation
- **RUNPOD_DEPLOYMENT_CHECKLIST.md** - RunPod setup guide
- **NEMO_WORKFLOW_GUIDE.md** - Full training workflow

---

## ✅ Checklist

Before RunPod deployment:

- [ ] Run `python tests/test_preflight.py`
- [ ] Verify all pre-flight checks pass
- [ ] Review downloaded models (both exist)
- [ ] Test audio samples downloaded
- [ ] Configs validated

On RunPod:

- [ ] Run `python tests/test_local_models.py --download --test`
- [ ] Review WER comparison report
- [ ] Konkani model performs better than Marathi
- [ ] Run `python tests/test_e2e_pipeline.py --test all`
- [ ] All 3 smoke tests pass
- [ ] Proceed to full training

After training:

- [ ] WER < 30% achieved
- [ ] Review worst samples
- [ ] Download checkpoints
- [ ] Stop RunPod pod (save money!)

---

**🚀 START NOW: `python tests/test_preflight.py`**
