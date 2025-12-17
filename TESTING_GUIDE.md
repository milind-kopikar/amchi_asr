# 🧪 Testing Guide & Model Switching

## 📋 Model Switching

### Available Models

**1. Marathi Base Model (Default)**
- Model: `ai4bharat/indicconformer_stt_mr_hybrid_rnnt_large`
- Use when: More training data, vocabulary similar to Marathi
- Download: `python scripts/download_model.py --model marathi`

**2. Goan Konkani Base Model**
- Model: `ai4bharat/indicconformer_stt_kok_hybrid_ctc_rnnt_large`
- Use when: Goan Konkani dialect, limited data
- Download: `python scripts/download_model.py --model konkani`

### How to Switch Models

**When Training:**
```bash
# Train with Marathi base
python scripts/nemo_train.py \
  --config configs/konkani_finetune.yaml \
  --model marathi \
  --output_dir results/marathi

# Train with Konkani base
python scripts/nemo_train.py \
  --config configs/konkani_finetune.yaml \
  --model konkani \
  --output_dir results/konkani
```

**When Validating/Testing:**
```bash
# Test Marathi-based model
python scripts/nemo_test.py \
  --model results/marathi/marathi_asr_final.nemo \
  --manifest data/test/manifest.jsonl \
  --model_type marathi

# Test Konkani-based model
python scripts/nemo_test.py \
  --model results/konkani/konkani_asr_final.nemo \
  --manifest data/test/manifest.jsonl \
  --model_type konkani
```

**The `--model_type` flag tells the script which model architecture to use!**

---

## 🧪 Quick Smoke Tests (Before Full Training)

### Test 1: Download Minimal Data

```bash
# Download only 3 train, 1 dev, 1 test samples
python scripts/download_smoke_test_data.py \
  --n_train 3 \
  --n_dev 1 \
  --n_test 1
```

**Output:**
- `data_smoke/train/` - 3 samples
- `data_smoke/dev/` - 1 sample
- `data_smoke/test/` - 1 sample

### Test 2: Quick Training Test (5 Epochs)

```bash
# Train on minimal data (should finish in ~5-10 minutes)
python scripts/nemo_train.py \
  --config configs/konkani_finetune_smoke.yaml \
  --model marathi \
  --freeze_encoder \
  --output_dir results/smoke_test
```

Create `configs/konkani_finetune_smoke.yaml`:
```yaml
# Copy from konkani_finetune.yaml and change:
trainer:
  max_epochs: 5  # Instead of 50

data:
  train_ds:
    manifest_filepath: "data_smoke/train/manifest.jsonl"
    batch_size: 2  # Instead of 8
  validation_ds:
    manifest_filepath: "data_smoke/dev/manifest.jsonl"
    batch_size: 2
```

### Test 3: Quick Validation

```bash
python scripts/nemo_validate.py \
  --model results/smoke_test/marathi_asr_final.nemo \
  --manifest data_smoke/dev/manifest.jsonl \
  --model_type marathi
```

---

## 🔬 End-to-End System Tests

### Run All Tests Automatically

```bash
# Run all 3 E2E tests (Overfitting, Before-After, Empty Data)
python tests/test_e2e_pipeline.py --model marathi --test all
```

**Expected Runtime:** ~30-60 minutes on GPU

### Individual Tests

#### Test 1: Overfitting Test
**Goal:** Train and test on same data → expect WER ≈ 0%

```bash
python tests/test_e2e_pipeline.py --model marathi --test 1
```

**Expected Result:**
```
✅ TEST 1 PASSED
WER on training data: 5.23% (expect <30%)
```

#### Test 2: Before-After Test
**Goal:** WER should improve after fine-tuning

```bash
python tests/test_e2e_pipeline.py --model marathi --test 2
```

**Expected Result:**
```
✅ TEST 2 PASSED
WER BEFORE: 78.45%
WER AFTER:  32.11%
IMPROVEMENT: 46.34% improvement
```

#### Test 3: Empty Data Test
**Goal:** WER should NOT improve with empty training data

```bash
python tests/test_e2e_pipeline.py --model marathi --test 3
```

**Expected Result:**
```
✅ TEST 3 PASSED
WER BEFORE: 78.45%
WER AFTER:  78.45% (should be same or worse)
```

### Test Results Location

After running tests, check:
```
test_e2e_results/
├── test1_overfitting/
│   ├── before/
│   ├── training/
│   └── after/
├── test2_before_after/
│   ├── before/
│   ├── training/
│   └── after/
├── test3_empty_data/
│   ├── before/
│   ├── training/
│   └── after/
└── test_summary.json  ← Overall results
```

---

## 📊 Interpreting Test Results

### Test 1: Overfitting Test
- **PASS:** WER < 30% on training data
- **FAIL:** WER > 30% → Model not learning, check config
- **Purpose:** Verify model can learn from data

### Test 2: Before-After Test
- **PASS:** WER_after < WER_before
- **FAIL:** WER_after ≥ WER_before → Fine-tuning not working
- **Purpose:** Verify fine-tuning improves model

### Test 3: Empty Data Test
- **PASS:** WER_after ≥ WER_before OR training fails
- **FAIL:** WER_after < WER_before → Model changing unexpectedly
- **Purpose:** Verify training doesn't randomly improve model

---

## 🚀 Recommended Testing Workflow

### On Local Windows Machine:
```bash
# 1. Download smoke test data
python scripts/download_smoke_test_data.py --n_train 3 --n_dev 1 --n_test 1

# 2. Verify data
python verify_data.py

# 3. Test config loading
python -c "from omegaconf import OmegaConf; print('✓ Config OK')"
```

### On RunPod (First Time):
```bash
# 1. Download base model
python scripts/download_model.py --model marathi

# 2. Download smoke test data
python scripts/download_smoke_test_data.py --n_train 3 --n_dev 1 --n_test 3

# 3. Run E2E tests (~30 min)
python tests/test_e2e_pipeline.py --model marathi --test all

# 4. If tests pass, download full data and train
python scripts/download_data_from_railway.py --output_dir data
python scripts/nemo_train.py --config configs/konkani_finetune.yaml --model marathi
```

---

## 🐛 Troubleshooting Tests

### Test Hangs or Times Out
- Each test has 30-minute timeout
- Check GPU availability: `nvidia-smi`
- Reduce batch size or epochs in config

### "Model Not Found" Error
- Download base model first: `python scripts/download_model.py --model marathi`
- Or skip before-test in Test 2

### "Empty Manifest" Error in Test 3
- This is expected! Test 3 deliberately uses empty data
- If training fails with empty data, that's actually a PASS

### WER is 100%
- Check audio files exist: `ls data_smoke/test/audio/*.wav`
- Verify manifest format: `cat data_smoke/test/manifest.jsonl`
- Ensure correct model_type flag

---

## 💡 Tips

1. **Always run smoke tests first** on RunPod before full training
2. **Test both models** (Marathi and Konkani) to compare WER
3. **Save test results** before stopping RunPod pod
4. **Use frozen encoder** for faster smoke tests (`--freeze_encoder`)
5. **Check test_summary.json** for detailed results

---

## 📖 Next Steps

After smoke tests pass:
1. ✅ Download full dataset: `python scripts/download_data_from_railway.py`
2. ✅ Train with full config: `python scripts/nemo_train.py --config configs/konkani_finetune.yaml`
3. ✅ Monitor training: `watch -n 1 nvidia-smi`
4. ✅ Test final model: `python scripts/nemo_test.py`
5. ✅ Download results and stop pod

**Good luck!** 🚀
