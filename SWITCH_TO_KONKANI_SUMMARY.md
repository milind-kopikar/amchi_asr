# Switch from Marathi to Konkani Base Model - Summary

**Date**: December 22, 2025  
**Task**: Prepare for fine-tuning AI4Bharat **Goan Konkani** model (instead of Marathi)

---

## ✅ What Was Prepared

### 1. Documentation Created
- ✅ **[KONKANI_MODEL_PLAN.md](KONKANI_MODEL_PLAN.md)** - Comprehensive deployment plan with:
  - Comparison of Marathi vs Konkani base models
  - Step-by-step RunPod deployment guide
  - Known issues and solutions from previous training
  - Expected results and success criteria
  - Experiment tracking template

- ✅ **[RUNPOD_QUICK_START.md](RUNPOD_QUICK_START.md)** - Quick reference guide with:
  - Copy-paste friendly commands
  - Troubleshooting solutions
  - Time and cost estimates
  - Full workflow from start to finish

### 2. Setup (Python 3.11 + Upstream NeMo)
- ✅ Use **setup_env.sh** with **Python 3.11** and **upstream** NeMo (`USE_UPSTREAM_NEMO=1`). Do not use the AI4Bharat NeMo fork for normal setup (it requires Python 3.9). See [SETUP_ENV.md](SETUP_ENV.md).

### 3. Existing Infrastructure (Already in Place)
- ✅ **scripts/nemo_train.py** - Supports both `--model marathi` and `--model konkani`
- ✅ **scripts/download_model.py** - Handles both models with shorthand flags
- ✅ **scripts/nemo_validate.py** - Model evaluation
- ✅ **scripts/nemo_test.py** - WER/CER calculation with error analysis
- ✅ **scripts/download_data_from_railway.py** - Data fetching from Railway
- ✅ **configs/konkani_finetune.yaml** - Training configuration

---

## 📊 Key Differences: Marathi vs Konkani Base

| Aspect | Marathi Base (Dec 18) | Konkani Base (NEW - Dec 22) |
|--------|--------------|---------------------|
| **Model** | `ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large` | `ai4bharat/indicconformer_stt_kok_hybrid_ctc_rnnt_large` |
| **Language** | Marathi (मराठी) | Goan Konkani (कोंकणी) |
| **Training Data** | 44 recordings (27 train, 5 dev, 8 test) | **250+ recordings (175+ train, 37+ dev, 37+ test)** |
| **Data Improvement** | Baseline | **6x more data!** |
| **Linguistic Distance to Amchi Konkani** | Moderate | **Close** (same language family) |
| **Previous WER (Validation)** | 65.38% | **Unknown - To be measured** |
| **Previous WER (Test)** | 87.36% | **Unknown - To be measured** |
| **Expected WER with More Data** | N/A | **< 50% (hypothesis: better model + 6x data)** |
| **Hypothesis** | Good baseline | **Much better (better model + 6x data)** |
| **Training Command** | `--model marathi` | `--model konkani` |
| **Download Path** | `models/indicconformer_mr/` | `models/indicconformer_kok/` |

---

## 🚀 Next Steps for RunPod

### Option 1: Train Only Konkani Model (~30 min, $0.35)
```bash
# Single-model workflow
bash setup_runpod.sh && source venv/bin/activate
python scripts/download_model.py --model konkani
python scripts/download_data_from_railway.py --base_url https://konkanicollector-production.up.railway.app --output_dir data/train --train_split 0.8
python scripts/nemo_train.py --config configs/konkani_finetune.yaml --model konkani --output_dir results/konkani_full --max_epochs 50
python scripts/nemo_test.py --model results/konkani_full/konkani_asr_final.nemo --manifest data/dev/manifest.jsonl --model_type konkani --output_dir test_results/konkani
```

### Option 2: Compare Both Models (~45 min, $0.52)
```bash
# Download both models
python scripts/download_model.py --model marathi
python scripts/download_model.py --model konkani

# Train Konkani
python scripts/nemo_train.py --config configs/konkani_finetune.yaml --model konkani --output_dir results/konkani_full --max_epochs 50

# Train Marathi
python scripts/nemo_train.py --config configs/konkani_finetune.yaml --model marathi --output_dir results/marathi_full --max_epochs 50

# Compare results
python scripts/nemo_test.py --model results/konkani_full/konkani_asr_final.nemo --manifest data/dev/manifest.jsonl --model_type konkani --output_dir test_results/konkani
python scripts/nemo_test.py --model results/marathi_full/marathi_asr_final.nemo --manifest data/dev/manifest.jsonl --model_type marathi --output_dir test_results/marathi

# Show comparison
echo "=== KONKANI BASE ===" && cat test_results/konkani/test_results.json | jq '{wer, cer}'
echo "=== MARATHI BASE ===" && cat test_results/marathi/test_results.json | jq '{wer, cer}'
```

**Recommendation**: Start with **Option 1** (Konkani only) since it's likely to perform better.

---

## 🎯 Success Criteria

### Minimum Acceptable Performance
- ✅ Training completes without errors
- ✅ Validation WER < 70%
- ✅ Model generates valid Devanagari output

### Good Performance (Hypothesis)
- ✅ Validation WER < 65% (matches/beats Marathi)
- ✅ Test WER < 80%
- ✅ Fewer substitution errors on Konkani-specific words

### Excellent Performance (Stretch Goal)
- ✅ Validation WER < 60%
- ✅ Test WER < 75%
- ✅ Model correctly handles Konkani phonemes that differ from Marathi

---

## 📝 Documentation to Create After Training

After training completes, create:

**`TRAINING_RESULTS_KONKANI_2025-12-22.md`** with:
```markdown
# Konkani ASR Training Results - Konkani Base Model

**Date**: December 22, 2025
**Base Model**: AI4Bharat Goan Konkani IndicConformer

## Results

| Metric | Value |
|--------|-------|
| Baseline WER (pre-fine-tuning) | XX.XX% |
| Fine-tuned WER (validation) | XX.XX% |
| Fine-tuned WER (test) | XX.XX% |
| Improvement | ±XX.XX pp |

## Comparison with Marathi Base

| Metric | Marathi Base | Konkani Base | Winner |
|--------|--------------|--------------|--------|
| Val WER | 65.38% | XX.XX% | ? |
| Test WER | 87.36% | XX.XX% | ? |

## Sample Transcriptions

[Add examples here]

## Conclusion

[Which model performed better and why?]
```

---

## 🔍 What Changed from Previous Setup

### New Files
- `KONKANI_MODEL_PLAN.md` - Comprehensive plan
- `RUNPOD_QUICK_START.md` - Quick reference
- `SWITCH_TO_KONKANI_SUMMARY.md` - This file

### Updated Files
- Use `setup_env.sh` with `USE_UPSTREAM_NEMO=1` (Python 3.11 + upstream NeMo). See SETUP_ENV.md.

### Unchanged (Already Working)
- All scripts in `scripts/` directory
- Configuration in `configs/`
- Documentation in `AI4BHARAT_SETUP_GUIDE.md`, `NEMO_WORKFLOW_GUIDE.md`

---

## 💡 Key Insights

### From Previous Training (Marathi Base - Dec 18)
1. **Data sufficiency**: 27 train + 5 dev samples achieved 65% WER
2. **Convergence**: Model converged around epoch 40-45
3. **No overfitting**: Validation and training WER tracked closely
4. **Time efficient**: Only 12 minutes on RTX 4090

### Expected for Konkani Base
1. **Better performance**: Goan Konkani linguistically closer to Amchi Konkani
2. **Faster convergence**: Base model may already know some Konkani patterns
3. **Lower WER**: Target < 60% (vs 65% with Marathi)
4. **Same time**: Training time should be similar (~12-15 min)

---

## ⚠️ Critical Reminders

1. **Python 3.11** - Use Python 3.11 and **upstream** NeMo. Do NOT use the AI4Bharat NeMo fork for normal setup (it requires Python 3.9 and fails on 3.11).
2. **USE_UPSTREAM_NEMO=1** - Set before `setup_env.sh` or install in venv with `pip install "nemo_toolkit[all]"` (see SETUP_ENV.md).
3. **conv_asr patch** - Apply after NeMo install (setup_env.sh does this).
4. **Model download** - Accept license on Hugging Face and login (or set HF_TOKEN) before downloading.

---

## 📚 Reference Documentation

- [KONKANI_MODEL_PLAN.md](KONKANI_MODEL_PLAN.md) - Detailed plan
- [RUNPOD_QUICK_START.md](RUNPOD_QUICK_START.md) - Quick commands
- [AI4BHARAT_SETUP_GUIDE.md](AI4BHARAT_SETUP_GUIDE.md) - Environment setup
- [NEMO_WORKFLOW_GUIDE.md](NEMO_WORKFLOW_GUIDE.md) - Full workflow
- [TRAINING_RESULTS_2025-12-18.md](TRAINING_RESULTS_2025-12-18.md) - Previous results

---

**Status**: ✅ Ready to deploy on RunPod  
**Estimated Time**: 30-45 minutes  
**Estimated Cost**: $0.35-0.52  
**Next Action**: Start RunPod pod and run setup script
