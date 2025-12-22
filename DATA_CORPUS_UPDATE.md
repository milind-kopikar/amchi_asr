# Data Corpus Update - December 22, 2025

## 🎉 Major Data Expansion

**Previous corpus (Dec 18)**: 44 recordings  
**Current corpus (Dec 22)**: **250+ recordings** (6x increase!)

---

## 📊 Impact on Model Training

### Data Quality Improvements

| Metric | Dec 18 (Marathi Base) | Dec 22 (Konkani Base) | Impact |
|--------|----------------------|----------------------|--------|
| **Total Recordings** | 44 | 250+ | 6x more data |
| **Training Samples** | 27 (61%) | ~175 (70%) | 6.5x increase |
| **Validation Samples** | 5 (11%) | ~37 (15%) | 7.4x increase |
| **Test Samples** | 8 (18%) | ~37 (15%) | 4.6x increase |
| **Speaker Diversity** | Limited | **High** (multiple speakers) |
| **Sentence Coverage** | Partial | **Comprehensive** (5 stories) |
| **Expected WER** | 65% val, 87% test | **40-55% (predicted)** |

### Key Benefits of Larger Corpus

1. **Better Generalization**
   - More speakers → Model learns speaker-independent features
   - Less overfitting to specific voices
   - Better performance on unseen speakers

2. **Improved Vocabulary Coverage**
   - More unique Konkani words and phrases
   - Better handling of rare phonemes
   - Improved word boundary detection

3. **Stronger Statistical Learning**
   - More training examples → Better parameter optimization
   - Smoother convergence
   - More stable validation metrics

4. **Reduced Overfitting Risk**
   - Larger validation set (37 vs 5) → More reliable WER estimates
   - More diverse test set → Better real-world performance estimation

---

## 🔀 Data Randomization Strategy

### Why Randomization Matters

Without randomization:
- ❌ All samples from Speaker A in train set
- ❌ All samples from Speaker B in test set
- ❌ Model learns speaker-specific patterns
- ❌ Poor generalization to new speakers

With randomization (seed=42):
- ✅ Each speaker's recordings spread across train/dev/test
- ✅ Model learns speaker-independent patterns
- ✅ Better generalization
- ✅ Reproducible splits (same seed = same split every time)

### Implementation

The `download_data_from_railway.py` script:
```python
# Shuffle recordings randomly to mix speakers
random.seed(42)  # Reproducible shuffle
random.shuffle(recordings)

# Then split 70/15/15
train = recordings[:175]
dev = recordings[175:212]
test = recordings[212:]
```

This ensures:
- **Speaker diversity** in each split
- **Reproducibility** (same seed gives same split)
- **No data leakage** (no overlap between splits)

---

## 📈 Expected Performance Improvements

### Previous Results (Dec 18, 44 samples, Marathi base)
- Baseline (no fine-tuning): 92% WER
- After fine-tuning: **65% WER** (validation), **87% WER** (test)
- Improvement: 27 percentage points

### Predicted Results (Dec 22, 250+ samples, Konkani base)
- Baseline (no fine-tuning): Unknown (likely ~70-80% WER, better than Marathi)
- After fine-tuning: **40-55% WER** (validation), **60-70% WER** (test)
- Expected improvement factors:
  - **Better base model**: Konkani closer to Amchi Konkani (-10 to -15 pp)
  - **6x more data**: Better generalization (-10 to -15 pp)
  - **Combined effect**: -20 to -30 percentage points improvement

### Conservative Estimate
- Validation WER: **< 55%**
- Test WER: **< 70%**
- This would be **17+ percentage points better** than previous best (87% → 70%)

---

## ✅ Verification Checklist

Before training, verify:

1. **Data downloaded completely**
   ```bash
   echo "Total recordings: $(cat data/train/manifest.jsonl data/dev/manifest.jsonl data/test/manifest.jsonl | wc -l)"
   # Should show 250+
   ```

2. **Speaker randomization applied**
   ```bash
   # Check that different speakers appear in each split
   head -5 data/train/manifest.jsonl | jq -r .audio_filepath
   head -5 data/dev/manifest.jsonl | jq -r .audio_filepath
   head -5 data/test/manifest.jsonl | jq -r .audio_filepath
   # Should see different recording IDs
   ```

3. **Devanagari text preserved**
   ```bash
   head -2 data/train/manifest.jsonl | jq -r .text
   # Should show Konkani Devanagari text, not corrupted
   ```

4. **Audio files downloaded**
   ```bash
   ls -l data/train/audio/*.wav | wc -l
   ls -l data/dev/audio/*.wav | wc -l
   ls -l data/test/audio/*.wav | wc -l
   # Counts should match manifest line counts
   ```

---

## 🚀 Training Configuration Update

With 6x more data, we can use more aggressive training:

### Recommended Changes

```yaml
# configs/konkani_finetune.yaml

# Increase batch size (more data = larger batches OK)
data:
  train_ds:
    batch_size: 16  # Was 8, now 16 (if GPU has memory)
    
  validation_ds:
    batch_size: 16

# May need more epochs to converge with more data
trainer:
  max_epochs: 75  # Was 50, increase to 75
  
# Or use patience-based early stopping
# (stop if no improvement for 10 epochs)
callbacks:
  - _target_: pytorch_lightning.callbacks.EarlyStopping
    monitor: val_wer
    patience: 10
    mode: min
```

### Alternative: Keep Original Config
- Original config (batch_size=8, max_epochs=50) should still work
- May converge faster with more data
- Safer if unsure about GPU memory

---

## 📝 Summary

**What changed**:
- ✅ Railway database now has 250+ approved recordings (6x increase)
- ✅ Data download script fetches ALL recordings (no limit)
- ✅ Randomization with seed=42 ensures speaker diversity
- ✅ 70/15/15 train/dev/test split (better than previous 80/10/10)
- ✅ Switching to Konkani base model (linguistically closer)

**Expected outcome**:
- 🎯 WER: 40-55% validation, 60-70% test (vs previous 65% val, 87% test)
- 🎯 Improvement: 17-25 percentage points better
- 🎯 Model will generalize better to new speakers
- 🎯 Better vocabulary coverage

**Action required**:
- None! The script automatically handles everything
- Just run `python scripts/download_data_from_railway.py` as documented
- Verify you get 250+ total samples across all splits

---

**Status**: ✅ Ready for RunPod deployment with enhanced corpus
