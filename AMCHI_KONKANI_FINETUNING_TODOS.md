# Amchi Konkani ASR — Fine-Tuning Optimization TODO

**Current best result (2026-03-02, 50-epoch run):**
- Test WER: **54.7%** on Story 5, 104 samples, 3 speakers
- Pilot baseline: 35.1% (20 epochs, 38 samples, 1 speaker — ashaheble only)
- ashaheble-only WER this run: 51.7% (same speaker, more data, still worse than pilot — overfitting)

**Root cause of underperformance:**
1. Model overfits: val_loss rose from 49 (epoch 6) to 98 (epoch 49); train_loss fell to 0.17
2. All 119M encoder parameters were trained — more params = more overfitting on 511 samples
3. Learning rate was flat 0.0001 — no warmup or decay

---

## Optimization Runs to Try (ordered by expected impact)

### Run A — Freeze Encoder, Fine-tune Decoder Only  **[HIGHEST PRIORITY]**
**Status:** NOT done in current run. The `fine_tune.py` already supports `freeze_encoder: true` (line 626).

The encoder has 115M parameters. The CTC decoder head has ~131K params + RNNT components ~4.3M total.
Freezing the encoder means only the decoder adapts to Konkani — massively reduces overfitting on 511 samples.

**New config:** `configs/amchi_konkani_frozen_encoder.yaml`
```yaml
# Same as amchi_konkani_50epoch.yaml but add:
freeze_encoder: true
trainer:
  max_epochs: 50
```
**Trainable params with frozen encoder:** ~4.4M instead of 119M.

---

### Run B — LR Warmup + Cosine Decay
**Status:** NOT done. Current config uses flat lr=0.0001.

```yaml
optim:
  name: adamw
  lr: 0.0001
  weight_decay: 0.001
  sched:
    name: CosineAnnealing
    warmup_steps: 500
    min_lr: 0.000001
```

---

### Run C — Combined: Frozen Encoder + LR Scheduling + 100 Epochs  **[RECOMMENDED NEXT]**
**Status:** NOT done. Combine Run A + Run B + longer training (model still improving at epoch 47).

```yaml
freeze_encoder: true
trainer:
  max_epochs: 100
optim:
  name: adamw
  lr: 0.0001
  weight_decay: 0.01
  sched:
    name: CosineAnnealing
    warmup_steps: 500
    min_lr: 0.000001
```

---

### Run D — SpecAugment Reduction
**Status:** NOT done. Default NeMo SpecAugment may be too aggressive for 500-sample dataset.

```yaml
model:
  spec_augmentation:
    freq_masks: 1
    freq_width: 20
    time_masks: 2
    time_width: 0.1
```

---

### Run E — Lower LR (0.00005)
**Status:** NOT done. Try only after Runs A-C.

---

## Summary Table

| Run | Freeze Encoder | LR Schedule | Max Epochs | Priority |
|-----|---------------|-------------|-----------|----------|
| Current (done) | No | Flat 1e-4 | 50 | Done |
| **A** | **Yes** | Flat 1e-4 | 50 | **High** |
| **B** | No | Cosine | 50 | Medium |
| **C** | **Yes** | Cosine | 100 | **Highest** |
| D | No | Flat 1e-4 | 50 | Low |
| E | No | Flat 5e-5 | 100 | Low |

---

## How to Run on Fresh RunPod Pod

```bash
cd /workspace/amchi_asr && git pull

# Fix cu128 torchvision/torchaudio mismatch (REQUIRED every new pod)
pip install "nemo_toolkit[asr]" --ignore-installed blinker -q
pip install --force-reinstall torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 -q

# Apply conv_asr patch
NEMO_FILE=$(python3 -c "import nemo.collections.asr.modules.conv_asr as m; print(m.__file__)" 2>/dev/null | tail -1)
cp patches/conv_asr_fixed.py "$NEMO_FILE"

# Download data
python3 scripts/download_data_from_railway.py \
  --base_url https://konkanicollector-production.up.railway.app \
  --output_dir data/amchi --use_story_split

# Run A (frozen encoder)
export APPLY_CONV_PATCH=1
python3 scripts/fine_tune.py --config configs/amchi_konkani_frozen_encoder.yaml

# Run C (frozen + cosine + 100 epochs) — recommended
python3 scripts/fine_tune.py --config configs/amchi_konkani_run_c.yaml
```

---

## Results Tracking

| Run | Date | Test WER | Val WER (best) | Speakers | Notes |
|-----|------|----------|----------------|---------|-------|
| Pilot (20 ep) | Jan 2026 | 35.1% | 65.4% | 1 (ashaheble) | 38 test samples |
| 50-epoch | 2026-03-02 | 54.7% | 53.2% CTC | 3 | 104 test samples |
| Run A | TBD | — | — | 3 | |
| Run C | TBD | — | — | 3 | |

Per-speaker WER (50-epoch run, Story 5):
- ashaheble: 51.7% (was 35.1% in pilot — same speaker)
- dipti.ajgaonkar21: 60.3% (new speaker)
- lalimomadi: 51.9% (new speaker)

---

## After Training: Analysis

```bash
python3 scripts/analyze_results.py \
  --results results/experiments/<timestamp>/postprocessed_results.json \
  --manifest data/amchi/test/manifest.jsonl \
  --output_dir results/amchi_analysis/
```

Produces: WER/CER histograms, speaker boxplots, Wilcoxon test, bootstrap CI, error breakdown, length vs WER scatter.
