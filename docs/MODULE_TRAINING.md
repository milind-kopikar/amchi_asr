# Module: Training — Fine-tuning IndicConformer for Amchi ASR

**Self-contained guide.** Read this to train or retrain any model in this project.
For checkpoint locations after training, see `docs/CHECKPOINTS_REGISTRY.md`.

---

## What this module does

Fine-tunes the AI4Bharat IndicConformer hybrid CTC/RNNT model on a custom dataset
using CTC-only training (the RNNT head is not updated). Outputs `.ckpt` checkpoints
and a `final_test_results.json` with per-sample WER.

---

## Key files

| File | Purpose |
|---|---|
| `scripts/fine_tune.py` | Main training entrypoint |
| `configs/` | One YAML config per experiment (see list below) |
| `patches/conv_asr_fixed.py` | Must be applied to NeMo after every install |
| `scripts/create_speaker_stratified_split.py` | Generates speaker-stratified manifests |
| `scripts/create_speed_perturbed_manifests.py` | Generates 3× speed-perturbed training data |

### Config files (most recent first)

| Config | Experiment | Notes |
|---|---|---|
| `configs/deaf_speech_sp_baseline_100epoch.yaml` | DS-D deaf speech (BEST) | Speed-perturbed, full FT, 100ep |
| `configs/deaf_speech_full_finetune_extended_100epoch.yaml` | DS-B deaf speech | Extended data, full FT |
| `configs/deaf_speech_frozen_encoder_100epoch.yaml` | DS-A deaf speech | Frozen encoder |
| `configs/deaf_speech_story4_50epoch.yaml` | Baseline deaf speech | Original 50-epoch run |
| `configs/amchi_konkani_run_c_stratified.yaml` | Run S Amchi Konkani (BEST) | Frozen encoder, stratified split |
| `configs/amchi_konkani_run_c.yaml` | Run C Amchi Konkani | Frozen encoder, story split |

---

## Environment setup (RunPod — do once per pod restart)

```bash
# 1. Install NeMo ASR (not persistent across RunPod restarts)
pip install "nemo_toolkit[asr]" --ignore-installed blinker -q

# 2. Fix PyTorch/torchaudio version mismatch (cu128 pod ships wrong cu124 libs)
pip install --force-reinstall torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128 -q

# 3. Apply conv_asr patch (NeMo install overwrites it — must redo every time)
NEMO_FILE=$(python3 -c "import nemo.collections.asr.modules.conv_asr as m; print(m.__file__)" 2>&1 | tail -1)
cp patches/conv_asr_fixed.py "$NEMO_FILE"

# 4. Confirm GPU is visible
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 5. Required env var for training
export APPLY_CONV_PATCH=1
```

> **Why the patch?** The upstream NeMo `conv_asr.py` has a bug that causes a crash
> during CTC-only training. `patches/conv_asr_fixed.py` is the corrected version.

---

## Running training

```bash
cd /workspace/amchi_asr
export APPLY_CONV_PATCH=1

python3 scripts/fine_tune.py \
  --config configs/<your_config>.yaml \
  --output_dir results/<your_run_name>
```

Results land in `results/<your_run_name>/experiments/<timestamp>/`:
- `epoch_metrics.csv` — val_WER and loss per epoch
- `final_test_results.json` — per-sample WER on test set
- `samples_epoch_XX.json` — dev predictions per epoch

Checkpoints land in `results/<your_run_name>/checkpoints/`.

---

## Speed perturbation (3× synthetic data)

Generates 0.9×, 1.0×, 1.1× speed variants of a training manifest:

```bash
# Edit the INPUT_MANIFEST variable in the script first, then:
python3 scripts/create_speed_perturbed_manifests.py
```

**Key lesson:** Only augment training data from the same story/speaker distribution
as the test set. Adding out-of-distribution samples hurts narrow test sets (see LEARNINGS.md §9).

**API note:** `torchaudio.functional.speed` (v2.6+) returns `(waveform, lengths_or_None)`,
NOT `(waveform, new_sample_rate)`. The output is already at the original sample rate.

---

## Choosing a config: freeze encoder or not?

| Situation | Recommendation |
|---|---|
| Target language is acoustically close to Marathi (e.g. Konkani) | `freeze_encoder: true` — prevents overfitting, 132K trainable params |
| Target domain is acoustically very different (e.g. deaf speech) | Leave freeze off — encoder must adapt to new acoustic patterns |
| Small dataset (< 200 samples) | Always add speed perturbation first |

---

## Monitoring training

```bash
# Live tail of training log (if running in background)
tail -f /tmp/training.log

# Check best checkpoint so far
grep "reached\|not in top" /tmp/training.log | tail -10

# Check epoch metrics
cat results/<run>/experiments/<timestamp>/epoch_metrics.csv | tail -5
```

---

## Two WER metrics — which one to trust

`fine_tune.py` logs two WER metrics simultaneously:
- **RNNT val_wer** → written to `epoch_metrics.csv`. Plateaus early (~epoch 6). **Ignore for quality signal.**
- **CTC val_wer** → shown in checkpoint filenames (e.g. `epoch=88-val_wer=0.334.ckpt`). Continues improving. **This is the primary metric.**

Always use the checkpoint filename WER, not the CSV, as the quality signal.

---

## Common issues

| Issue | Fix |
|---|---|
| `OSError: Could not load libtorchaudio.so` | Re-run the cu128 torch reinstall (step 2 above) |
| `RuntimeError: operator torchvision::nms does not exist` | Same fix |
| Training runs on CPU (very slow) | Check `CUDA_VISIBLE_DEVICES`; run GPU check script |
| OOM error | Reduce `batch_size` in config (currently 4) |
| `conv_asr` crash | Re-apply the patch (step 3 above) |

See also: `LEARNINGS.md` for a full history of issues and fixes.
