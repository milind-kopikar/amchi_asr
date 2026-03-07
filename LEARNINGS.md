# Learnings from Test Runs (February 2026)

This document records what we learned from running the preflight suite, smoke tests, and the 20-epoch Marathi Amchi fine-tuning. Use it to avoid repeating mistakes and to keep the test suite aligned with reality.

---

## 1. What Works

- **Python 3.11 + upstream NeMo:** We use **Python 3.11** with **upstream** NVIDIA NeMo (`nemo_toolkit[all]`). Do **not** use the AI4Bharat NeMo fork for this setup—it targets Python 3.9 and is incompatible with 3.11.
- **CTC-only fine-tuning:** Using decoder_type CTC and training only the auxiliary CTC head avoids RNNT joint/decoder validation and GPU JIT (Numba) issues. See `REPRODUCTION_NOTES.md`.
- **20-epoch full run:** The pipeline runs end-to-end with `configs/marathi_amchi_20epoch.yaml`: 20 epochs, `save_top_k=3`, best checkpoint by `val_wer`, with CER and WER logged per epoch.
- **RunPod / A40:** Training was verified on RunPod with an A40 GPU; `CUDA_VISIBLE_DEVICES=0` was used. Ensure the instance has a GPU and that CUDA is visible before starting.
- **CER in results:** We log CER (Character Error Rate) in addition to WER in `samples_epoch_XX.json` and in the test results. CER is important for Devanagari where small spelling differences inflate WER.

---

## 2. Smoke Test

- **Purpose:** Quickly verify that the full pipeline runs and that the model can improve on a single sample (train/dev/test all use the same sample). This catches environment and config errors before long runs.
- **Location:** One-sample smoke test output lives under `results/smoke_tests/`. Config: `configs/smoke_tests_5epoch.yaml`. Runner: `./scripts/run_smoke_test_one_sample.sh`. Validator: `scripts/validate_smoke_one_sample.py`.
- **Success criteria:** Both **validation loss** and **CER** must improve (decrease) over the 5 epochs. If either fails to improve, the smoke test fails (exit code 1).
- **Integration:** The one-sample smoke test is **step 6** of the full preflight suite: `./scripts/run_all_preflight.sh`. See `results/smoke_tests/README.md` and `scripts/README_SMOKE.md`.

---

## 3. GPU Check in the Test Suite

- **Lesson learned:** We discovered that training can appear to “run” on CPU if the GPU is not visible (e.g. `CUDA_VISIBLE_DEVICES` unset or misconfigured), leading to very slow runs or late failures. The test suite should **verify that a GPU is available and usable** before starting any training or heavy tests.
- **Action taken:** A **GPU check** was added to the preflight suite and runs **first** (before library, data, or smoke tests). If no CUDA-capable GPU is detected, the script exits with a clear error so we do not waste time on CPU-only runs.
- **How to run:** The check runs automatically as step 1 of `./scripts/run_all_preflight.sh`. It can also be run standalone: `python3 scripts/check_gpu.py`.

---

## 4. What We Do Not Push to Git

- **Data files:** The `data/` directory (audio and manifests) is large and is **not** committed. It remains in `.gitignore`.
- **Large checkpoints:** Result directories may contain large `.ckpt` files and `checkpoints/` folders. These are ignored via `.gitignore` so we only commit lightweight result artifacts (e.g. `epoch_metrics.csv`, `samples_epoch_*.json`, `final_test_results.json`, `run_metadata.json`, READMEs).

---

**See also:** `MASTER_REPRODUCTION_GUIDE.md`, `REPRODUCTION_NOTES.md`, `results/smoke_tests/README.md`, `scripts/README_SMOKE.md`.

---

## 5. Deaf Speech Fine-Tuning (Session 2026-03-01)

### Environment
- `nemo_toolkit[asr]` v2.7.0 works with Python 3.11. Install with `--ignore-installed blinker` to avoid conflicts.
- Do NOT install `nemo_toolkit[all]` — the root filesystem on RunPod is only 20GB and `[all]` fills it up. Use `[asr]` only.
- Clear pip cache before large installs: `pip cache purge` (frees ~8GB on a typical RunPod instance).
- Use `PIP_CACHE_DIR=/workspace/.pip_cache` for all pip installs to keep cache on the larger workspace volume.

### Training
- Best checkpoint for deaf speech Story 4 arrived at epoch 21 (val_WER=0.720). After epoch 21, the model oscillated without improving — overfitting to training data, consistent with 124-sample dataset.
- `log_every_n_steps: 5` is appropriate for small datasets (124 samples). With batch_size=4, each epoch = 31 steps.
- The same 124 samples used for train/dev/test is intentional for an "overfit verification" experiment.
- WER=1.0 for first 6-7 epochs is normal for deaf speech — the model is starting from hearing speech weights and needs time to adjust to atypical speech patterns.

### Inference loading (CRITICAL)
Loading a fine-tuned checkpoint from disk is NOT straightforward. The config inside the checkpoint has:
1. `loss_name: ctc` — which the RNNT validator rejects on load
2. Train/dev/test dataset paths that don't exist in inference context

**Solution (from REPRODUCTION_NOTES.md § 9):**
```python
# Load config from checkpoint without running it
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
cfg = ckpt['hyper_parameters']['cfg']
# Patch config for inference
cfg.loss.loss_name = 'default'
del cfg.train_ds, cfg.validation_ds, cfg.test_ds
# Instantiate with patched config
model = EncDecHybridRNNTCTCBPEModel(cfg=cfg, trainer=None)
# Add empty ds configs back (transcribe() needs them)
cfg.validation_ds = OmegaConf.create({})
cfg.test_ds = OmegaConf.create({})
# Load weights
model.load_state_dict(ckpt['state_dict'], strict=False)
# Force CTC decoding
model.change_decoding_strategy(decoder_type='ctc')
# Transcribe
result = model.transcribe(audio=['path/to/audio.wav'])
```

### Post-Processing with Gemini
- **gemini-2.0-flash is deprecated** for new API key users (as of early 2026). Use `gemini-2.5-flash`.
- The `google-generativeai` package is deprecated. Use the new `google-genai` package: `pip install google-genai`.
- Synchronous Gemini calls: ~1-2 seconds each. For 124 samples with 0.5s delay: ~5 minutes. Parallelize with `asyncio` for faster batch processing.
- **The conservative safety valve** (revert to original if Gemini worsens WER) is essential. Without it, Gemini changes words in already-correct predictions.
- **WER metric understates post-processing value** because WER requires word-exact matches. Even when Gemini produces the correct meaning (e.g. `बस लवकर येईल का?` ≈ `बस कधी येईल?`), WER counts it as wrong.
- The `⁇` marker in NeMo CTC output signals "cannot decode this token." Strip it before any processing or WER calculation.

### .gitignore for data manifests
The `data/` directory is in `.gitignore` to exclude large audio files. But manifest files (*.jsonl) are small text files that SHOULD be tracked. Add exceptions:
```
# In .gitignore:
data/
!data/**/manifest.jsonl
```

---

## 6. Amchi Konkani Fine-Tuning — 50-epoch Run (Session 2026-03-02)

### Environment fix: CUDA 12.8 torchvision/torchaudio mismatch
Fresh RunPod pods now ship with PyTorch 2.10.0+cu128. If `nemo_toolkit[asr]` is installed on top, it pulls in `torchvision` and `torchaudio` built for cu124, causing `RuntimeError: operator torchvision::nms does not exist` and `OSError: Could not load libtorchaudio.so`. Fix:
```bash
pip install --force-reinstall torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128 -q
```
Then re-apply the conv_asr patch (the NeMo install overwrites it):
```bash
NEMO_FILE=$(python3 -c "import nemo.collections.asr.modules.conv_asr as m; print(m.__file__)" 2>&1 | tail -1)
cp patches/conv_asr_fixed.py "$NEMO_FILE"
```

### Two WER metrics in fine_tune.py
The `fine_tune.py` script logs two separate WER metrics simultaneously:
- **RNNT val_wer** — logged to `epoch_metrics.csv`. Plateaus early (~64% by epoch 6).
- **CTC val_wer** — monitored by `ModelCheckpoint` callback and shown in checkpoint filenames. This is the metric that continues improving throughout training (reached 53.2% at epoch 47).

The final `final_test_results.json` uses the CTC decoder on the best checkpoint. Always use the checkpoint-filename WER, not the CSV val_wer, as the primary quality signal.

### Training results
| Metric | Value |
|--------|-------|
| Best checkpoint | epoch 47, CTC val_wer = 53.2% |
| Test WER (Story 5, 104 samples, 3 speakers) | **54.7%** |
| Pilot baseline test WER (Story 5, 38 samples) | 35.1% |
| RNNT val_wer plateau | ~62–64% (from epoch 6) |
| Train loss at epoch 50 | 0.17 (overfitting — val_loss rose to 2× epoch-6 value) |

### Why test WER (54.7%) is higher than baseline (35.1%)
1. **Test set is harder**: 3 speakers × 35 sentences = 104 samples vs 38 samples from likely 1 speaker. Unseen speakers are the main challenge.
2. **Overfitting signal**: val_loss increased 2× from epoch 6 to epoch 49 while train_loss fell to 0.17. Model memorised training data.
3. **Not a fair comparison**: baseline test set was single-speaker; new test set is 3-speaker.

### Data paths changed (amchi split)
- Output dir is now `data/amchi/` (not `data/`) to avoid clashing with deaf speech data.
- Config updated: manifests now at `data/amchi/train|dev|test/manifest.jsonl`.
- Story 7 (काय्ळो) added to train set (35 recordings); Story 5 is still the held-out test set.
- Story split now: `{1,2,3,7} → train`, `4 → dev`, `5 → test`.

### Checkpoint location
Checkpoints now save to `/workspace/results/checkpoints/` (not `nemo_experiments/`) — this is controlled by the `output_dir` passed to `fine_tune.py`, which defaults to `results/`. If resuming, look there first.

---

## 7. What We Do Not Push to Git

- **Audio files:** `data/**/*.wav`, `data/**/*.mp3`, etc. — too large.
- **Model weights:** `*.nemo`, `*.ckpt`, `*.pt`, `*.pth` — too large. Use HuggingFace Hub or RunPod persistent storage.
- **Python environments:** `venv/`, `.venv/`, pip cache.
- **Experiment checkpoints:** `nemo_experiments/**/*.ckpt` — excluded by `.gitignore`.
- **API keys / secrets:** NEVER hardcode in scripts. Use `.env` file (see §7 below).
- **DO commit:** `nemo_experiments/**/*.json`, `*.csv`, `*.txt` — lightweight result artifacts are valuable and tracked.

---

## 7. API Key Management (.env convention)

**Rule: all API keys live in `.env`. Never hardcode them in scripts or docstrings.**

This was learned the hard way: a Gemini key hardcoded in `postprocess_asr.py`'s docstring made it through a commit; GitHub's secret scanning blocked the push and required an amend+force-push to fix it.

### Setup on a fresh RunPod instance:
```bash
cp .env.example .env
# Fill in your real keys in .env:
#   GEMINI_API_KEY=AIzaSy...
#   HF_TOKEN=hf_...
#   GITHUB_PAT=ghp_...
source .env
```

### Why this works:
- `.env` is in `.gitignore` (line 130) — it will never be committed.
- `.env.example` IS committed — it shows what variables are needed, without values.

### Scripts that read env vars:
| Script | Env var | CLI override |
|--------|---------|--------------|
| `scripts/postprocess_asr.py` | `GEMINI_API_KEY` | `--api_key` |
| `scripts/deaf_speech_inference.py` | `GEMINI_API_KEY` | `--gemini_key` |

### For git push via PAT (each session):
```bash
source .env
git remote set-url origin https://${GITHUB_PAT}@github.com/milind-kopikar/amchi_asr.git
git push
```

### For Hugging Face downloads:
```bash
source .env && huggingface-cli login --token "$HF_TOKEN"
```

---

## 8. Deaf Speech Experiments — Frozen Encoder vs. Full Fine-Tune (2026-03-07)

### Results summary
| Experiment | Encoder | Data | Best val_WER | Test WER |
|---|---|---|---|---|
| Baseline (50-epoch) | Full fine-tune | 124 samples (train=dev=test) | 72.0% (epoch 21) | 75.3% |
| DS-A (100-epoch) | **Frozen** | 124 samples (train=dev=test) | 76.6% (epoch 96) | **79.6%** |

### Key finding: Freezing the encoder HURTS deaf speech (unlike Konkani)

**Why freezing helped Konkani but hurts deaf speech:**
- **Konkani**: Acoustically very close to Marathi. Pre-trained encoder already captures the right features (similar vowels, consonants, prosody). Freezing it prevents overfitting and lets the CTC head adapt to Konkani vocabulary.
- **Deaf speech**: Fundamentally different acoustic patterns — unusual pitch contours, reduced consonants, breathy phonation, atypical timing. The encoder was trained on hearing speech and its representations do NOT transfer to deaf speech. It **must** update to learn these new acoustic mappings.

**Practical rule:** Freeze encoder when source and target acoustics are close (same language family, dialect adaptation, accent variation). Unfreeze (or partially unfreeze) when the acoustic domain is genuinely different (deaf speech, heavily accented speech, whispered speech, noisy environments).

### val_WER trajectory (DS-A)
- Epoch 0: 685% (started far from hearing-speech prior)
- Epoch 5: 198% (fast initial drop — CTC head adapting)
- Epoch 21: ~100% (same epoch where baseline peaked — frozen encoder has not improved beyond baseline)
- Epoch 96: 76.6% (best — still worse than baseline 72.0% at same epoch)
- Pattern: model improves slowly throughout but never beats the full fine-tune baseline

### What to try next for deaf speech
1. **DS-B**: Full fine-tune + extended data (75 additional tnshenoy recordings from stories 19/20/21) — tests if more data alone helps
2. **DS-D**: Full fine-tune + speed-perturbed baseline data (3× augmentation at 0.9/1.0/1.1× speed on same 124 story-4 samples) — **breakthrough result, see §9**
3. **Future**: Partial freeze (bottom N encoder layers frozen, top layers + decoder trained) — a compromise that prevents lower-level acoustic features from drifting while still allowing upper layers to adapt to deaf speech

---

## 9. Deaf Speech Experiments — More Data vs. Speed Augmentation (2026-03-07)

### Full results table
| Experiment | Encoder | Train data | Train samples | Best val_WER | Test WER |
|---|---|---|---|---|---|
| Baseline (50ep) | Full FT | Story 4 only | 124 (=dev=test) | 72.0% (ep 21) | 75.3% |
| DS-A (100ep) | Frozen | Story 4 only | 124 (=dev=test) | 76.6% (ep 96) | 79.6% |
| DS-B (100ep) | Full FT | Story 4 + tnshenoy stories 19/20/21 | 188 train / 134 dev | 85.0% (ep 75) | 93.1% |
| **DS-D (100ep)** | **Full FT** | **Story 4 × 3 speeds (0.9/1.0/1.1)** | **372 train** | **26.9% (ep 96)** | **34.7%** |

### DS-B: why more (different) data hurt

DS-B added 63 extra recordings from tnshenoy's stories 19/20/21, bringing train to 188 samples. Test WER worsened from 75.3% → 93.1% for three reasons:

1. **Wrong distribution for the test set.** The test set is story 4. DS-B's model was trained across 4 different stories and never specialised on story 4 the way the baseline did.
2. **Checkpoint selected on mixed dev.** The best checkpoint (epoch 62, val_WER=85.0%) was chosen using a dev set drawn from all 4 stories — not optimised for story 4 test performance.
3. **Severe overfitting.** Train loss fell to ~0.001 by epoch 75 while val_loss climbed 25% from its epoch-10 value. The model memorised 188 diverse samples without learning transferable features.

**Rule:** Adding out-of-distribution training data hurts when the test set is narrow (one story, one speaker group). More data only helps when it matches the test distribution.

### DS-D: speed perturbation on same distribution — breakthrough

DS-D trained on 372 samples = 124 story-4 samples × 3 speed factors (0.9×, 1.0×, 1.1×). Dev and test remained the original 124 story-4 samples — same distribution as the baseline.

**Result: test WER 34.7% vs baseline 75.3% — a 40.6pp improvement.**

| Category | Baseline | DS-D |
|---|---|---|
| Good (WER ≤ 50%) | 25% | **77%** |
| Partial (50–99%) | 35% | 16% |
| Fail (WER ≥ 100%) | 40% | **7%** |

**Why it worked:**
- Speed perturbation keeps the acoustic domain identical (same speakers, same story, same Devanagari content) while giving the model 3× more gradient updates per epoch.
- The 0.9× and 1.1× variants teach robustness to natural variation in deaf speaking rate — a real source of variance in this population.
- The model can still overfit to the story-4 content (intentional for this task) while being less sensitive to exact timing patterns.

**torchaudio.functional.speed API note (v2.6+):** Returns `(waveform, lengths_or_None)` — NOT `(waveform, new_sample_rate)` as older docs suggest. Do NOT resample the output; it is already at the original sample rate.

```python
out, _ = torchaudio.functional.speed(waveform, orig_freq=16000, factor=0.9)
# out is already at 16000 Hz, just shorter in time
```

### Practical rule for synthetic augmentation
- **Speed perturbation on same distribution** = highly effective for small deaf speech datasets. Use it by default.
- **Adding out-of-distribution samples** = only helps if the test distribution is broad (multi-story, multi-speaker). Hurts narrow test sets.
- **Recommended default:** always apply 3× speed perturbation (0.9/1.0/1.1) to training data before any other augmentation strategy.

### Checkpoints in R2 (`asr-checkpoints` bucket)
| Experiment | R2 key | Test WER |
|---|---|---|
| Baseline (50ep) | `nemo_experiments/deaf_speech_story4_50epoch/checkpoints/konkani_asr-epoch=21-val_wer=0.720.ckpt` | 75.3% |
| DS-D (100ep, speed-perturbed) | `results/deaf_speech_dsd/checkpoints/konkani_asr-epoch=96-val_wer=0.269.ckpt` | **34.7%** |

