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

## 6. What We Do Not Push to Git

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
