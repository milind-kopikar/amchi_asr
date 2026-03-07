# AGENT START HERE — Amchi ASR Project

**This is the single entry point.** Read this file first. It tells you the current
state of every experiment, where all checkpoints are, and exactly which doc to read
for the task you need to do. All other docs are referenced from here.

**Last updated: 2026-03-07**

---

## 1. What is this project?

Fine-tuning the AI4Bharat IndicConformer ASR model (Marathi base) for two use cases:
1. **Amchi Konkani** — speech recognition for Konkani language recordings
2. **Deaf Speech** — recognising speech from deaf/hard-of-hearing speakers (Marathi story 4)

Both use the same base model, training pipeline, and CTC-only fine-tuning strategy.

---

## 2. Current experiment results (as of 2026-03-07)

### Amchi Konkani

| Experiment | Split | Encoder | Test WER | Status |
|---|---|---|---|---|
| Baseline (50ep) | Story-based | Full FT | 54.7% | Done |
| **Run C** (100ep) | Story-based | **Frozen** | **49.1%** | Done ✓ |
| **Run S** (100ep) | **Speaker-stratified** | **Frozen** | **34.1%** | Done ✓ ⭐ BEST |

**Key finding:** Stratified split (ensuring all test speakers appear in training) reduced
WER by 15pp over story-based split. Frozen encoder prevents overfitting on small dataset.

### Deaf Speech

| Experiment | Data | Encoder | Test WER | Status |
|---|---|---|---|---|
| Baseline (50ep) | 124 samples | Full FT | 75.3% | Done |
| DS-A (100ep) | 124 samples | Frozen | 79.6% | Done (worse — freeze hurts deaf speech) |
| DS-B (100ep) | 188 samples (extended) | Full FT | 93.1% | Done (worse — OOD data hurts) |
| **DS-D** (100ep) | **372 samples (3× speed)** | **Full FT** | **34.7%** | Done ✓ ⭐ BEST |

**Key finding:** Speed perturbation (0.9×/1.0×/1.1×) on the same 124 samples gives 3×
training data from the same distribution → 40.6pp WER improvement over baseline.

---

## 3. Where are the checkpoints?

**→ Full details: [`docs/CHECKPOINTS_REGISTRY.md`](docs/CHECKPOINTS_REGISTRY.md)**

Quick reference — the two production-ready checkpoints in R2:

| Model | R2 public URL |
|---|---|
| **Amchi Konkani Run S** | `https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev/results/run_c_stratified_split/checkpoints/konkani_asr-epoch=88-val_wer=0.334.ckpt` |
| **Deaf Speech DS-D** | `https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev/results/deaf_speech_dsd/checkpoints/konkani_asr-epoch=96-val_wer=0.269.ckpt` |

R2 bucket: `asr-checkpoints` | Account ID: `c90f9011c5a59d5bf40c808f40e3e34b`

---

## 4. Module guide — "I need to do X, read Y"

This project is organised into four modules. Each has a self-contained doc in `docs/`.

### 4.1 Training a model
**→ [`docs/MODULE_TRAINING.md`](docs/MODULE_TRAINING.md)**

Covers: RunPod environment setup, applying the conv_asr patch, running `fine_tune.py`,
speed perturbation, choosing freeze vs. full FT, monitoring, common errors.

Key script: `scripts/fine_tune.py` | Key configs: `configs/*.yaml`

### 4.2 Running inference on audio
**→ [`docs/MODULE_INFERENCE.md`](docs/MODULE_INFERENCE.md)**

Covers: the non-obvious checkpoint loading pattern (manual config patch + strict=False),
CTC decoding, Gemini post-processing, latency expectations, audio requirements.

Key scripts: `scripts/deaf_speech_inference.py`, `scripts/postprocess_asr.py`

### 4.3 Building a serverless endpoint (Docker → RunPod)
**→ [`docs/MODULE_SERVERLESS.md`](docs/MODULE_SERVERLESS.md)**

Covers: Docker build/push, RunPod endpoint creation, environment variables,
checkpoint URLs to use, handler input/output format, cold start behaviour.
**Build on your local machine — not on the RunPod pod.**

Key files: `runpod/handler_deaf.py`, `runpod/Dockerfile.deaf`

### 4.4 Checkpoint locations and R2 storage
**→ [`docs/CHECKPOINTS_REGISTRY.md`](docs/CHECKPOINTS_REGISTRY.md)**

Covers: every experiment's best checkpoint with local path, R2 key, and public URL.
Also covers the base model location and how to upload new checkpoints.

---

## 5. Data

### Deaf Speech (story_id=22, "दैनंदिन कामे १")
| Path | Contents | In git? |
|---|---|---|
| `data/deaf_speech/audio/` | 124 WAV files, 16kHz mono | NO (too large) |
| `data/deaf_speech/train/manifest.jsonl` | 124-sample train manifest | Yes |
| `data/deaf_speech/dev/manifest.jsonl` | 124-sample dev manifest | Yes |
| `data/deaf_speech/test/manifest.jsonl` | 124-sample test manifest | Yes |
| `data/deaf_speech_sp/audio/` | 372 speed-perturbed WAVs (DS-D train data) | NO |
| `data/deaf_speech_sp/train/manifest.jsonl` | DS-D training manifest | Yes |

Re-download audio: `python3 scripts/download_data_from_railway.py`
Railway API: `https://deafspeechcollector-production.up.railway.app/`

### Amchi Konkani
| Path | Contents | In git? |
|---|---|---|
| `data/amchi/audio/` | WAV files per story | NO |
| `data/amchi/{train,dev,test}/manifest.jsonl` | Story-based split (Run C) | Yes |
| `data/amchi_stratified/{train,dev,test}/manifest.jsonl` | Speaker-stratified split (Run S) | Yes |

Railway API: `https://konkanicollector-production.up.railway.app/`

---

## 6. Environment — key facts

| Item | Value |
|---|---|
| Python | 3.11 |
| NeMo | `nemo_toolkit[asr]` v2.7.0 (upstream — NOT the AI4Bharat fork) |
| PyTorch | Must match pod CUDA — reinstall after NeMo (see MODULE_TRAINING.md) |
| GPU required | Yes — RTX 4000 Ada (20GB) or A40 (48GB) recommended |
| Root disk | 20GB on RunPod — run `pip cache purge` if low on space |
| All secrets | `.env` file (git-ignored). Template: `.env.example` |

**Required after every RunPod restart:**
```bash
pip install "nemo_toolkit[asr]" --ignore-installed blinker -q
pip install --force-reinstall torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128 -q
NEMO_FILE=$(python3 -c "import nemo.collections.asr.modules.conv_asr as m; print(m.__file__)" 2>&1 | tail -1)
cp patches/conv_asr_fixed.py "$NEMO_FILE"
export APPLY_CONV_PATCH=1
```

---

## 7. Git workflow

```bash
source .env
git remote set-url origin https://${GITHUB_PAT}@github.com/milind-kopikar/amchi_asr.git
git push origin master
```

**Committed:** configs, scripts, manifests (*.jsonl), result JSONs/CSVs, docs, patches.
**Not committed:** audio (*.wav), weights (*.ckpt, *.nemo, *.pt), venvs, pip cache, `.env`.

---

## 8. Documentation map

### Current and maintained
| Doc | What it covers |
|---|---|
| **`AGENT_START_HERE.md`** (this file) | Project hub, current results, navigation |
| **`docs/CHECKPOINTS_REGISTRY.md`** | All checkpoint R2 URLs, local paths, base model |
| **`docs/MODULE_TRAINING.md`** | How to train / retrain — environment, configs, scripts |
| **`docs/MODULE_INFERENCE.md`** | How to load a checkpoint and transcribe audio |
| **`docs/MODULE_SERVERLESS.md`** | How to build Docker image and deploy to RunPod |
| **`LEARNINGS.md`** | Hard-won lessons from every session (read before starting) |
| **`AGENT_HANDOFF.md`** | Session-by-session build history and rationale |

### Detailed reference (consult when needed)
| Doc | What it covers |
|---|---|
| `MASTER_REPRODUCTION_GUIDE.md` | Full from-scratch setup guide |
| `REPRODUCTION_NOTES.md` | CTC-only loading strategy, inference smoke test recipe |
| `SETUP_ENV.md` | Detailed environment setup steps |
| `RUNPOD_SERVERLESS_DEAF.md` | Deep-dive: deaf speech serverless endpoint |
| `RUNPOD_SERVERLESS_AMCHI_KONKANI.md` | Deep-dive: Konkani serverless endpoint |
| `R2_SETUP_CHECKPOINTS.md` | How to create R2 bucket and API tokens from scratch |
| `DATA_SNAPSHOT_AMCHI_KONKANI.md` | Konkani story-split convention |
| `AMCHI_KONKANI_NEXT_EXPERIMENTS.md` | Step-by-step guide for Run C and Run S |

### Historical (earlier sessions, may be partially outdated)
`HANDOFF_SERVERLESS_RESUME.md`, `AMCHI_KONKANI_FINETUNING_TODOS.md`,
`KONKANI_MODEL_PLAN.md`, `TRAINING_RESULTS_2025-12-18.md`, `DEMO_WEBAPP_GUIDE.md`,
`PROJECT_STATUS.md`, `DEPLOYMENT.md`, `ARCHITECTURE.md`, `ROADMAP.md`,
and other pre-2026 docs in the repo root.

---

## 9. What to do next

### Deploy a serverless endpoint
1. Read **`docs/MODULE_SERVERLESS.md`**
2. Build locally: `docker build -f runpod/Dockerfile.deaf -t deaf-speech-asr-runpod .`
3. Push to DockerHub, create RunPod endpoint
4. Use DS-D checkpoint URL from `docs/CHECKPOINTS_REGISTRY.md`

### Run inference locally
1. Read **`docs/MODULE_INFERENCE.md`**
2. Run `scripts/deaf_speech_inference.py` with a WAV file path

### Train a new experiment
1. Read **`docs/MODULE_TRAINING.md`**
2. Copy an existing config from `configs/`, edit it, run `fine_tune.py`
3. Upload best checkpoint with `scripts/upload_checkpoint_to_r2.py`
4. Add the R2 URL to `docs/CHECKPOINTS_REGISTRY.md`
5. Update the results table in this file (Section 2)
