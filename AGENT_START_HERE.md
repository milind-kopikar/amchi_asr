# Agent start here — Amchi ASR project map

**Read this file first.** It is the single entry point to the repo: where everything is documented, how to continue from the last session, and how to run the full pipeline.

---

## ⚡ CURRENT TASK (2026-03-07) — Two new Amchi Konkani experiments

**Read [`AMCHI_KONKANI_NEXT_EXPERIMENTS.md`](AMCHI_KONKANI_NEXT_EXPERIMENTS.md) for the
complete step-by-step guide.** It covers full RunPod setup, both experiments, evaluation,
and how to save results back to GitHub.

| | Experiment 1 — Run C | Experiment 2 — Run S |
|---|---|---|
| Config | `configs/amchi_konkani_run_c.yaml` | `configs/amchi_konkani_run_c_stratified.yaml` |
| Data split | Story-based (existing) | Speaker-stratified (generate first) |
| Generate split? | No | Yes — `python3 scripts/create_speaker_stratified_split.py ...` |
| Freeze encoder? | Yes (132K trainable params) | Yes (132K trainable params) |
| Key question | Does freezing fix overfitting? | Does dipti WER improve with 27 train samples? |

Do the RunPod environment setup once (§3 of the experiment guide), then run both.

---

## 1. Where we left off (last updated 2026-03-07)

### Active track: Amchi Konkani ASR — analysis complete, two experiments queued

- **50-epoch run COMPLETE:** Test WER **54.7%** (Story 5, 104 samples, 3 speakers).
- **Root cause analysis done** (`scripts/analyze_runs_comparison.py`, results in `results/amchi_analysis/`):
  - Overfitting: all 115M encoder params trained on 511 samples; CTC val_loss 1.86× worse by epoch 49
  - Speaker imbalance: dipti had only 3 train samples but 35 test samples → WER 60.3%
  - RNNT does NOT corrupt the encoder (training_step is CTC-only monkey-patched)
  - Pilot 35.1% was on val=test same file (selection bias) + single speaker
- **Two experiments designed** — see `AMCHI_KONKANI_NEXT_EXPERIMENTS.md`

**Key environment note:** Fresh pods have PyTorch cu128 — after `pip install nemo_toolkit[asr]`, run:
```bash
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 -q
```
See LEARNINGS.md §6 for full details.

---

### Deaf Speech track (done — inference endpoint pending)

- **Fine-tuning COMPLETE:** 50 epochs on 124 deaf speech recordings. Best checkpoint at **epoch 21, val_WER=72.0%**.
- **Post-processing BUILT:** `scripts/postprocess_asr.py` (Gemini FILL/RECONSTRUCT). WER 75.3% → 74.2%, but human readability improved significantly.
- **What remains:** Build RunPod serverless inference endpoint (see RUNPOD_SERVERLESS_DEAF.md).

**To continue deaf speech track:** Read **[AGENT_HANDOFF.md](AGENT_HANDOFF.md)**.

---

## 2. Documentation index (where to look for what)

### 2.1 Next focus: Amchi Konkani fine-tuning

| Doc | Use when |
|-----|----------|
| **[AMCHI_KONKANI_NEXT_EXPERIMENTS.md](AMCHI_KONKANI_NEXT_EXPERIMENTS.md)** | **⚡ START HERE.** Step-by-step guide for Run C and Run S. Covers setup, data, training, evaluation, saving results. |
| **[AMCHI_KONKANI_FINETUNING_TODOS.md](AMCHI_KONKANI_FINETUNING_TODOS.md)** | Broader list of optimisation runs (A–E) with rationale and results tracking table. |
| `configs/amchi_konkani_run_c.yaml` | Run C: freeze encoder + cosine LR + 100 epochs (story-based split). |
| `configs/amchi_konkani_run_c_stratified.yaml` | Run S: same as Run C but with speaker-stratified data paths. |
| `scripts/create_speaker_stratified_split.py` | Generates `data/amchi_stratified/` manifests for Run S. Run before training. |
| `scripts/analyze_runs_comparison.py` | Statistical comparison of any two runs vs pilot. Run locally after results downloaded. |
| **[DATA_SNAPSHOT_AMCHI_KONKANI.md](DATA_SNAPSHOT_AMCHI_KONKANI.md)** | Documents the story-based split convention. |

### 2.2 Deaf speech (done) — inference endpoint

| Doc | Use when |
|-----|----------|
| **[AGENT_HANDOFF.md](AGENT_HANDOFF.md)** | Full session history and current state of deaf speech track. |
| **[RUNPOD_SERVERLESS_DEAF.md](RUNPOD_SERVERLESS_DEAF.md)** | Build/deploy/test the RunPod serverless endpoint (Docker image not yet built). |
| **[DEMO_WEBAPP_GUIDE.md](DEMO_WEBAPP_GUIDE.md)** | Phase 1 web app: build Next.js demo on local machine → Railway. |
| **[REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md)** | CTC-only loading strategy, tokenizer fix, inference smoke test recipe. **Critical reading for inference code.** |

### 2.2 Post-processing module

| File | Description |
|------|-------------|
| `scripts/postprocess_asr.py` | Gemini-powered post-processor. FILL mode (anchor words present) + RECONSTRUCT mode (all garbled). Conservative safety valve prevents WER regression. |
| `nemo_experiments/deaf_speech_story4_50epoch/experiments/20260301_003725/postprocessed_results.json` | Full 124-sample post-processing results (WER before/after + corrected text). |
| `nemo_experiments/deaf_speech_story4_50epoch/experiments/20260301_003725/postprocess_report.txt` | Human-readable sentence-by-sentence comparison report. |

### 2.3 Environment, training, reproduction (from scratch)

| Doc | Use when |
|-----|----------|
| **[MASTER_REPRODUCTION_GUIDE.md](MASTER_REPRODUCTION_GUIDE.md)** | **Single source of truth** for setup and training on a fresh RunPod instance. |
| **[SETUP_ENV.md](SETUP_ENV.md)** | Detailed environment setup: Python 3.11, NeMo v2.7.0, PyTorch, patches. |
| **[REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md)** | CTC-only strategy, tokenizer discovery, inference smoke test. |
| **[LEARNINGS.md](LEARNINGS.md)** | Accumulated hard-won lessons across all sessions. |

### 2.4 Data

| Doc/Path | Description |
|----------|-------------|
| `data/deaf_speech/audio/` | 124 WAV files (story_id=22 deaf recordings, 16kHz mono) |
| `data/deaf_speech/train/manifest.jsonl` | Training manifest (all 124 samples) |
| `data/deaf_speech/dev/manifest.jsonl` | Dev manifest (same 124 samples) |
| `data/deaf_speech/test/manifest.jsonl` | Test manifest (same 124 samples) |
| `configs/deaf_speech_story4_50epoch.yaml` | Training config used for this run |
| `scripts/download_data_from_railway.py` | Downloads data from Railway API |
| **[DATA_SNAPSHOT_AMCHI_KONKANI.md](DATA_SNAPSHOT_AMCHI_KONKANI.md)** | Amchi Konkani data split convention |

### 2.5 Model and checkpoints

| Path | Description |
|------|-------------|
| `models/indicconformer_stt_mr_hybrid_ctc_rnnt_large/indicconformer_stt_mr_hybrid_rnnt_large.nemo` | AI4Bharat base Marathi model (499MB) |
| `tokenizers/marathi_tokenizer.model` | Correct Marathi SentencePiece tokenizer (extracted from .nemo) |
| `nemo_experiments/deaf_speech_story4_50epoch/checkpoints/konkani_asr-epoch=21-val_wer=0.720.ckpt` | **Best checkpoint** (5.3GB total for top-3 + last) |

### 2.6 Other (reference only when needed)

- **HANDOFF_SERVERLESS_RESUME.md** — Earlier serverless endpoint guide (Amchi Konkani). Refer only if building serverless (not persistent pod).
- **RUNPOD_QUICK_START.md**, **RUNPOD_SETUP.md** — RunPod general usage.
- **KONKANI_MODEL_PLAN.md**, **TRAINING_RESULTS_2025-12-18.md** — Amchi Konkani planning and past results.

---

## 3. How to continue from here

### A. Build the deaf speech inference endpoint (tomorrow's main task)

1. Read **[AGENT_HANDOFF.md](AGENT_HANDOFF.md)** for the exact recipe.
2. Build an inference script (`scripts/deaf_speech_inference.py`) based on the pattern in `REPRODUCTION_NOTES.md` § 9 (Inference Smoke Test Strategy).
3. Test on a sample WAV from `data/deaf_speech/audio/` and measure latency.
4. Run post-processing (`scripts/postprocess_asr.py`) on the output and show side-by-side.
5. (Optional) Wrap in a RunPod serverless handler.

### B. Start from scratch (fresh RunPod)

1. **Environment:** `bash setup_env.sh` or follow MASTER_REPRODUCTION_GUIDE.md. Python 3.11, upstream NeMo (`nemo_toolkit[asr]`).
2. **Model:** Download via `scripts/download_model_from_hf.py` with HF token (see AGENT_HANDOFF.md).
3. **Data:** Already committed in `data/deaf_speech/*/manifest.jsonl`. Audio is NOT in git (too large — re-download from Railway: `python3 scripts/download_data_from_railway.py`).
4. **Train:** `export APPLY_CONV_PATCH=1 && python3 scripts/fine_tune.py --config configs/deaf_speech_story4_50epoch.yaml`
5. **Post-process:** `python3 scripts/postprocess_asr.py --input <final_test_results.json> --output <out.json> --report <out.txt> --api_key <GEMINI_KEY>`

### C. Start Amchi Konkani training (second track, when ready)

Follow the same methodology as Section B but:
- Change model to `models/konkani_model.nemo` and tokenizer to `tokenizers/konkani_tokenizer.model`
- Use `data/train`, `data/dev`, `data/test` (Amchi Konkani data)
- Post-processing for Konkani will be different (discuss with user first)

---

## 4. Key paths summary

| Purpose | Path |
|---------|------|
| **Best deaf speech checkpoint** | `nemo_experiments/deaf_speech_story4_50epoch/checkpoints/konkani_asr-epoch=21-val_wer=0.720.ckpt` |
| Training config (deaf speech story 4) | `configs/deaf_speech_story4_50epoch.yaml` |
| Fine-tuning entrypoint | `scripts/fine_tune.py` |
| Post-processing script | `scripts/postprocess_asr.py` |
| Post-processing results (JSON) | `nemo_experiments/deaf_speech_story4_50epoch/experiments/20260301_003725/postprocessed_results.json` |
| Post-processing report (text) | `nemo_experiments/deaf_speech_story4_50epoch/experiments/20260301_003725/postprocess_report.txt` |
| Base Marathi model | `models/indicconformer_stt_mr_hybrid_ctc_rnnt_large/indicconformer_stt_mr_hybrid_rnnt_large.nemo` |
| Marathi tokenizer | `tokenizers/marathi_tokenizer.model` |
| Data manifests (deaf speech) | `data/deaf_speech/{train,dev,test}/manifest.jsonl` |
| Data audio (deaf speech, NOT in git) | `data/deaf_speech/audio/*.wav` |
| NeMo conv_asr patch | `patches/conv_asr_fixed.py` |

---

## 5. One-line summary for the agent

- **To continue tomorrow:** Read **AGENT_HANDOFF.md** → build `scripts/deaf_speech_inference.py` → test with sample audio → measure latency → show post-processing side-by-side.
- **To understand the whole project:** Read **MASTER_REPRODUCTION_GUIDE.md** and **REPRODUCTION_NOTES.md** (especially the Inference Smoke Test Strategy section).
- **To retrain:** Follow section 3.B above. The `.gitignore` is set up to commit configs, scripts, manifests, and JSON results but NOT audio, checkpoints, or model weights.
- **To run post-processing standalone:** `python3 scripts/postprocess_asr.py --help`

All important information is in the docs listed above; this file is the map to them.
