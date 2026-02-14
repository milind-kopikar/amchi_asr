# Agent start here — Amchi ASR project map

**Read this file first.** It is the single entry point to the repo: where everything is documented, how to continue from the last session, and how to run the full pipeline from scratch (install → finetune → checkpoints → inference endpoint).

---

## 1. Where we left off (last session)

- **Checkpoint** is in Cloudflare R2 (public URL, no expiry). Best run: 20 epochs, epoch 18 best, ~55% test WER.
- **RunPod Serverless endpoint** is *not* created yet. Remaining steps: build Docker image (no checkpoint in image), push to Docker Hub, create endpoint in RunPod with `CHECKPOINT_URL`, then test.
- **Single handoff file for “continue tomorrow”:** **[HANDOFF_SERVERLESS_RESUME.md](HANDOFF_SERVERLESS_RESUME.md)** — open it and follow “What you still need to do.”

**To continue:** Point the agent (or yourself) at **HANDOFF_SERVERLESS_RESUME.md** and execute the steps there.

---

## 2. Documentation index (where to look for what)

All important docs are listed below. Use this section to find the right file for the task.

### 2.1 Resume / endpoint / deployment (current focus)

| Doc | Use when |
|-----|----------|
| **[HANDOFF_SERVERLESS_RESUME.md](HANDOFF_SERVERLESS_RESUME.md)** | Resuming next day: build image, push, create RunPod serverless endpoint, test. Contains CHECKPOINT_URL and exact commands. |
| **[RUNPOD_SERVERLESS_DEPLOY.md](RUNPOD_SERVERLESS_DEPLOY.md)** | Full deploy guide: options A/B/C for checkpoint, build, push, create endpoint, test script. |
| **[RUNPOD_INFERENCE_ENDPOINT.md](RUNPOD_INFERENCE_ENDPOINT.md)** | Architecture: serverless vs persistent pod, request/response, web app flow, moving to Hugging Face later. |
| **[RUNPOD_R2_AND_IMAGE_HOSTING.md](RUNPOD_R2_AND_IMAGE_HOSTING.md)** | Why the Docker image goes in a registry (not R2); why checkpoints go in R2; flow summary. |
| **[R2_SETUP_CHECKPOINTS.md](R2_SETUP_CHECKPOINTS.md)** | Set up R2 bucket, API token, upload script, public URL. |
| **[runpod/README.md](runpod/README.md)** | Short pointer to handler vs FastAPI and main RunPod docs. |

### 2.2 Environment, training, reproduction (from scratch)

| Doc | Use when |
|-----|----------|
| **[MASTER_REPRODUCTION_GUIDE.md](MASTER_REPRODUCTION_GUIDE.md)** | **Single source of truth** for setup, verification, and training on a fresh environment (e.g. RunPod). Read this for “install everything and run training.” |
| **[SETUP_ENV.md](SETUP_ENV.md)** | Detailed environment setup: Python 3.11, NeMo, PyTorch, patches, preflight. |
| **[REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md)** | Technical details: CTC-only strategy, tokenizer, config edits, inference smoke test. |
| **[LEARNINGS.md](LEARNINGS.md)** | What works (Py3.11 + NeMo, smoke test, GPU check), what we don’t push to git. |
| **[README.md](README.md)** | Project overview, quick start, framework options. |

### 2.3 Data, model, RunPod (operational)

| Doc | Use when |
|-----|----------|
| **[RUNPOD_QUICK_START.md](RUNPOD_QUICK_START.md)** | Quick RunPod setup and commands. |
| **[RUNPOD_SETUP.md](RUNPOD_SETUP.md)** | RunPod setup and usage. |
| **[DATA_SNAPSHOT_AMCHI_KONKANI.md](DATA_SNAPSHOT_AMCHI_KONKANI.md)** | Data split convention (Story 4 = dev, Story 5 = test). |
| **[scripts/README_SMOKE.md](scripts/README_SMOKE.md)** | Smoke tests overview. |
| **[results/smoke_tests/README.md](results/smoke_tests/README.md)** | One-sample smoke test: how to run and validate. |

### 2.4 Other (reference only when needed)

- **AGENT_HANDOFF.md** — Older handoff; prefer HANDOFF_SERVERLESS_RESUME.md for endpoint work.
- **AI4BHARAT_SETUP_GUIDE.md**, **AI4BHARAT_MODEL_ACCESS.md** — Model access and AI4Bharat-specific setup.
- **KONKANI_MODEL_PLAN.md**, **TRAINING_RESULTS_2025-12-18.md** — Planning and past results.
- **DATA_*.md**, **MANIFEST_GUIDE.md**, **AUDIO_*.md** — Data and audio details when you need them.

---

## 3. Three ways to use this repo

### A. Continue from where we left off (serverless endpoint)

1. Read **[HANDOFF_SERVERLESS_RESUME.md](HANDOFF_SERVERLESS_RESUME.md)**.
2. Start a RunPod pod (or any machine with Docker and the repo).
3. Build image, push to Docker Hub, create serverless endpoint with `CHECKPOINT_URL`, test with `scripts/test_runpod_endpoint.py`.

### B. Start from scratch: full pipeline (install → train → checkpoint → endpoint)

1. **Environment:** Follow **[MASTER_REPRODUCTION_GUIDE.md](MASTER_REPRODUCTION_GUIDE.md)** § 1–2 and **[SETUP_ENV.md](SETUP_ENV.md)**. Use Python 3.11, upstream NeMo, RunPod (or similar GPU).
2. **Data:** Get data (e.g. `data/amchi/` with train/dev/test manifests). See MASTER_REPRODUCTION_GUIDE and **scripts/download_data_from_railway.py** if pulling from Railway.
3. **Verify:** Run preflight and smoke: `./scripts/run_all_preflight.sh` (includes GPU check). See **[LEARNINGS.md](LEARNINGS.md)** and **[results/smoke_tests/README.md](results/smoke_tests/README.md)**.
4. **Train:** e.g. `python scripts/fine_tune.py --config configs/marathi_amchi_20epoch.yaml` (or the config you use). Checkpoints go under `results/<run_name>/checkpoints/`.
5. **Checkpoint to R2:** **[R2_SETUP_CHECKPOINTS.md](R2_SETUP_CHECKPOINTS.md)** + `scripts/upload_checkpoint_to_r2.py --public-url`. Get public URL.
6. **Endpoint:** **[RUNPOD_SERVERLESS_DEPLOY.md](RUNPOD_SERVERLESS_DEPLOY.md)** — build image (no checkpoint), push to registry, create RunPod serverless endpoint with `CHECKPOINT_URL` = R2 public URL, test.

### C. Retrain / new checkpoint / change endpoint

- **New training run:** Same as B steps 2–4; use a new config or output dir. New checkpoints under `results/<new_run>/checkpoints/`.
- **New checkpoint to R2:** Run `scripts/upload_checkpoint_to_r2.py --file path/to/new.ckpt --public-url`. R2 key will mirror path (e.g. `results/.../checkpoints/...`). Get public URL from Cloudflare.
- **Point endpoint at new checkpoint:** In RunPod → endpoint → Edit → set **CHECKPOINT_URL** to the new public URL. No need to rebuild the Docker image.
- **Inference code / model changes:** Edit `scripts/amchi_inference.py`, `runpod/handler.py`, or training scripts as needed; then rebuild and push the serverless image if the handler or dependencies changed.

---

## 4. Key paths in the repo

| Purpose | Path |
|--------|------|
| Serverless handler (loads from CHECKPOINT_URL or CHECKPOINT_PATH) | `runpod/handler.py` |
| Shared inference (load .ckpt, transcribe) | `scripts/amchi_inference.py` |
| Dockerfile for serverless (no checkpoint in image) | `runpod/Dockerfile.serverless` |
| Upload checkpoint to R2 | `scripts/upload_checkpoint_to_r2.py` |
| Test RunPod endpoint (single file or manifest) | `scripts/test_runpod_endpoint.py` |
| Demo test-set WER with best checkpoint | `scripts/demo_test_set_wer.py` |
| Full preflight + smoke (includes GPU check) | `scripts/run_all_preflight.sh` |
| Fine-tuning entrypoint | `scripts/fine_tune.py` |
| 20-epoch training config | `configs/marathi_amchi_20epoch.yaml` |
| Best checkpoint (this run) in R2 (public URL) | See HANDOFF_SERVERLESS_RESUME.md or R2_SETUP_CHECKPOINTS.md |

---

## 5. One-line summary for the agent

- **To continue tomorrow:** Open **HANDOFF_SERVERLESS_RESUME.md** and do the steps (build image, push, create endpoint, test).
- **To understand the whole project:** Read **MASTER_REPRODUCTION_GUIDE.md** and **RUNPOD_INFERENCE_ENDPOINT.md**; use this file (AGENT_START_HERE.md) to find any other doc.
- **To run from scratch:** Follow section 3.B above; docs are linked there.
- **To add a new checkpoint or point endpoint elsewhere:** Section 3.C and **R2_SETUP_CHECKPOINTS.md** + RunPod endpoint env var.

All important information is in the docs listed in § 2; this file is the map to them.
