# Phase 2: RunPod Serverless Endpoint — Deaf Speech ASR

Step-by-step guide to deploy and test the deaf speech inference endpoint.

---

## Architecture

```
Web App  ──POST { audio_base64 }──►  RunPod Serverless Endpoint
                                         │
                                    handler_deaf.py
                                         │
                               ┌─────────┴──────────┐
                               │                    │
                          NeMo ASR              Gemini 2.5 Flash
                      (deaf speech              post-processing
                       checkpoint)              (FILL/RECONSTRUCT)
                               │                    │
                               └─────────┬──────────┘
                                         │
                                  { raw, corrected,
                                    mode, latency_ms }
```

**Checkpoint location:** Cloudflare R2 bucket `asr-checkpoints`
**Docker image:** Docker Hub (no checkpoint inside — downloaded from R2 at worker start)

---

## Key files

| File | Purpose |
|------|---------|
| `runpod/handler_deaf.py` | RunPod serverless handler — loads model, runs ASR + post-processing |
| `runpod/Dockerfile.deaf` | Docker image build spec (NeMo + google-genai + runpod SDK) |
| `scripts/test_deaf_endpoint.py` | Test script — single file, URL, or all 124 samples |
| `scripts/upload_checkpoint_to_r2.py` | Upload .ckpt to R2 |

---

## Step 1: Upload checkpoint to Cloudflare R2

**DONE (2026-03-01).** The checkpoint has already been uploaded.

Public URL (no expiry):
```
https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev/nemo_experiments/deaf_speech_story4_50epoch/checkpoints/konkani_asr-epoch=21-val_wer=0.720.ckpt
```

R2 bucket: `asr-checkpoints`
Object key: `nemo_experiments/deaf_speech_story4_50epoch/checkpoints/konkani_asr-epoch=21-val_wer=0.720.ckpt`

> To re-upload in future (e.g. different checkpoint), run from the RunPod pod:
> ```bash
> pip install boto3 -q
> export R2_ACCOUNT_ID="..." R2_ACCESS_KEY_ID="..." R2_SECRET_ACCESS_KEY="..." R2_BUCKET_NAME="asr-checkpoints"
> python3 scripts/upload_checkpoint_to_r2.py --file <path/to/new.ckpt> --public-url
> ```

**You can now stop and terminate the RunPod pod** — the checkpoint is safely on R2.

---

## Step 2: Build Docker image (on your local machine)

```bash
# Pull the latest repo
git clone https://github.com/milind-kopikar/amchi_asr.git   # or git pull
cd amchi_asr

# Build the deaf speech image (~15–20 min first time: downloads PyTorch + NeMo)
docker build -f runpod/Dockerfile.deaf -t deaf-speech-asr-runpod .
```

No checkpoint is baked in. The image is ~8–10 GB.

---

## Step 3: Push to Docker Hub

```bash
docker login
docker tag deaf-speech-asr-runpod YOUR_DOCKERHUB_USERNAME/deaf-speech-asr-runpod:latest
docker push YOUR_DOCKERHUB_USERNAME/deaf-speech-asr-runpod:latest
```

Replace `YOUR_DOCKERHUB_USERNAME` with your Docker Hub username.

---

## Step 4: Create RunPod Serverless endpoint

1. Go to **RunPod Console → Serverless → New Endpoint**
2. **Image:** `YOUR_DOCKERHUB_USERNAME/deaf-speech-asr-runpod:latest`
3. **GPU:** RTX 4000 Ada or A40 (≥ 20 GB VRAM recommended; T4 works but slower)
4. **Container disk:** 20 GB
5. **No Network Volume needed** (checkpoint downloads from R2)
6. **Environment variables** — add both:

| Name | Value |
|------|-------|
| `CHECKPOINT_URL` | `https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev/nemo_experiments/deaf_speech_story4_50epoch/checkpoints/konkani_asr-epoch=21-val_wer=0.720.ckpt` ← confirmed, no expiry |
| `GEMINI_API_KEY` | your Gemini API key (`AIzaSy…`) — read from `.env` |

7. **Workers:** Min 0, Max 1 (scale up for demo days)
8. Create endpoint → note the **Endpoint ID**

---

## Step 5: Test the endpoint

```bash
cd amchi_asr
export RUNPOD_API_KEY="your_runpod_api_key"      # Settings → API Keys in RunPod
export RUNPOD_ENDPOINT_ID="your_endpoint_id"

# Single file (local WAV):
python3 scripts/test_deaf_endpoint.py \
  --audio data/deaf_speech/audio/131.wav \
  --reference "दूध किती आहे?"

# Single file (Railway URL — no local file needed):
python3 scripts/test_deaf_endpoint.py \
  --audio_url "https://deafspeechcollector-production.up.railway.app/api/recordings/131/audio" \
  --reference "दूध किती आहे?"

# All 124 samples:
python3 scripts/test_deaf_endpoint.py --all_samples
```

Expected output for sample 131:
```
────────────────────────────────────────────────────────────────
  Raw ASR   : ू किती ⁇  [ASR only]
  Corrected : हे किती आहे?  [FILL]
  Reference : दूध किती आहे?
  WER raw   : 66.7%
  WER corr  : 33.3%  ↓ improved
  Latency   : ASR 270ms | Post-process 1450ms | Total 1720ms
────────────────────────────────────────────────────────────────
```

The **first request** will be slow (cold start: download 1.4 GB checkpoint + NeMo model load = ~90s).
Subsequent requests from the same warm worker: ~2s total.

---

## Demo day warm-up procedure

Before a demo, send one warm-up request 5–10 minutes early:

```bash
# Warm-up ping (any short audio works)
python3 scripts/test_deaf_endpoint.py \
  --audio_url "https://deafspeechcollector-production.up.railway.app/api/recordings/131/audio"
```

Then set **min workers = 1** in the RunPod endpoint settings so the worker stays warm during the demo.
After the demo, set **min workers = 0** to stop paying.

Cost with min_workers=1 (RTX 4000 Ada): ~$0.44/hr → ~$1.30 for a 3-hour demo session.

---

## Local smoke test (no RunPod, no Docker)

To verify the handler works on your machine before deploying:

```bash
# Install deps (one-time; same as Dockerfile steps 1-3):
pip install "nemo_toolkit[asr]" --ignore-installed blinker
pip install google-genai jiwer runpod

# Apply conv_asr patch:
python3 -c "import nemo.collections.asr.modules.conv_asr as m; import shutil; shutil.copy('patches/conv_asr_fixed.py', m.__file__)"

# Set env vars:
export CHECKPOINT_PATH="nemo_experiments/deaf_speech_story4_50epoch/checkpoints/konkani_asr-epoch=21-val_wer=0.720.ckpt"
export GEMINI_API_KEY="your_key"

# Run with --local flag:
python3 scripts/test_deaf_endpoint.py \
  --audio data/deaf_speech/audio/131.wav \
  --reference "दूध किती आहे?" \
  --local
```

---

## Handler input/output reference

**Request body** (`job["input"]`):

```json
{ "audio_base64": "<base64-encoded 16 kHz mono WAV>" }
```
or
```json
{ "audio_url": "https://..." }
```

**Response** (`job["output"]`):

```json
{
  "raw":        "ू किती ⁇",
  "corrected":  "हे किती आहे?",
  "mode":       "FILL",
  "latency_ms": { "asr": 270, "postprocess": 1450, "total": 1720 }
}
```

**Error response:**
```json
{ "error": "..." }
```

**Mode values:**

| Mode | Meaning |
|------|---------|
| `FILL` | Gemini filled garbled slots using anchor words |
| `RECONSTRUCT` | Gemini reconstructed full sentence from fragments |
| `FILL_REVERTED` | Fill attempted but safety valve reverted (Gemini worsened output) |
| `SKIP` | Perfect transcription (WER=0), no post-processing needed |
| `SKIPPED` | GEMINI_API_KEY not set; raw ASR returned as-is |
| `PP_ERROR` | Post-processing API call failed; raw ASR returned |

---

## Connecting Phase 2 to the Phase 1 web app

In the Next.js web app, add a "Live Inference" toggle:

1. Add env var to Railway: `NEXT_PUBLIC_RUNPOD_ENDPOINT_ID` and `NEXT_PUBLIC_RUNPOD_API_KEY`
2. Replace the `setTimeout` in the Transcribe handler with a real fetch to:
   ```
   https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync?wait=120000
   ```
   with body `{ "input": { "audio_base64": "<base64>" } }`
3. The response shape (`raw`, `corrected`, `mode`, `latency_ms`) is the same as pre-computed data.

See `DEMO_WEBAPP_GUIDE.md` §9 for the exact frontend code.
