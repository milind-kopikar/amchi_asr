# RunPod Serverless Endpoint — Amchi Konkani ASR
## Build Guide for Local Claude Agent

**Read this first. This is a complete, self-contained guide to build, deploy, and test the Amchi Konkani serverless inference endpoint from a local machine.**

---

## 0. Current status

| Step | Status | Notes |
|------|--------|-------|
| Handler code (`runpod/handler.py`) | ✅ Done | In this repo — uses `scripts/amchi_inference.py` |
| Dockerfile (`runpod/Dockerfile.serverless`) | ✅ Done | In this repo |
| Inference script (`scripts/amchi_inference.py`) | ✅ Done | In this repo |
| Checkpoint → Cloudflare R2 | ✅ Done | Uploaded 2026-03-02 (see §2) |
| Public R2 URL enabled | ❌ TODO | Enable public access on `asr-checkpoints` bucket (§2) |
| Docker image built | ❌ TODO | Build on local machine (§4) |
| Docker image pushed to Docker Hub | ❌ TODO | Push to `milindkopi` account (§5) |
| RunPod Serverless endpoint created | ❌ TODO | In RunPod Console (§6) |
| Endpoint tested | ❌ TODO | Using test script (§7) |

---

## 1. Credentials the agent needs to ask for

Before starting, ask the user for these values. Do NOT commit them to git.

| Credential | Where to find it | Used in step |
|------------|-----------------|--------------|
| **Docker Hub password** (or PAT) | hub.docker.com → Account Settings → Security → New Access Token | Step 5 (`docker login`) |
| **RunPod API key** | RunPod Console → Settings → API Keys | Step 7 (test script) |
| **Cloudflare R2 credentials** (if public URL not yet enabled) | See §2 | Step 2 only |

Docker Hub username: **`milindkopi`**

---

## 2. Enable public access on R2 checkpoint (REQUIRED before deployment)

The 50-epoch Amchi Konkani checkpoint was uploaded to Cloudflare R2 on 2026-03-02.

**Bucket:** `asr-checkpoints`
**Account ID:** `c90f9011c5a59d5bf40c808f40e3e34b`
**Object key:** `results/checkpoints/konkani_asr-epoch=47-val_wer=0.532.ckpt`
**R2 endpoint:** `https://c90f9011c5a59d5bf40c808f40e3e34b.r2.cloudflarestorage.com`

### To get a public (no-expiry) URL:
1. Go to [Cloudflare Dashboard → R2](https://dash.cloudflare.com) → `asr-checkpoints` bucket
2. Click **Settings** → **Public access** → **Allow Access**
3. Note the public bucket URL shown (looks like `https://pub-XXXX.r2.dev`)
4. Your checkpoint URL is: `https://pub-XXXX.r2.dev/results/checkpoints/konkani_asr-epoch=47-val_wer=0.532.ckpt`
5. Verify: `curl -I "<that URL>"` should return HTTP 200

Set this URL as `CHECKPOINT_URL` in the RunPod endpoint (§6).

### Alternative: presigned URL (7-day expiry)
If you'd rather not enable public access, generate a time-limited presigned URL:
```bash
cd /path/to/amchi_asr
export R2_ACCOUNT_ID="c90f9011c5a59d5bf40c808f40e3e34b"
export R2_ACCESS_KEY_ID="<ask user for access key>"
export R2_SECRET_ACCESS_KEY="<ask user for secret>"
export R2_BUCKET_NAME="asr-checkpoints"
python3 scripts/upload_checkpoint_to_r2.py \
  --file results/checkpoints/konkani_asr-epoch=47-val_wer=0.532.ckpt \
  --presigned-expiry 604800
```
This prints a presigned URL valid for 7 days. Use it as `CHECKPOINT_URL` in §6. Regenerate before it expires.

---

## 3. What's in the repo (handler and inference code)

These files already exist and are committed:

| File | Purpose |
|------|---------|
| `runpod/handler.py` | RunPod handler — downloads checkpoint, loads model, transcribes audio |
| `runpod/Dockerfile.serverless` | Docker image definition for the Amchi Konkani worker |
| `scripts/amchi_inference.py` | Shared inference: `load_model_from_ckpt()`, `transcribe_audio()`, `transcribe_audio_bytes()` |
| `patches/conv_asr_fixed.py` | Required NeMo patch applied during Docker build |

The handler accepts:
- `{ "audio_base64": "<base64 WAV>" }` — preferred for web app
- `{ "audio_url": "<URL to 16 kHz mono WAV>" }` — for testing

The handler returns:
```json
{ "transcription": "<Devanagari text>" }
```
or on error:
```json
{ "error": "description" }
```

> **Note:** This endpoint returns raw ASR output with no Gemini post-processing.
> Post-processing (using `scripts/postprocess_asr.py`) can be run client-side.
> See §8 if you want to add post-processing to the handler in a future iteration.

---

## 4. Prerequisites on the local machine

```bash
# Verify Docker is installed and running
docker --version   # need 20.x or later
docker info        # must succeed

# Pull or clone the repo
git clone https://github.com/milind-kopikar/amchi_asr.git
cd amchi_asr
# OR if already cloned:
git pull

# Verify required files exist
ls runpod/Dockerfile.serverless     # must exist
ls runpod/handler.py                # must exist
ls scripts/amchi_inference.py       # must exist
ls patches/conv_asr_fixed.py        # must exist
```

---

## 5. Build the Docker image

Run from the **repo root** (`amchi_asr/`):

```bash
docker build -f runpod/Dockerfile.serverless -t amchi-asr-runpod .
```

This will:
1. Pull `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` as base
2. Install Python 3.11, ffmpeg, build tools
3. Install PyTorch (CUDA 11.8) — ~2 GB download
4. Install NeMo ASR toolkit (`nemo_toolkit[all]`) — ~3–4 GB download
5. Install RunPod SDK
6. Copy repo code into `/app`
7. Apply `conv_asr_fixed.py` patch to NeMo

**Expected duration:** 20–35 minutes on first build (large downloads). Subsequent builds use Docker layer cache.

**Expected final image size:** ~10–13 GB

When complete you should see:
```
Successfully built <image_id>
Successfully tagged amchi-asr-runpod:latest
```

If the build fails, see §9 (Troubleshooting).

---

## 6. Push to Docker Hub

```bash
# Log in (enter Docker Hub password / PAT when prompted)
docker login --username milindkopi

# Tag with full Docker Hub path (include the 50epoch tag to distinguish from pilot run)
docker tag amchi-asr-runpod milindkopi/amchi-asr-runpod:50epoch

# Also tag as latest for convenience
docker tag amchi-asr-runpod milindkopi/amchi-asr-runpod:latest

# Push (uploads ~10-13 GB — takes 5–20 min depending on upload speed)
docker push milindkopi/amchi-asr-runpod:50epoch
docker push milindkopi/amchi-asr-runpod:latest
```

When complete, the image is publicly accessible at:
```
milindkopi/amchi-asr-runpod:50epoch
```

---

## 7. Create the RunPod Serverless endpoint

Do this in a browser at **https://www.runpod.io/console/serverless**.

1. Click **New Endpoint**
2. Fill in the form:

| Field | Value |
|-------|-------|
| **Name** | `amchi-konkani-asr` (or any name) |
| **Container image** | `milindkopi/amchi-asr-runpod:50epoch` |
| **Container registry credentials** | Leave blank (Docker Hub public image) |
| **GPU type** | RTX 4000 Ada Generation (preferred) or A40 |
| **Container disk** | `20` GB |
| **Min workers** | `0` (scale to zero when idle — no cost) |
| **Max workers** | `1` |
| **Idle timeout** | `5` seconds |

3. Under **Environment Variables**, add:

| Name | Value |
|------|-------|
| `CHECKPOINT_URL` | `<public R2 URL from §2>` |

4. Click **Deploy**
5. Wait for status to show **Ready**
6. **Copy the Endpoint ID** (e.g. `abc1def23ghi`) shown on the endpoint detail page

---

## 8. Test the endpoint

Set environment variables:
```bash
export RUNPOD_API_KEY="<RunPod API key>"
export RUNPOD_ENDPOINT_ID="<Endpoint ID from step 7>"
```

### Quick test — single audio URL

```bash
python3 scripts/test_runpod_endpoint.py \
  --audio_url "https://konkanicollector-production.up.railway.app/api/recordings/<RECORDING_ID>/audio" \
  --reference "<expected Devanagari text>"
```

Substitute a real recording ID from the Railway API. To find one:
```bash
python3 -c "
import requests, json
recs = requests.get('https://konkanicollector-production.up.railway.app/api/recordings').json()
story5 = [r for r in recs if r.get('status') == 'approved' and r.get('story_id') == 5]
print(json.dumps(story5[:3], ensure_ascii=False, indent=2))
"
```

### Manual curl test

```bash
# Get a Story 5 recording ID from the command above, then:
RECORDING_ID=<id>
AUDIO_URL="https://konkanicollector-production.up.railway.app/api/recordings/${RECORDING_ID}/audio"

# Fetch audio, base64 encode, send to endpoint
curl -s "$AUDIO_URL" | base64 -w 0 > /tmp/audio.b64

curl -X POST "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync?wait=120000" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"input\": {\"audio_base64\": \"$(cat /tmp/audio.b64)\"}}"
```

**Expected response:**
```json
{
  "id": "...",
  "status": "COMPLETED",
  "output": {
    "transcription": "<Devanagari text>"
  }
}
```

> **First request is slow (~90–120 seconds cold start):** The worker downloads the 1.4 GB checkpoint from R2 and loads the NeMo model. All subsequent requests from the same warm worker take ~1–3 seconds.

---

## 9. Troubleshooting

### Docker build fails on NeMo install
NeMo has a conflict with `blinker` on Ubuntu 22.04. If it fails, try adding `--ignore-installed blinker`:
```bash
docker build --no-cache -f runpod/Dockerfile.serverless -t amchi-asr-runpod .
```
If it still fails, check the Dockerfile.serverless for the NeMo install line and add `--ignore-installed blinker`.

### Docker build fails on conv_asr patch
Verify the patch exists:
```bash
ls -la patches/conv_asr_fixed.py
```
If missing: `git pull`.

### First request times out
The `/runsync` wait defaults to 120 seconds. Cold start (download 1.4 GB + NeMo load) takes 90–120 seconds. Send a warm-up request first, then test.

### `Model load failed` in response
`CHECKPOINT_URL` is unreachable or wrong. Verify:
```bash
curl -I "<your CHECKPOINT_URL>"
# Should return HTTP 200
```
If it returns 403, public access is not enabled on the R2 bucket (see §2).

### `No checkpoint found` error
Neither `CHECKPOINT_PATH` nor `CHECKPOINT_URL` is set in the endpoint environment variables. Go to RunPod Console → your endpoint → Edit → Environment Variables.

### Docker push is slow
The image is ~10–13 GB. On a home connection (10 Mbps upload) this takes 2–3 hours. Consider pushing from a cloud VM or faster connection.

---

## 10. Demo day warm-up procedure

```bash
# 5–10 minutes before demo:
# 1. Set Min Workers = 1 in RunPod Console (endpoint → Edit → Min Workers → 1)

# 2. Send a warm-up request (triggers cold start now, not during demo):
export RUNPOD_API_KEY="..."
export RUNPOD_ENDPOINT_ID="..."
curl -s "https://konkanicollector-production.up.railway.app/api/recordings/<id>/audio" | \
  base64 -w 0 > /tmp/warmup.b64
curl -X POST "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync?wait=180000" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"input\": {\"audio_base64\": \"$(cat /tmp/warmup.b64)\"}}"

# 3. Once warm-up returns successfully, the worker is ready.
#    Demo requests will complete in ~1–3 seconds.

# After demo:
# Set Min Workers back to 0 (RunPod Console → endpoint → Edit → Min Workers → 0)
```

Cost: RTX 4000 Ada at ~$0.44/hr × 3 hrs = **~$1.32 per demo session**.

---

## 11. Architecture reference

```
Web App / Test Script
        │
        │  POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync
        │  Body: { "input": { "audio_base64": "<base64 WAV>" } }
        │
        ▼
RunPod Serverless Worker
  Image:  milindkopi/amchi-asr-runpod:50epoch  (from Docker Hub)
  GPU:    RTX 4000 Ada
        │
        │  On first start: download 1.4 GB checkpoint from R2 to /tmp
        │
        ▼
  handler.py  →  scripts/amchi_inference.py
        │
        ├── NeMo EncDecHybridRNNTCTCBPEModel (CTC mode)
        │   Fine-tuned: Amchi Konkani Stories 1,2,3,7 (train), Story 5 test WER=54.7%
        │   Best checkpoint: epoch 47, CTC val_WER=53.2%
        │   Load pattern: config-patch + strict=False (see REPRODUCTION_NOTES.md §9)
        │   ASR latency: ~270ms (estimate; GPU-dependent)
        │
        └── Raw transcription returned (no server-side post-processing)
            Post-processing can be applied client-side via scripts/postprocess_asr.py
        │
        ▼
  Response: { "transcription": "<Devanagari text>" }
```

---

## 12. Input/output reference

**Request** (`POST .../runsync`, body `"input"` field):
```json
{ "audio_base64": "<base64-encoded 16 kHz mono WAV>" }
```
or
```json
{ "audio_url": "https://konkanicollector-production.up.railway.app/api/recordings/<id>/audio" }
```

**Successful response** (`.output` field):
```json
{ "transcription": "तें काय आसा?" }
```

**Error response:**
```json
{ "error": "description of what went wrong" }
```

---

## 13. Checkpoint details (for reference)

| Field | Value |
|-------|-------|
| **Model** | AI4Bharat IndicConformer (Marathi base, `stt_mr_hybrid_ctc_rnnt_large`) |
| **Fine-tuned on** | 511 Amchi Konkani samples (Stories 1,2,3,7) |
| **Best epoch** | 47 |
| **CTC val_WER** | 53.2% |
| **Test WER** | 54.7% (Story 5, 104 samples, 3 speakers) |
| **Per-speaker WER** | ashaheble: 51.7% / dipti.ajgaonkar21: 60.3% / lalimomadi: 51.9% |
| **Checkpoint size** | 1.4 GB |
| **R2 bucket** | `asr-checkpoints` |
| **R2 object key** | `results/checkpoints/konkani_asr-epoch=47-val_wer=0.532.ckpt` |

---

## 14. Next steps after this endpoint is working

1. **Improve WER:** Run optimization experiments from `AMCHI_KONKANI_FINETUNING_TODOS.md`:
   - Run A: frozen encoder (configs/amchi_konkani_frozen_encoder.yaml)
   - Run C: frozen encoder + cosine LR + 100 epochs (configs/amchi_konkani_run_c.yaml)
   - Upload new best checkpoint to R2, update `CHECKPOINT_URL` in RunPod endpoint

2. **Run statistical analysis:** `python3 scripts/analyze_results.py` on the test results — see `AMCHI_KONKANI_FINETUNING_TODOS.md §After Training: Analysis`.

3. **Add post-processing:** Decide on Option A/B/C from `AMCHI_KONKANI_TRAINING_GUIDE.md §8` and optionally add Gemini post-processing to the handler (model it on `runpod/handler_deaf.py`).

4. **Connect to web app:** Same pattern as `DEMO_WEBAPP_GUIDE.md §9 / RUNPOD_SERVERLESS_DEAF.md §12`. Replace `"transcription"` field with the `corrected` field once post-processing is added.
