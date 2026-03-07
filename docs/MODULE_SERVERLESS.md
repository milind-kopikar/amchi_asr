# Module: Serverless Deployment — RunPod Endpoint

**Self-contained guide.** Read this to build and deploy a RunPod serverless endpoint
for either the Amchi Konkani or Deaf Speech ASR model.
For checkpoint URLs, see `docs/CHECKPOINTS_REGISTRY.md`.

---

## What this module does

Packages the inference pipeline into a Docker image that RunPod's serverless platform
can run on-demand. The worker downloads the checkpoint from R2 on startup, then
accepts audio (as a base64 string or URL) and returns a transcription.

---

## Key files

| File | Purpose |
|---|---|
| `runpod/handler_deaf.py` | Serverless handler for deaf speech model |
| `runpod/handler.py` | Serverless handler for Amchi Konkani model |
| `runpod/Dockerfile.deaf` | Docker image for deaf speech endpoint |
| `runpod/Dockerfile` | Docker image for Amchi Konkani endpoint |
| `scripts/test_deaf_endpoint.py` | Test script for the deployed endpoint |

---

## Architecture

```
[Client] --audio--> [RunPod Serverless Endpoint]
                         |
                    [Docker container]
                         |
                    handler_deaf.py
                    1. Download checkpoint from R2 (on cold start)
                    2. Load model (patched config, CTC strategy)
                    3. Transcribe audio
                    4. Run Gemini post-processing
                    5. Return {raw, corrected, latency}
```

---

## Step-by-step: Build and deploy (run on your LOCAL machine, not RunPod)

### Prerequisites
- Docker installed locally
- DockerHub account
- RunPod account
- R2 credentials (from `docs/CHECKPOINTS_REGISTRY.md`)
- Gemini API key (from `GEMINI_API_KEY` in `.env`)

### 1. Pull latest code

```bash
git clone https://github.com/milind-kopikar/amchi_asr.git
cd amchi_asr
```

### 2. Build the Docker image (deaf speech)

```bash
docker build -f runpod/Dockerfile.deaf -t deaf-speech-asr-runpod .
```

### 3. Push to DockerHub

```bash
docker tag deaf-speech-asr-runpod YOUR_DOCKERHUB_USERNAME/deaf-speech-asr-runpod:latest
docker push YOUR_DOCKERHUB_USERNAME/deaf-speech-asr-runpod:latest
```

### 4. Create RunPod Serverless Endpoint

1. Go to [RunPod Console](https://www.runpod.io/console/serverless) → **New Endpoint**
2. **Container Image:** `YOUR_DOCKERHUB_USERNAME/deaf-speech-asr-runpod:latest`
3. **GPU:** RTX 4000 Ada (20GB) or A40 (48GB) — model needs ~8GB VRAM
4. **Environment variables:**

| Variable | Value |
|---|---|
| `CHECKPOINT_URL` | `https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev/results/deaf_speech_dsd/checkpoints/konkani_asr-epoch=96-val_wer=0.269.ckpt` |
| `GEMINI_API_KEY` | Your Gemini API key |

5. Save and note the **Endpoint ID**.

### 5. Test the endpoint

```bash
python3 scripts/test_deaf_endpoint.py \
  --endpoint_id YOUR_ENDPOINT_ID \
  --api_key YOUR_RUNPOD_API_KEY \
  --audio data/deaf_speech/audio/131.wav
```

---

## Checkpoint URLs for RunPod environment variables

| Model | `CHECKPOINT_URL` |
|---|---|
| **Deaf speech DS-D (BEST)** | `https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev/results/deaf_speech_dsd/checkpoints/konkani_asr-epoch=96-val_wer=0.269.ckpt` |
| **Amchi Konkani Run S (BEST)** | `https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev/results/run_c_stratified_split/checkpoints/konkani_asr-epoch=88-val_wer=0.334.ckpt` |
| Amchi Konkani Run C | `https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev/results/run_c_story_split/checkpoints/konkani_asr-epoch=66-val_wer=0.504.ckpt` |
| Deaf speech baseline | `https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev/nemo_experiments/deaf_speech_story4_50epoch/checkpoints/konkani_asr-epoch=21-val_wer=0.720.ckpt` |

> The bucket has public access enabled — no authentication needed to download.

---

## Handler input/output format

**Input** (JSON):
```json
{
  "input": {
    "audio": "<base64-encoded WAV>",
    "audio_url": "https://...",   // alternative to audio field
    "gemini_key": "AIzaSy..."     // optional override
  }
}
```

**Output** (JSON):
```json
{
  "raw": "किती आहे ⁇",
  "corrected": "हे किती आहे?",
  "mode": "FILL",
  "asr_latency_s": 0.3,
  "postprocess_latency_s": 1.5
}
```

---

## Cold start behaviour

The first request to a new worker takes ~60–90 seconds because:
1. Docker container starts (~10s)
2. Checkpoint downloads from R2 (~30–60s for 1.3GB file)
3. NeMo model initialises and loads weights (~15s)

Subsequent requests on the same worker are fast (~2s end-to-end).

---

## Related docs (for deep context)
- `RUNPOD_SERVERLESS_DEAF.md` — earlier detailed guide for deaf speech endpoint
- `RUNPOD_SERVERLESS_AMCHI_KONKANI.md` — Amchi Konkani endpoint guide
- `HANDOFF_SERVERLESS_RESUME.md` — last known state of serverless build
- `R2_SETUP_CHECKPOINTS.md` — how to create R2 bucket and API tokens
