# Phase 2: RunPod Serverless Endpoint — Deaf Speech ASR
## Build Guide for Local Claude Agent

**Read this first. This is a complete, self-contained guide to build, deploy, and test the deaf speech serverless inference endpoint from a local machine.**

---

## 0. Current status

| Step | Status | Notes |
|------|--------|-------|
| Handler code (`runpod/handler_deaf.py`) | ✅ Done | In this repo |
| Dockerfile (`runpod/Dockerfile.deaf`) | ✅ Done | In this repo |
| Test script (`scripts/test_deaf_endpoint.py`) | ✅ Done | In this repo |
| Checkpoint → Cloudflare R2 | ✅ Done | Public URL, no expiry (see §2) |
| Docker image built | ❌ TODO | Build on local machine (§3) |
| Docker image pushed to Docker Hub | ❌ TODO | Push to `milindkopi` account (§4) |
| RunPod Serverless endpoint created | ❌ TODO | In RunPod Console (§5) |
| Endpoint tested | ❌ TODO | Using test script (§6) |

---

## 1. Credentials the agent needs to ask for

Before starting, ask the user for these four values. They should NOT be committed to git.

| Credential | Where to find it | Used in step |
|------------|-----------------|--------------|
| **Docker Hub password** (or PAT) | hub.docker.com → Account Settings → Security → New Access Token | Step 4 (`docker login`) |
| **RunPod API key** | RunPod Console → Settings → API Keys | Step 6 (test script) |
| **Gemini API key** (`AIzaSy…`) | Google AI Studio or `.env` file | Step 5 (endpoint env var) |

Docker Hub username is already known: **`milindkopi`**
The checkpoint URL and R2 upload are already done — no R2 credentials needed.

---

## 2. What's already on Cloudflare R2 (no action needed)

The 1.4 GB model checkpoint was uploaded on 2026-03-01. Use this URL verbatim.

```
CHECKPOINT_URL=https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev/nemo_experiments/deaf_speech_story4_50epoch/checkpoints/konkani_asr-epoch=21-val_wer=0.720.ckpt
```

This URL has **no expiry**. No presigning needed. The RunPod worker downloads it once on first start and caches it for the lifetime of that worker.

---

## 3. Prerequisites on the local machine

```bash
# Verify Docker is installed and running
docker --version          # need 20.x or later
docker info               # must succeed (Docker daemon running)

# Verify you are in the repo root
ls runpod/Dockerfile.deaf   # must exist
ls runpod/handler_deaf.py   # must exist
ls patches/conv_asr_fixed.py  # must exist (used during Docker build)
```

If `git pull` hasn't been run yet:
```bash
git clone https://github.com/milind-kopikar/amchi_asr.git
cd amchi_asr
```
or if already cloned:
```bash
git pull
```

---

## 4. Build the Docker image

Run from the **repo root** (`amchi_asr/`):

```bash
docker build -f runpod/Dockerfile.deaf -t deaf-speech-asr-runpod .
```

This will:
1. Pull `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` as base
2. Install Python 3.11, ffmpeg, build tools
3. Install PyTorch (CUDA 11.8) — downloads ~2 GB
4. Install NeMo ASR toolkit — downloads ~3 GB
5. Install `google-genai`, `jiwer`, `runpod` SDK
6. Copy the repo code into `/app`
7. Apply the `conv_asr_fixed.py` patch to NeMo

**Expected duration:** 15–25 minutes on first build (large downloads). Subsequent builds use Docker layer cache and are much faster.

**Expected final image size:** ~8–10 GB

When it completes you should see:
```
Successfully built <image_id>
Successfully tagged deaf-speech-asr-runpod:latest
```

If the build fails, see §8 (Troubleshooting).

---

## 5. Push to Docker Hub

```bash
# Log in (enter the Docker Hub password / PAT when prompted)
docker login --username milindkopi

# Tag with the full Docker Hub path
docker tag deaf-speech-asr-runpod milindkopi/deaf-speech-asr-runpod:latest

# Push (this uploads ~8–10 GB — takes 5–15 min depending on upload speed)
docker push milindkopi/deaf-speech-asr-runpod:latest
```

When complete, the image is publicly accessible at:
```
milindkopi/deaf-speech-asr-runpod:latest
```

---

## 6. Create the RunPod Serverless endpoint

Do this in a browser at **https://www.runpod.io/console/serverless**.

1. Click **New Endpoint**
2. Fill in the form:

| Field | Value |
|-------|-------|
| **Name** | `deaf-speech-asr` (or any name) |
| **Container image** | `milindkopi/deaf-speech-asr-runpod:latest` |
| **Container registry credentials** | Leave blank (Docker Hub public image) |
| **GPU type** | RTX 4000 Ada Generation (preferred) or A40 |
| **Container disk** | `20` GB |
| **Min workers** | `0` (scale to zero when idle — no cost) |
| **Max workers** | `1` |
| **Idle timeout** | `5` seconds (worker goes cold after 5s idle) |

3. Under **Environment Variables**, add these two:

| Name | Value |
|------|-------|
| `CHECKPOINT_URL` | `https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev/nemo_experiments/deaf_speech_story4_50epoch/checkpoints/konkani_asr-epoch=21-val_wer=0.720.ckpt` |
| `GEMINI_API_KEY` | `<the Gemini API key from step 1>` |

4. Click **Deploy** (or **Create Endpoint**)
5. Wait for status to show **Ready**
6. **Copy the Endpoint ID** — it looks like `abc1def23ghi` and is shown on the endpoint detail page

---

## 7. Test the endpoint

Install the test dependency if not already present:
```bash
pip install jiwer requests
```

Set environment variables:
```bash
export RUNPOD_API_KEY="<RunPod API key from step 1>"
export RUNPOD_ENDPOINT_ID="<Endpoint ID from step 6>"
```

### Quick test — single sample via Railway audio URL (no local files needed)

```bash
python3 scripts/test_deaf_endpoint.py \
  --audio_url "https://deafspeechcollector-production.up.railway.app/api/recordings/131/audio" \
  --reference "दूध किती आहे?"
```

**Expected output:**
```
────────────────────────────────────────────────────────────────
  Raw ASR   : ू किती ⁇
  Corrected : हे किती आहे?  [FILL]
  Reference : दूध किती आहे?
  WER raw   : 66.7%
  WER corr  : 33.3%  ↓ improved
  Latency   : ASR 270ms | Post-process 1450ms | Total 1720ms
────────────────────────────────────────────────────────────────
```

> **First request is slow** (~90 seconds cold start): the worker downloads the 1.4 GB checkpoint from R2 and loads the NeMo model. All subsequent requests from the same warm worker take ~2 seconds.

### Test a few more samples

```bash
# Sample 135 — "बस कधी येईल?"
python3 scripts/test_deaf_endpoint.py \
  --audio_url "https://deafspeechcollector-production.up.railway.app/api/recordings/135/audio" \
  --reference "बस कधी येईल?"

# Sample 132 — "एक लिटर दूध द्या."
python3 scripts/test_deaf_endpoint.py \
  --audio_url "https://deafspeechcollector-production.up.railway.app/api/recordings/132/audio" \
  --reference "एक लिटर दूध द्या."
```

### Run all 124 test samples

```bash
python3 scripts/test_deaf_endpoint.py --all_samples
```

---

## 8. Troubleshooting

### Docker build fails on NeMo install
NeMo has a conflict with `blinker` on Ubuntu 22.04. The Dockerfile already has `--ignore-installed blinker` to handle this. If it still fails, try:
```bash
docker build --no-cache -f runpod/Dockerfile.deaf -t deaf-speech-asr-runpod .
```

### Docker build fails on conv_asr patch step
The patch step requires `patches/conv_asr_fixed.py` to exist in the repo. Verify:
```bash
ls -la patches/conv_asr_fixed.py
```
If missing, pull the latest repo: `git pull`.

### First request times out
The default `/runsync` wait is 120 seconds. The cold start (download 1.4 GB + load NeMo) can take 90–120 seconds. If it times out:
- Send a warm-up request and wait for it to complete before testing
- Or increase `?wait=` parameter in the test script

### `PP_ERROR` in response (post-processing failed)
The `GEMINI_API_KEY` environment variable may be wrong or missing. Verify it's set correctly in the RunPod endpoint's environment variables. The raw ASR output is still returned in this case.

### `Model load failed` in response
The `CHECKPOINT_URL` is unreachable or wrong. Verify the URL manually:
```bash
curl -I "https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev/nemo_experiments/deaf_speech_story4_50epoch/checkpoints/konkani_asr-epoch=21-val_wer=0.720.ckpt"
# Should return HTTP 200
```

### Docker push is very slow
The image is ~8–10 GB. On a home internet connection (10 Mbps upload) this takes ~2 hours. Consider pushing from a cloud VM or using a faster connection.

---

## 9. Demo day warm-up procedure

On the day of a demo, do this 5–10 minutes before:

```bash
# 1. Set min_workers=1 in RunPod Console (endpoint → Edit → Min Workers → 1)
#    This keeps one worker warm and avoids mid-demo cold starts.

# 2. Send a warm-up request to trigger the cold start now, not during demo:
export RUNPOD_API_KEY="..."
export RUNPOD_ENDPOINT_ID="..."
python3 scripts/test_deaf_endpoint.py \
  --audio_url "https://deafspeechcollector-production.up.railway.app/api/recordings/131/audio"

# 3. Once that returns successfully, the worker is warm and ready.
#    Demo requests will complete in ~2 seconds.
```

After the demo:
```bash
# Set min_workers back to 0 to avoid ongoing GPU costs
# (RunPod Console → endpoint → Edit → Min Workers → 0)
```

Cost: RTX 4000 Ada at ~$0.44/hr × 3 hrs = **~$1.32 per demo session**.

---

## 10. Architecture reference

```
Web App / Test Script
        │
        │  POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync
        │  Body: { "input": { "audio_base64": "<base64 WAV>" } }
        │
        ▼
RunPod Serverless Worker
  Image:  milindkopi/deaf-speech-asr-runpod:latest  (from Docker Hub)
  GPU:    RTX 4000 Ada
        │
        │  On first start: download checkpoint from R2 to /tmp
        │
        ▼
  handler_deaf.py
        │
        ├── NeMo EncDecHybridRNNTCTCBPEModel (CTC mode)
        │   Fine-tuned: deaf speech Story 4, epoch 21, val_WER=72%
        │   Load pattern: config-patch + strict=False (see REPRODUCTION_NOTES.md §9)
        │   ASR latency: ~270ms
        │
        └── Gemini 2.5 Flash post-processing
            FILL mode: anchor words present → fill garbled slots
            RECONSTRUCT mode: no anchors → full reconstruction
            Safety valve: revert if Gemini worsens output
            PP latency: ~1.5s
        │
        ▼
  Response: { "raw", "corrected", "mode", "latency_ms" }
```

---

## 11. Input/output reference

**Request** (`POST .../runsync`, body `"input"` field):
```json
{ "audio_base64": "<base64-encoded 16 kHz mono WAV>" }
```
or
```json
{ "audio_url": "https://deafspeechcollector-production.up.railway.app/api/recordings/131/audio" }
```

**Successful response** (`.output` field of RunPod response):
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
{ "error": "description of what went wrong" }
```

**Mode values:**

| Mode | Meaning |
|------|---------|
| `FILL` | Gemini filled garbled slots using trusted anchor words |
| `RECONSTRUCT` | Gemini reconstructed full sentence from phonetic fragments |
| `FILL_REVERTED` | Fill tried but safety valve reverted (corrected was worse than raw) |
| `SKIP` | Perfect ASR (WER=0), no post-processing needed |
| `SKIPPED` | `GEMINI_API_KEY` not set; raw ASR returned unchanged |
| `PP_ERROR` | Post-processing API call failed; raw ASR returned as fallback |

---

## 12. Connecting to the Phase 1 web app (future)

Once the endpoint is working, the web app can call it for live inference. In the Next.js app:

1. Add to Railway environment variables:
   - `NEXT_PUBLIC_RUNPOD_ENDPOINT_ID` = your endpoint ID
   - `NEXT_PUBLIC_RUNPOD_API_KEY` = your RunPod API key

2. Replace the simulated `setTimeout` in the Transcribe button handler with:

```typescript
const handleTranscribeLive = async () => {
  setIsTranscribing(true)

  // Fetch the audio and encode to base64
  const audioResp = await fetch(selectedSample.audio_url)
  const audioBuffer = await audioResp.arrayBuffer()
  const base64 = btoa(String.fromCharCode(...new Uint8Array(audioBuffer)))

  // Call RunPod endpoint
  const endpointId = process.env.NEXT_PUBLIC_RUNPOD_ENDPOINT_ID
  const apiKey = process.env.NEXT_PUBLIC_RUNPOD_API_KEY
  const resp = await fetch(
    `https://api.runpod.ai/v2/${endpointId}/runsync?wait=120000`,
    {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ input: { audio_base64: base64 } }),
    }
  )
  const data = await resp.json()
  const output = data.output  // { raw, corrected, mode, latency_ms }

  setResult({ ...selectedSample, raw_asr: output.raw, corrected: output.corrected, mode: output.mode })
  setIsTranscribing(false)
}
```

The response shape (`raw`, `corrected`, `mode`) matches the pre-computed `samples.json` fields exactly, so no UI changes are needed.

See `DEMO_WEBAPP_GUIDE.md` §9 for full context.
