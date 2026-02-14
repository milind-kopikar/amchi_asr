# RunPod Serverless: Deploy and Test Amchi ASR

Step-by-step to deploy the Amchi ASR inference endpoint as a **RunPod Serverless** endpoint and test it with sample audio.

---

## Why push the image to a registry?

**Serverless workers are not your pod.** When you create a Serverless endpoint, RunPod runs your code on *separate* worker machines. Those workers have no access to your pod’s disk (e.g. `/workspace`). The only way RunPod can get your code and environment onto a worker is to run a **container image**. That image must be pulled from somewhere on the internet—a **container registry** (Docker Hub, GitHub Container Registry, etc.). So you:

1. **Build** the image (on your RunPod pod or your laptop).
2. **Push** it to a registry (e.g. Docker Hub) so it’s reachable by URL.
3. When creating the endpoint, you tell RunPod: “Use image `youruser/amchi-asr-runpod:latest`.” RunPod then runs `docker pull youruser/amchi-asr-runpod:latest` when it starts a worker.

Without pushing to a registry, RunPod has no way to load your image onto the serverless workers.

---

## Option A vs Option B (checkpoint on `/workspace`)

You already have the checkpoint on your RunPod pod at  
`results/2026-02-13_marathi_amchi_20epoch/checkpoints/marathi_amchi_20epoch-epoch=18-val_wer=0.550.ckpt`.  
Your pod’s `/workspace` is **not** visible to Serverless workers, so the checkpoint must either be **inside the image** (Option A) or on a **RunPod Network Volume** attached to the endpoint (Option B).

| | Option A: Embed in image | Option B: Network Volume |
|--|---------------------------|---------------------------|
| **Where checkpoint lives** | Copied into the Docker image at build time. | On a RunPod **Network Volume** (separate from your pod’s disk). |
| **Build on RunPod?** | Yes. On your pod: `cp ... checkpoint.ckpt`, uncomment `COPY` in Dockerfile, then `docker build`. | Yes. Build the image *without* the checkpoint (smaller, faster push). |
| **Push to registry?** | Yes. Image is large (~2–3 GB) because it contains the .ckpt. | Yes. Image is smaller (no .ckpt). |
| **Extra setup** | None. | Create a Network Volume in RunPod, copy the .ckpt from your pod to that volume, attach the volume when creating the endpoint, set `CHECKPOINT_PATH`. |
| **Easier for one endpoint?** | **Yes.** One build, one push, create endpoint; no volume to manage. | Better if you want to swap checkpoints without rebuilding the image. |

**Recommendation when the checkpoint is already on your pod:** Use **Option A**. On the same RunPod pod: copy the checkpoint to `checkpoint.ckpt`, uncomment the `COPY` line, build the image, push to Docker Hub, then create the Serverless endpoint pointing at that image. No Network Volume needed.

---

## What you need from your side

1. **RunPod account** and **API key** (Settings → API Keys in RunPod console).
2. **Docker** (on your RunPod pod or laptop) to build the image.
3. **The best checkpoint** on the machine where you build the image, OR a RunPod Network Volume that holds the checkpoint (so the worker can load it at runtime).

---

## 1. Get the checkpoint into the image or onto a volume

**Option A: Embed checkpoint in the image (simplest for a single endpoint)**

On your RunPod pod (or any machine that has the checkpoint):

```bash
cd /workspace/amchi_asr
cp results/2026-02-13_marathi_amchi_20epoch/checkpoints/marathi_amchi_20epoch-epoch=18-val_wer=0.550.ckpt checkpoint.ckpt
```

Then uncomment the `COPY` line in `runpod/Dockerfile.serverless`:

```dockerfile
COPY checkpoint.ckpt /app/checkpoint.ckpt
```

**Option B: Use a RunPod Network Volume (no need to put checkpoint in the image)**

1. In RunPod Console → **Storage** → create a **Network Volume**.
2. Upload or copy the `.ckpt` file to that volume (e.g. from a pod that has it).
3. When creating the Serverless endpoint, **attach this volume** to the worker.
4. In the endpoint’s **Environment Variables**, set `CHECKPOINT_PATH` to the path of the `.ckpt` on the volume (e.g. `/runpod-volume/checkpoint.ckpt` or `/workspace/checkpoint.ckpt` depending on how RunPod mounts it).

**Option C: Checkpoint from a URL (e.g. Cloudflare R2)**

Keep the image small and store checkpoints in R2 (or any HTTP-accessible URL). The worker downloads the `.ckpt` from the URL when it first loads the model. Set **Environment Variable** `CHECKPOINT_URL` to a presigned URL (or public URL) of the `.ckpt` file in R2. See **[RUNPOD_R2_AND_IMAGE_HOSTING.md](RUNPOD_R2_AND_IMAGE_HOSTING.md)** for where to host the Docker image vs checkpoints and how to use R2.

---

## 2. Build the Docker image

From the **repo root** (on the RunPod pod or any machine with Docker and the repo):

```bash
cd /workspace/amchi_asr
docker build -f runpod/Dockerfile.serverless -t amchi-asr-runpod .
```

If you used Option A, ensure `checkpoint.ckpt` is in the repo root and the `COPY` line is uncommented. The build may take 10–20 minutes (PyTorch + NeMo).

---

## 3. Push the image to a registry

RunPod Serverless needs to **pull** your image from a registry when it starts a worker. You can build the image on your RunPod pod (where the checkpoint already is) and push from there.

**On your RunPod pod (after building):**

1. Log in to Docker Hub (or your registry):  
   `docker login`  
   Use your Docker Hub username and password (or a personal access token).

2. Tag and push:

```bash
docker tag amchi-asr-runpod YOUR_DOCKERHUB_USER/amchi-asr-runpod:latest
docker push YOUR_DOCKERHUB_USER/amchi-asr-runpod:latest
```

Replace `YOUR_DOCKERHUB_USER` with your Docker Hub username. The first push may take a while if you used Option A (image includes the checkpoint). After that, RunPod can use the image URL when you create the endpoint.

---

## 4. Create the Serverless endpoint on RunPod

1. Go to **RunPod Console** → **Serverless** → **New Endpoint**.
2. **Template**: “Import from Docker Registry” (or “Custom” and paste your image URL).
3. **Image**: `YOUR_DOCKERHUB_USER/amchi-asr-runpod:latest` (or your image URL).
4. **GPU**: e.g. T4 or A40 (NeMo inference needs a GPU).
5. **Container disk**: e.g. 20 GB (enough for the image and NeMo cache).
6. **Volume** (optional): If you use Option B, attach your Network Volume and set the mount path.
7. **Environment variables**:  
   - Option A: leave unset (default `/app/checkpoint.ckpt` if embedded) or set `CHECKPOINT_PATH=/app/checkpoint.ckpt`.
   - Option B: `CHECKPOINT_PATH=/runpod-volume/marathi_amchi_20epoch-epoch=18-val_wer=0.550.ckpt` (adjust path to match your volume).
   - Option C (R2): `CHECKPOINT_URL=https://...` (presigned or public URL to the `.ckpt` in R2).
8. **Workers**: Min 0, Max 1 or 2 to start (scale up later if needed).
9. Create the endpoint and note the **Endpoint ID** (e.g. `abc123xyz`).

---

## 5. Test the endpoint with the test script

Use the provided test script to send a test WAV and print the transcription (and optional WER vs reference).

**One-off test (single file):**

```bash
cd /workspace/amchi_asr
export RUNPOD_API_KEY="your_runpod_api_key"
export RUNPOD_ENDPOINT_ID="your_endpoint_id"

python scripts/test_runpod_endpoint.py \
  --audio data/amchi/test/audio/570.wav
```

You can also pass API key and endpoint via flags:

```bash
python scripts/test_runpod_endpoint.py \
  --endpoint-id YOUR_ENDPOINT_ID \
  --api-key YOUR_RUNPOD_API_KEY \
  --audio data/amchi/test/audio/570.wav
```

**Test with reference (to see WER):**

```bash
python scripts/test_runpod_endpoint.py \
  --endpoint-id YOUR_ENDPOINT_ID \
  --api-key YOUR_RUNPOD_API_KEY \
  --audio data/amchi/test/audio/570.wav \
  --reference "रोहन होड ज़ाल्लो!"
```

**Run on several test samples (from manifest):**

```bash
python scripts/test_runpod_endpoint.py \
  --endpoint-id YOUR_ENDPOINT_ID \
  --api-key YOUR_RUNPOD_API_KEY \
  --manifest data/amchi/test/manifest.jsonl
```

The script will:

- Read each test audio path from the manifest (or use `--audio` for a single file).
- Base64-encode the WAV and send it to the RunPod endpoint (`/runsync`).
- Print the returned **transcription** and, if a reference is provided (or from the manifest), the **WER** for that sample.

The first request may take longer (cold start: worker loads the model). Later requests should be faster.

---

## 6. Quick curl check (optional)

If you prefer not to use the script:

```bash
# Encode one test file to base64 (Linux/macOS)
B64=$(base64 -w0 data/amchi/test/audio/570.wav)

curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync?wait=120000" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"input\": {\"audio_base64\": \"$B64\"}}"
```

Response should look like: `{"output": {"transcription": "..."}, "status": "COMPLETED", ...}`.

---

## Summary

| Step | You do |
|------|--------|
| 1 | Put checkpoint in image (Option A) or on a Network Volume (Option B). |
| 2 | Build image: `docker build -f runpod/Dockerfile.serverless -t amchi-asr-runpod .` |
| 3 | Push image to Docker Hub (or your registry). |
| 4 | In RunPod Console, create Serverless endpoint from that image; set GPU, env, volume if needed. |
| 5 | Run `scripts/test_runpod_endpoint.py` with `--audio` or `--manifest` and your endpoint ID + API key. |

After this, the serverless endpoint is independent of your training pod: you can stop or terminate the pod and keep using the endpoint.
