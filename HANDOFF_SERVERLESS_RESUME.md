# Handoff: Resume RunPod Serverless Endpoint (Next Session)

**For the full project map and doc index, read [AGENT_START_HERE.md](AGENT_START_HERE.md) first.** This file is the “continue tomorrow” checklist only.

Use this when you start again after stopping your RunPod. Everything you need to finish the serverless inference endpoint is below.

---

## What’s already done

- **R2 bucket:** Checkpoint is in Cloudflare R2 bucket `asr-checkpoints`.
- **R2 object key:**  
  `results/2026-02-13_marathi_amchi_20epoch/checkpoints/marathi_amchi_20epoch-epoch=18-val_wer=0.550.ckpt`
- **Public checkpoint URL (no expiry):**  
  `https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev/results/2026-02-13_marathi_amchi_20epoch/checkpoints/marathi_amchi_20epoch-epoch=18-val_wer=0.550.ckpt`
- **Code:** Handler supports `CHECKPOINT_URL` (downloads from R2 when a worker starts). Repo is ready to build the serverless image.

---

## What you still need to do (when you resume)

### 1. Start a RunPod pod (or use any machine with Docker)

- You need a place with Docker and the repo (e.g. a new RunPod pod, same project).
- Clone or pull the repo: `git clone https://github.com/milind-kopikar/amchi_asr.git` (or pull if already there).

### 2. Build the Docker image (no checkpoint inside)

From the repo root:

```bash
cd /workspace/amchi_asr   # or wherever the repo is
docker build -f runpod/Dockerfile.serverless -t amchi-asr-runpod .
```

- Do **not** copy the checkpoint into the image; the worker will get it from R2 via `CHECKPOINT_URL`.
- Build can take 10–20 minutes (PyTorch + NeMo).

### 3. Push the image to Docker Hub

```bash
docker login
docker tag amchi-asr-runpod YOUR_DOCKERHUB_USERNAME/amchi-asr-runpod:latest
docker push YOUR_DOCKERHUB_USERNAME/amchi-asr-runpod:latest
```

Replace `YOUR_DOCKERHUB_USERNAME` with your Docker Hub username.

### 4. Create the Serverless endpoint in RunPod

1. Go to **RunPod Console** → **Serverless** → **New Endpoint**.
2. **Image:** `YOUR_DOCKERHUB_USERNAME/amchi-asr-runpod:latest`
3. **GPU:** e.g. T4 or A40.
4. **Container disk:** e.g. 20 GB.
5. **Environment variables** → Add:
   - **Name:** `CHECKPOINT_URL`
   - **Value:**  
     `https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev/results/2026-02-13_marathi_amchi_20epoch/checkpoints/marathi_amchi_20epoch-epoch=18-val_wer=0.550.ckpt`
6. **Workers:** e.g. Min 0, Max 1 or 2.
7. Create the endpoint and copy the **Endpoint ID**.

### 5. Test the endpoint

From any machine with the repo (e.g. your RunPod pod or laptop):

```bash
cd /workspace/amchi_asr
export RUNPOD_API_KEY="your_runpod_api_key"      # from RunPod → Settings → API Keys
export RUNPOD_ENDPOINT_ID="the_endpoint_id"      # from step 4
python scripts/test_runpod_endpoint.py --audio data/amchi/test/audio/570.wav
```

Or with a manifest (full test set):

```bash
python scripts/test_runpod_endpoint.py --manifest data/amchi/test/manifest.jsonl
```

---

## Quick reference

| Item | Value |
|------|--------|
| **CHECKPOINT_URL** (for RunPod env) | `https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev/results/2026-02-13_marathi_amchi_20epoch/checkpoints/marathi_amchi_20epoch-epoch=18-val_wer=0.550.ckpt` |
| **R2 bucket** | `asr-checkpoints` |
| **R2 public bucket URL** | `https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev` |
| **Best checkpoint (this run)** | Epoch 18, val_wer 0.55, ~55% test WER |

---

## Where things live in the repo

| What | Path |
|------|------|
| Serverless handler (loads from CHECKPOINT_URL or CHECKPOINT_PATH) | `runpod/handler.py` |
| Dockerfile (no checkpoint) | `runpod/Dockerfile.serverless` |
| Upload checkpoint to R2 | `scripts/upload_checkpoint_to_r2.py` |
| Test endpoint (single file or manifest) | `scripts/test_runpod_endpoint.py` |
| Full deploy steps | `RUNPOD_SERVERLESS_DEPLOY.md` |
| R2 setup and public URL | `R2_SETUP_CHECKPOINTS.md` |
| Image vs R2, why registry | `RUNPOD_R2_AND_IMAGE_HOSTING.md` |

---

## Adding more checkpoints later

1. Upload to R2 (from a pod that has the new `.ckpt`):
   ```bash
   export R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_BUCKET_NAME=asr-checkpoints
   python scripts/upload_checkpoint_to_r2.py --file results/DATE_runname/checkpoints/best.ckpt --public-url
   ```
2. Public URL will be:  
   `https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev/<object-key>`  
   (object key is printed by the script).
3. In RunPod → your endpoint → Edit → set **CHECKPOINT_URL** to that new URL. No need to rebuild the Docker image.

---

## If you need to re-create R2 credentials

- Cloudflare Dashboard → **R2** → **Manage R2 API Tokens** → Create API token (Object Read & Write).
- Use those env vars only when running `upload_checkpoint_to_r2.py` on a machine that has the checkpoint file. Do not paste secrets in chat or commit them to the repo.

You can stop the RunPod pod; the serverless endpoint (once created) is independent and will keep working. When you resume, use any RunPod pod or machine with Docker to build/push the image and create the endpoint, then test with the commands above.
