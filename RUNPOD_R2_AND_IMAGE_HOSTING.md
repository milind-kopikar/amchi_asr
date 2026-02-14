# Where to Host the Docker Image and Checkpoints (R2, Registry)

## Short answers

| What | Where to host | Why |
|------|----------------|-----|
| **Docker image** | A **container registry** (Docker Hub, GitHub Container Registry, etc.), **not** R2 | RunPod Serverless workers run your code by doing `docker pull`. Only registries speak the Docker/OCI API. R2 is object storage (S3-compatible); you cannot `docker pull` from an R2 bucket URL. |
| **Checkpoints** | **R2** (or any S3-compatible storage) | Keeps the image small and stable. You can upload new `.ckpt` files without rebuilding the image. The worker **downloads** the checkpoint from R2 when it starts, then loads the model. |

---

## Recommended layout (image in registry, checkpoints in R2)

1. **Docker image (no checkpoint inside)**  
   - Build the image from this repo (without embedding the `.ckpt`).  
   - Push to **Docker Hub** (or ghcr.io, etc.).  
   - RunPod Serverless pulls this image when starting a worker.

2. **Checkpoints in R2**  
   - Upload your `.ckpt` files to an R2 bucket (e.g. `amchi-checkpoints/`).  
   - Create a **presigned URL** for the checkpoint you want the endpoint to use (e.g. valid for 24 hours or 7 days).  
   - In the RunPod endpoint’s **Environment Variables**, set `CHECKPOINT_URL` to that presigned URL (you can refresh it periodically or use a long-lived URL if your bucket allows).

3. **What the worker does**  
   - RunPod **pulls the image** from the registry (once per worker).  
   - Your handler code runs **inside** that image. On first use, `get_model()` sees `CHECKPOINT_URL`, **downloads** the file from that URL to a local path (e.g. `/tmp/amchi_checkpoint.ckpt`), then loads the model from that path.  
   - RunPod does **not** “pull the checkpoint from R2 and create a Docker image.” The image is fixed; the worker just downloads a file from R2 and uses it.

So: **one image in a registry, many checkpoints in R2; same image can load different checkpoints by changing `CHECKPOINT_URL`.**

---

## Flow summary

```
You:  Build image (no .ckpt) → push to Docker Hub
      Upload mymodel.ckpt to R2 → get presigned URL

RunPod:  Creates worker → docker pull youruser/amchi-asr-runpod:latest
         Runs your handler

Your code:  Sees CHECKPOINT_URL → downloads .ckpt from R2 to /tmp
            Loads model from /tmp/amchi_checkpoint.ckpt
            Handles inference requests
```

---

## Setting CHECKPOINT_URL (R2 presigned URL)

You can generate a presigned URL for your R2 object in several ways:

- **Cloudflare Dashboard**: R2 → bucket → object → “Create presigned URL”.  
- **Script (boto3, R2 S3 API):** Use `generate_presigned_url('get_object', Params={'Bucket': '...', 'Key': '...'}, ExpiresIn=86400)`.  
- **R2 public bucket:** If the bucket (or object) is public, you can use the public URL directly as `CHECKPOINT_URL` (no presigning).

Put that URL in the RunPod endpoint’s environment as `CHECKPOINT_URL`. The handler will download from it once per worker and then load the model.

---

## When to refresh the presigned URL

Presigned URLs expire (e.g. after 1 hour or 7 days). If your endpoint’s workers stay warm, they keep using the already-downloaded checkpoint. When a **new** worker starts after the URL has expired, the download will fail. So either:

- Use a long expiry (e.g. 7 days) and refresh the env var in RunPod before it expires, or  
- Make the checkpoint object **public** (if acceptable) and use the public URL so it never expires.

---

## Summary

- **Docker image** → Host on a **container registry** (Docker Hub, etc.). RunPod pulls from there. Do **not** put the image in R2.  
- **Checkpoints** → Store in **R2**. Use **Option “URL”**: image has no checkpoint; set `CHECKPOINT_URL` (R2 presigned or public URL); the worker downloads the `.ckpt` from R2 at startup and loads it.  
- This keeps the image small, lets you update checkpoints in R2 without rebuilding the image, and keeps a single image that can load any checkpoint you point it to.
