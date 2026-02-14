# RunPod Inference Endpoint for Amchi ASR

This doc describes how to serve the **best checkpoint** from the 20-epoch run as an inference endpoint on RunPod, how to **reproduce the ~55% WER** on the test set, and how the **web app** will use it (with future migration to Hugging Face).

---

## 1. Best checkpoint and test-set WER

- **Best checkpoint:**  
  `results/2026-02-13_marathi_amchi_20epoch/checkpoints/marathi_amchi_20epoch-epoch=18-val_wer=0.550.ckpt`  
  (best by validation WER; epoch 18.)

- **Reported metrics (test set):**  
  Mean WER **~55%**, Mean CER **~16.3%** (see `results/2026-02-13_marathi_amchi_20epoch/experiments/20260213_205208/final_test_results.json`).

- **Demo script (reproduce WER on test dir):**  
  Run inference on the test manifest and print WER/CER:

  ```bash
  cd /workspace/amchi_asr
  python scripts/demo_test_set_wer.py \
    --checkpoint results/2026-02-13_marathi_amchi_20epoch/checkpoints/marathi_amchi_20epoch-epoch=18-val_wer=0.550.ckpt \
    --manifest data/amchi/test/manifest.jsonl \
    --output results_demo.json
  ```

  This uses the same model loading and transcription path as the RunPod endpoint (see `scripts/amchi_inference.py`).

---

## 2. Endpoint architecture options on RunPod

You can run inference in two ways on RunPod.

### Option A: Serverless endpoint (scale-to-zero, pay per request)

- **Use when:** Variable or low traffic; you want scale-to-zero and no always-on GPU.
- **How it works:** You deploy a **Docker image** that runs the RunPod serverless worker. The worker loads the model **once** when it starts; each request is a **job** with input (e.g. base64 WAV). RunPod runs the handler and returns the result.
- **Handler:** `runpod/handler.py`. It expects `job["input"]` with either:
  - **`audio_base64`:** base64-encoded 16 kHz mono WAV (best for web app).
  - **`audio_url`:** URL to a 16 kHz WAV file (worker downloads and transcribes).
- **Response:** `{ "transcription": "..." }` or `{ "error": "..." }`.
- **Payload limit:** RunPod request payload is 10–20 MB; keep audio short or compress (16 kHz WAV is fine for typical utterance lengths).

**Deploy and test:** See **[RUNPOD_SERVERLESS_DEPLOY.md](RUNPOD_SERVERLESS_DEPLOY.md)** for step-by-step build, push, endpoint creation, and the test script (`scripts/test_runpod_endpoint.py`).

**Deploy steps (high level):**

1. Build a Docker image that:
   - Installs Python, CUDA, PyTorch, NeMo, and project deps.
   - Copies the repo and the best checkpoint (or mounts it).
   - Sets `CHECKPOINT_PATH` to the path of the `.ckpt` inside the image.
   - Runs `python runpod/handler.py` (which calls `runpod.serverless.start({"handler": handler})`).
2. Push the image to a registry (e.g. Docker Hub).
3. In RunPod Console → Serverless → Create Endpoint → use your image, GPU type (e.g. A40/T4), and attach a volume if the checkpoint is on volume instead of in the image.

**Local test (without RunPod):**  
Use a `test_input.json` with `{"input": {"audio_base64": "<base64 of a 16kHz WAV>"}}` and run the handler script locally so it processes one job (see RunPod docs for local testing).

### Option B: Persistent pod (FastAPI on same machine as training)

- **Use when:** Your code and data already live on a RunPod pod; you want a simple HTTP API without building a serverless image.
- **How it works:** On the **same** RunPod instance where you trained, run a FastAPI app that loads the best checkpoint once and exposes `POST /transcribe`.
- **App:** `runpod/app_fastapi.py`.

  ```bash
  cd /workspace/amchi_asr
  CHECKPOINT_PATH=results/2026-02-13_marathi_amchi_20epoch/checkpoints/marathi_amchi_20epoch-epoch=18-val_wer=0.550.ckpt \
    uvicorn runpod.app_fastapi:app --host 0.0.0.0 --port 8000
  ```

- **Endpoints:**
  - **POST /transcribe**
    - **Multipart:** form field `audio` = 16 kHz mono WAV file.
    - **JSON:** `{ "audio_base64": "<base64>" }`.
  - **GET /health** for liveness.
- **Response:** `{ "transcription": "..." }` or `{ "error": "..." }` with appropriate status code.

Expose the pod (e.g. RunPod’s HTTP proxy or your own tunnel) so the web app can call `http(s)://<pod>/transcribe`.

---

## 3. End-to-end flow (web app → endpoint → user)

1. **User speaks** in the web app.
2. **Web app** records or uploads audio, converts to **16 kHz mono WAV** (e.g. browser MediaRecorder + resample, or server-side ffmpeg).
3. **Web app** sends the WAV to the inference endpoint:
   - **Serverless:** POST to RunPod endpoint with `{"input": {"audio_base64": "<base64>"}}` (see RunPod “Send requests” docs for exact API).
   - **FastAPI:** POST to `http(s)://<pod>:8000/transcribe` with multipart `audio` or JSON `audio_base64`.
4. **Endpoint** runs Amchi ASR (best checkpoint), returns **raw transcription** only: `{ "transcription": "..." }`.
5. **Web app** applies its **post-processing** (e.g. punctuation, spell-check, formatting).
6. **Web app** shows the **final transcription** to the user.

So: **endpoint = raw ASR only; post-processing = in the web app.** That keeps the endpoint simple and makes it easy to change post-processing or swap endpoints (e.g. move to Hugging Face later).

---

## 4. Moving to Hugging Face later

- **Same contract:** Keep the API shape your web app uses: e.g. “POST body with audio (file or base64) → JSON `{ "transcription": "..." }`.” Then you can implement the same contract on Hugging Face (Inference Endpoints or a Space with a custom API).
- **Code:** The core logic in `scripts/amchi_inference.py` is framework-agnostic (load .ckpt, transcribe). You can:
  - Export the best checkpoint to a format HF expects (e.g. `.nemo` or ONNX) if you move off RunPod, or
  - Run the same Python inference code inside an HF Space/Endpoint Docker image.
- **Secrets:** Store RunPod/HF API keys and endpoint URLs in the web app’s config or env; no need to change endpoint code if you only swap the base URL and auth.

---

## 5. File reference

| File | Purpose |
|------|--------|
| `scripts/amchi_inference.py` | Shared loader and `transcribe_audio` / `transcribe_audio_bytes` (used by CLI, RunPod, demo). |
| `scripts/smoke_test_inference.py` | CLI single-file inference (same logic as above). |
| `scripts/demo_test_set_wer.py` | Run inference on test manifest and report WER/CER (reproduce ~55% WER). |
| `runpod/handler.py` | RunPod serverless handler: load model once, handle `audio_base64` / `audio_url`, return transcription. |
| `runpod/app_fastapi.py` | FastAPI app for persistent pod: POST /transcribe (file or base64), GET /health. |
| `results/.../checkpoints/marathi_amchi_20epoch-epoch=18-val_wer=0.550.ckpt` | Best checkpoint (epoch 18, val_wer 0.55). |

---

## 6. Quick start (persistent pod)

On the RunPod where you already have the repo and checkpoint:

```bash
cd /workspace/amchi_asr
pip install fastapi uvicorn  # if not already
uvicorn runpod.app_fastapi:app --host 0.0.0.0 --port 8000
# From another terminal or your machine (if port is forwarded):
# curl -X POST -F "audio=@data/amchi/test/audio/570.wav" http://localhost:8000/transcribe
```

Then point your web app at `http://<runpod-ip>:8000/transcribe` (or the URL RunPod gives you) and send 16 kHz WAV; use the returned `transcription` for post-processing and display.
