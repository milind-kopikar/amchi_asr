# RunPod inference for Amchi ASR

- **Serverless:** `handler.py` — use with RunPod Serverless. Full deploy and test: **[RUNPOD_SERVERLESS_DEPLOY.md](../RUNPOD_SERVERLESS_DEPLOY.md)**. See also [RUNPOD_INFERENCE_ENDPOINT.md](../RUNPOD_INFERENCE_ENDPOINT.md). Set `CHECKPOINT_PATH` in the environment to your `.ckpt` path.
- **Persistent pod:** `app_fastapi.py` — run on the same pod where you trained:  
  `uvicorn runpod.app_fastapi:app --host 0.0.0.0 --port 8000`  
  Then `POST /transcribe` with 16 kHz mono WAV (multipart file or `audio_base64` in JSON).

Both use the shared loader and transcription in `scripts/amchi_inference.py`.
