"""
FastAPI app for Amchi ASR inference on a persistent RunPod (or any) machine.
Run on the same pod where you trained; no Docker required for the app.

  uvicorn runpod.app_fastapi:app --host 0.0.0.0 --port 8000

POST /transcribe:
  - Body: multipart/form-data with file "audio" (16 kHz mono WAV)
  - Or: JSON { "audio_base64": "<base64>" }

Response: { "transcription": "..." }
"""
import os
import base64
import tempfile
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in __import__("sys").path:
    __import__("sys").path.insert(0, _REPO_ROOT)

from scripts.amchi_inference import load_model_from_ckpt, transcribe_audio_bytes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("amchi_api")

CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    os.path.join(_REPO_ROOT, "results/2026-02-13_marathi_amchi_20epoch/checkpoints/marathi_amchi_20epoch-epoch=18-val_wer=0.550.ckpt"),
)
_model = None


def get_model():
    global _model
    if _model is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading model from %s", CHECKPOINT_PATH)
        _model = load_model_from_ckpt(CHECKPOINT_PATH, device=device)
        logger.info("Model loaded on %s", device)
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Optional: load model at startup to fail fast
    try:
        get_model()
    except Exception as e:
        logger.warning("Model not loaded at startup: %s", e)
    yield


app = FastAPI(title="Amchi ASR", lifespan=lifespan)


class TranscribeBody(BaseModel):
    audio_base64: str | None = None


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile | None = File(None),
    body: TranscribeBody | None = None,
):
    """Transcribe 16 kHz mono WAV. Send as multipart file 'audio' or JSON with audio_base64."""
    wav_bytes = None
    if audio:
        wav_bytes = await audio.read()
    elif body and body.audio_base64:
        try:
            wav_bytes = base64.b64decode(body.audio_base64)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)
    else:
        return JSONResponse({"error": "Provide 'audio' file or JSON audio_base64"}, status_code=400)

    if not wav_bytes or len(wav_bytes) > 20 * 1024 * 1024:
        return JSONResponse({"error": "Audio empty or too large (max 20 MB)"}, status_code=400)

    try:
        model = get_model()
        text = transcribe_audio_bytes(model, wav_bytes)
        return {"transcription": text or ""}
    except Exception as e:
        logger.exception("Inference failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/health")
async def health():
    return {"status": "ok"}
