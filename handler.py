"""
RunPod Serverless handler for Amchi ASR inference.
Loads the best checkpoint once at worker init; each job receives audio (base64 WAV) and returns transcription.

Input (job["input"]):
  - audio_base64: base64-encoded 16 kHz mono WAV (preferred for web app)
  - OR audio_url: URL to a 16 kHz WAV file (worker will download and transcribe)

Output:
  - { "transcription": "..." }
  - Or { "error": "..." } on failure

Checkpoint (one of these):
  - CHECKPOINT_PATH: path to .ckpt on disk (e.g. in image or on attached volume).
  - CHECKPOINT_URL: URL to .ckpt (e.g. R2 presigned URL). Worker downloads to /tmp once, then loads.
"""
import os
import base64
import logging
import tempfile
import urllib.request

# Add repo root so we can import scripts
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in __import__("sys").path:
    __import__("sys").path.insert(0, _REPO_ROOT)

from scripts.amchi_inference import load_model_from_ckpt, transcribe_audio_bytes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("runpod_amchi")

# Resolved path to use for loading (local file or downloaded from URL)
_RESOLVED_CHECKPOINT_PATH = None
MODEL = None

def _resolve_checkpoint_path():
    """Return path to checkpoint: local CHECKPOINT_PATH or download from CHECKPOINT_URL to /tmp."""
    global _RESOLVED_CHECKPOINT_PATH
    if _RESOLVED_CHECKPOINT_PATH is not None:
        return _RESOLVED_CHECKPOINT_PATH
    path_env = os.environ.get("CHECKPOINT_PATH")
    url_env = os.environ.get("CHECKPOINT_URL")
    if path_env and os.path.isfile(path_env):
        _RESOLVED_CHECKPOINT_PATH = path_env
        logger.info("Using checkpoint at %s", _RESOLVED_CHECKPOINT_PATH)
        return _RESOLVED_CHECKPOINT_PATH
    if url_env:
        local = os.path.join(tempfile.gettempdir(), "amchi_checkpoint.ckpt")
        logger.info("Downloading checkpoint from CHECKPOINT_URL to %s ...", local)
        urllib.request.urlretrieve(url_env, local)
        _RESOLVED_CHECKPOINT_PATH = local
        logger.info("Downloaded checkpoint to %s", local)
        return _RESOLVED_CHECKPOINT_PATH
    default = os.path.join(
        _REPO_ROOT,
        "results/2026-02-13_marathi_amchi_20epoch/checkpoints/marathi_amchi_20epoch-epoch=18-val_wer=0.550.ckpt",
    )
    _RESOLVED_CHECKPOINT_PATH = default if os.path.isfile(default) else None
    return _RESOLVED_CHECKPOINT_PATH

def get_model():
    global MODEL
    if MODEL is None:
        path = _resolve_checkpoint_path()
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                "No checkpoint found. Set CHECKPOINT_PATH to a local file or CHECKPOINT_URL to a download URL (e.g. R2 presigned)."
            )
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        logger.info("Loading Amchi ASR model from %s ...", path)
        MODEL = load_model_from_ckpt(path, device=device)
        logger.info("Model loaded on %s", device)
    return MODEL


def handler(job):
    """Process one inference job. input: { audio_base64?: str, audio_url?: str }."""
    job_id = job.get("id", "?")
    inp = job.get("input") or {}

    # Get WAV bytes
    wav_bytes = None
    if inp.get("audio_base64"):
        try:
            wav_bytes = base64.b64decode(inp["audio_base64"])
        except Exception as e:
            return {"error": f"Invalid audio_base64: {e}"}
    elif inp.get("audio_url"):
        try:
            with urllib.request.urlopen(inp["audio_url"], timeout=30) as r:
                wav_bytes = r.read()
        except Exception as e:
            return {"error": f"Failed to fetch audio_url: {e}"}
    else:
        return {"error": "Provide audio_base64 or audio_url in input"}

    if not wav_bytes or len(wav_bytes) > 20 * 1024 * 1024:  # 20 MB limit
        return {"error": "Audio too large or empty (max 20 MB)"}

    try:
        model = get_model()
        transcription = transcribe_audio_bytes(model, wav_bytes)
        return {"transcription": transcription or ""}
    except Exception as e:
        logger.exception("Inference failed for job %s", job_id)
        return {"error": str(e)}


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
