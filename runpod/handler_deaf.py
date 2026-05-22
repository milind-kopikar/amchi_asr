"""
RunPod Serverless handler for Deaf Speech ASR inference.

Differs from handler.py (Amchi Konkani model):
  - Uses load_model() from scripts/deaf_speech_inference.py, which applies
    the config-patch + strict=False pattern required for the hybrid CTC/RNNT
    checkpoint saved during deaf speech fine-tuning.
  - Runs Gemini post-processing (FILL/RECONSTRUCT modes) on every result.
  - Returns rich output: raw ASR, corrected text, mode, and per-step latency.

Environment variables:
  CHECKPOINT_PATH   — absolute path to .ckpt file (if checkpoint is in image
                      or on an attached RunPod Network Volume)
  CHECKPOINT_URL    — HTTP(S) URL to .ckpt (e.g. R2 public URL); downloaded
                      to /tmp on first worker start and cached for the lifetime
                      of that worker.
  GEMINI_API_KEY    — Gemini API key for post-processing. If unset, the handler
                      returns raw ASR only (mode="SKIPPED").

Input (job["input"]):
  audio_base64: str   — base64-encoded 16 kHz mono WAV (preferred for web app)
  OR
  audio_url: str      — URL to a 16 kHz mono WAV (worker downloads and transcribes)

Output:
  {
    "raw":          "ू किती ⁇",
    "corrected":    "हे किती आहे?",
    "mode":         "FILL",
    "latency_ms":   { "asr": 270, "postprocess": 1450, "total": 1720 }
  }
  OR
  { "error": "..." }
"""

import base64
import logging
import os
import sys
import tempfile
import time
import urllib.request

# ---------------------------------------------------------------------------
# Path setup — add repo root so we can import scripts/
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Silence NeMo / PyTorch Lightning verbose logs before any nemo import
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for _noisy in ("nemo", "nemo_logger", "pytorch_lightning", "lightning", "lightning.pytorch"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

from scripts.deaf_speech_inference import load_model, transcribe_wav  # noqa: E402
from scripts.postprocess_asr import postprocess_sample                # noqa: E402

logger = logging.getLogger("runpod_deaf")

# ---------------------------------------------------------------------------
# Globals — model and checkpoint path cached for the lifetime of the worker
# ---------------------------------------------------------------------------
_MODEL = None
_RESOLVED_CKPT_PATH = None


def _resolve_checkpoint() -> str:
    """Return a local path to the .ckpt file, downloading from URL if needed."""
    global _RESOLVED_CKPT_PATH
    if _RESOLVED_CKPT_PATH is not None:
        return _RESOLVED_CKPT_PATH

    path_env = os.environ.get("CHECKPOINT_PATH", "")
    url_env = os.environ.get("CHECKPOINT_URL", "")

    if path_env and os.path.isfile(path_env):
        logger.info("Using checkpoint at CHECKPOINT_PATH: %s", path_env)
        _RESOLVED_CKPT_PATH = path_env
        return _RESOLVED_CKPT_PATH

    if url_env:
        local = os.path.join(tempfile.gettempdir(), "deaf_speech_checkpoint.ckpt")
        if os.path.isfile(local):
            logger.info("Using cached checkpoint at %s", local)
        else:
            logger.info("Downloading checkpoint from CHECKPOINT_URL …")
            from scripts.runpod_http import download_url_to_path
            download_url_to_path(url_env, local)
            size_mb = os.path.getsize(local) / (1024 * 1024)
            logger.info("Downloaded %.0f MB → %s", size_mb, local)
        _RESOLVED_CKPT_PATH = local
        return _RESOLVED_CKPT_PATH

    raise FileNotFoundError(
        "No checkpoint found. Set CHECKPOINT_PATH (path to local .ckpt) "
        "or CHECKPOINT_URL (URL to .ckpt, e.g. Cloudflare R2 public URL)."
    )


def _get_model():
    """Load and cache the deaf speech model (once per worker process)."""
    global _MODEL
    if _MODEL is None:
        import torch
        ckpt_path = _resolve_checkpoint()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading deaf speech model on %s …", device)
        _MODEL = load_model(ckpt_path, device=device)
        logger.info("Model ready on %s", device)
    return _MODEL


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    """Process one RunPod inference job.

    Expected input: { "audio_base64": "<base64 WAV>" }
              OR    { "audio_url": "<URL to WAV>" }
    """
    job_id = job.get("id", "?")
    inp = job.get("input") or {}

    # ------------------------------------------------------------------
    # 1. Decode audio bytes
    # ------------------------------------------------------------------
    wav_bytes: bytes | None = None

    if inp.get("audio_base64"):
        try:
            wav_bytes = base64.b64decode(inp["audio_base64"])
        except Exception as exc:
            return {"error": f"Invalid audio_base64: {exc}"}

    elif inp.get("audio_url"):
        try:
            from scripts.runpod_http import fetch_url_bytes
            wav_bytes = fetch_url_bytes(inp["audio_url"], timeout=30)
        except Exception as exc:
            return {"error": f"Failed to fetch audio_url: {exc}"}

    else:
        return {"error": "Provide 'audio_base64' (base64 WAV) or 'audio_url' in input"}

    if not wav_bytes:
        return {"error": "Empty audio received"}
    if len(wav_bytes) > 20 * 1024 * 1024:
        return {"error": "Audio too large (max 20 MB). Send shorter clips."}

    # ------------------------------------------------------------------
    # 2. Write to temp WAV file (NeMo transcribe() requires a file path)
    # ------------------------------------------------------------------
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            fh.write(wav_bytes)
            tmp_path = fh.name

        # ---------------------------------------------------------------
        # 3. ASR
        # ---------------------------------------------------------------
        try:
            model = _get_model()
        except Exception as exc:
            logger.exception("Model load failed for job %s", job_id)
            return {"error": f"Model load failed: {exc}"}

        raw_asr, asr_sec = transcribe_wav(model, tmp_path)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # 4. Gemini post-processing (optional — requires GEMINI_API_KEY)
    # ------------------------------------------------------------------
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    pp_sec = 0.0
    corrected = raw_asr
    mode = "SKIPPED"

    if gemini_key:
        try:
            import google.genai as genai
            client = genai.Client(api_key=gemini_key)
            t0 = time.perf_counter()
            # original_wer=1.0 → all modes eligible (no safety-valve skip in inference)
            result = postprocess_sample(client, raw_asr, original_wer=1.0)
            pp_sec = time.perf_counter() - t0
            corrected = result["corrected"]
            mode = result["mode"]
        except Exception as exc:
            logger.warning("Post-processing failed for job %s: %s", job_id, exc)
            corrected = raw_asr
            mode = "PP_ERROR"
    else:
        logger.info("GEMINI_API_KEY not set — returning raw ASR only (mode=SKIPPED)")

    total_sec = asr_sec + pp_sec

    return {
        "raw": raw_asr,
        "corrected": corrected,
        "mode": mode,
        "latency_ms": {
            "asr": int(asr_sec * 1000),
            "postprocess": int(pp_sec * 1000),
            "total": int(total_sec * 1000),
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
