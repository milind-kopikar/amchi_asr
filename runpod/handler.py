"""
RunPod Serverless handler for Amchi Konkani ASR inference, with Gemini
post-processing.

Mirrors the architecture of ``runpod/handler_deaf.py``:
  - Loads the fine-tuned checkpoint once per worker
  - Loads the Amchi Konkani dictionary once per worker (baked into the image)
  - Per job: ASR → optional Gemini post-processing → return {raw, corrected, mode, latency_ms}

Environment variables
---------------------
CHECKPOINT_PATH    Absolute path to a .ckpt on disk. Used if set and the file exists.
CHECKPOINT_URL     URL to a .ckpt (e.g. R2 public URL). Downloaded to /tmp on
                   first use.
AMCHI_DICT_PATH    Path to the dictionary JSON (default: /app/data/amchi_konkani_dict.json
                   when running inside the Docker image, otherwise
                   <repo_root>/data/amchi_konkani_dict.json).
GEMINI_API_KEY     Optional. If unset, post-processing is skipped (mode="SKIPPED")
                   and ``corrected`` equals ``raw``.

Input  (``job["input"]``)
-------------------------
  ``audio_base64``: base64-encoded 16 kHz mono WAV (preferred for webapp)
  ``audio_url``:    HTTP(S) URL to a 16 kHz mono WAV (worker downloads it)

Output
------
  {
    "raw":           "<raw ASR>",
    "corrected":     "<Gemini-corrected or raw>",
    "transcription": "<alias for corrected — kept for backward compatibility>",
    "mode":          "PASSTHROUGH" | "FILL" | "RECONSTRUCT" | "SKIPPED" | "PP_ERROR" | "SKIP",
    "latency_ms":    { "asr": int, "postprocess": int, "total": int }
  }

Or on failure:

  { "error": "<message>" }
"""

from __future__ import annotations

import base64
import logging
import os
import sys
import tempfile
import time
import urllib.request
from typing import Optional

# Shared HTTP helpers — every external fetch (audio_url, checkpoint download)
# must set a User-Agent or Cloudflare returns 403 on R2 .r2.dev URLs.
# See scripts/runpod_http.py for the rationale and the test seam contract.

# ---------------------------------------------------------------------------
# Path setup — add repo root so we can import scripts/
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Silence NeMo / Lightning verbose logs before any nemo import
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for _noisy in ("nemo", "nemo_logger", "pytorch_lightning", "lightning", "lightning.pytorch"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

logger = logging.getLogger("runpod_amchi")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_AUDIO_BYTES = 20 * 1024 * 1024  # 20 MB cap to protect the worker
URL_FETCH_TIMEOUT_SECONDS = 30


# ---------------------------------------------------------------------------
# Lazy-loaded singletons (one per worker process)
# ---------------------------------------------------------------------------

_MODEL = None
_DICT_WORDS: Optional[set] = None
_RESOLVED_CHECKPOINT_PATH: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers — kept as small pure-ish functions so they can be unit-tested.
# ---------------------------------------------------------------------------

def _decode_audio_input(inp: dict) -> tuple[Optional[bytes], Optional[dict]]:
    """Extract WAV bytes from a job input.

    Parameters
    ----------
    inp
        ``job["input"]`` dict. Expected keys: ``audio_base64`` (str)
        OR ``audio_url`` (str). ``audio_base64`` takes precedence.

    Returns
    -------
    (wav_bytes, error_dict)
        On success: ``(bytes, None)``. On any error or invalid input:
        ``(None, {"error": "<message>"})``. Never raises.
    """
    if not isinstance(inp, dict):
        return None, {"error": "Job input must be a dict"}

    if inp.get("audio_base64"):
        try:
            wav_bytes = base64.b64decode(inp["audio_base64"], validate=False)
        except Exception as exc:
            return None, {"error": f"Invalid audio_base64: {exc}"}
    elif inp.get("audio_url"):
        try:
            from scripts.runpod_http import fetch_url_bytes
            wav_bytes = fetch_url_bytes(
                inp["audio_url"], timeout=URL_FETCH_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            return None, {"error": f"Failed to fetch audio_url: {exc}"}
    else:
        return None, {"error": "Provide audio_base64 or audio_url in input"}

    if not wav_bytes:
        return None, {"error": "Empty audio received"}
    if len(wav_bytes) > MAX_AUDIO_BYTES:
        return None, {
            "error": f"Audio too large (max {MAX_AUDIO_BYTES // (1024 * 1024)} MB)"
        }
    return wav_bytes, None


def _resolve_checkpoint_path() -> Optional[str]:
    """Return the local path to the checkpoint, downloading from URL if necessary.

    Returns
    -------
    str | None
        Path to the .ckpt file, or None if nothing is configured.
    """
    global _RESOLVED_CHECKPOINT_PATH
    if _RESOLVED_CHECKPOINT_PATH is not None:
        return _RESOLVED_CHECKPOINT_PATH

    path_env = os.environ.get("CHECKPOINT_PATH", "")
    url_env = os.environ.get("CHECKPOINT_URL", "")

    if path_env and os.path.isfile(path_env):
        logger.info("Using CHECKPOINT_PATH: %s", path_env)
        _RESOLVED_CHECKPOINT_PATH = path_env
        return _RESOLVED_CHECKPOINT_PATH

    if url_env:
        local = os.path.join(tempfile.gettempdir(), "amchi_checkpoint.ckpt")
        if not os.path.isfile(local):
            logger.info("Downloading checkpoint from CHECKPOINT_URL …")
            from scripts.runpod_http import download_url_to_path
            download_url_to_path(url_env, local)
            size_mb = os.path.getsize(local) / (1024 * 1024)
            logger.info("Downloaded %.0f MB → %s", size_mb, local)
        else:
            logger.info("Using cached checkpoint at %s", local)
        _RESOLVED_CHECKPOINT_PATH = local
        return _RESOLVED_CHECKPOINT_PATH

    # Last-resort default for local development only — not relied on in production.
    default = os.path.join(
        _REPO_ROOT,
        "results/2026-02-13_marathi_amchi_20epoch/checkpoints/"
        "marathi_amchi_20epoch-epoch=18-val_wer=0.550.ckpt",
    )
    _RESOLVED_CHECKPOINT_PATH = default if os.path.isfile(default) else None
    return _RESOLVED_CHECKPOINT_PATH


def _default_dict_path() -> str:
    """Resolve the default dictionary path.

    The Docker image copies the dictionary to ``/app/data/amchi_konkani_dict.json``;
    locally during development we fall back to ``<repo_root>/data/amchi_konkani_dict.json``.
    """
    in_image = "/app/data/amchi_konkani_dict.json"
    if os.path.isfile(in_image):
        return in_image
    return os.path.join(_REPO_ROOT, "data", "amchi_konkani_dict.json")


def _load_dictionary_once(dict_path: Optional[str] = None) -> set:
    """Load the Amchi Konkani dictionary as a set of Devanagari words.

    Parameters
    ----------
    dict_path
        Explicit path. If None, reads ``AMCHI_DICT_PATH`` env var, else uses
        the default location.

    Returns
    -------
    set[str]
        Devanagari words from the dictionary. Cached after the first call.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    global _DICT_WORDS
    if _DICT_WORDS is not None:
        return _DICT_WORDS

    path = dict_path or os.environ.get("AMCHI_DICT_PATH") or _default_dict_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Konkani dictionary not found at {path}. "
            "Run `scripts/build_amchi_dict.py` or set AMCHI_DICT_PATH."
        )

    # Lazy import to keep the test surface small.
    from scripts.amchi_postprocess_asr import load_dictionary

    _DICT_WORDS = load_dictionary(path)
    logger.info("Loaded %d Konkani dictionary words from %s", len(_DICT_WORDS), path)
    return _DICT_WORDS


def _apply_amchi_postprocess(
    raw_asr: str,
    dict_words: set,
    gemini_key: str,
    *,
    postprocess_sample_fn=None,
    genai_client_factory=None,
) -> tuple[str, str, float]:
    """Run Gemini post-processing on a raw ASR transcript.

    Parameters
    ----------
    raw_asr
        Raw ASR output (may contain ``⁇`` markers).
    dict_words
        Set of valid Konkani Devanagari words (from ``_load_dictionary_once``).
    gemini_key
        Gemini API key. If empty / falsy, post-processing is skipped.
    postprocess_sample_fn, genai_client_factory
        Test seams — let unit tests inject mocks instead of calling Gemini.

    Returns
    -------
    (corrected, mode, postprocess_sec)
        ``mode`` is one of ``"PASSTHROUGH"``, ``"FILL"``, ``"RECONSTRUCT"``,
        ``"SKIPPED"`` (no API key), ``"SKIP"`` (empty input), or ``"PP_ERROR"``.
    """
    if not raw_asr:
        return "", "SKIP", 0.0
    if not gemini_key:
        return raw_asr, "SKIPPED", 0.0

    # Lazy imports — the Gemini SDK is heavy and not needed for non-PP tests.
    if postprocess_sample_fn is None:
        from scripts.amchi_postprocess_asr import postprocess_sample as postprocess_sample_fn
    if genai_client_factory is None:
        import google.genai as genai
        genai_client_factory = lambda: genai.Client(api_key=gemini_key)

    try:
        client = genai_client_factory()
        t0 = time.perf_counter()
        # original_wer=1.0 → "always run a mode" (no safety-valve skip at inference)
        # reference is unused by the function (kept for batch evaluation) so we pass "".
        result = postprocess_sample_fn(
            client,
            prediction=raw_asr,
            reference="",
            dict_words=dict_words,
            original_wer=1.0,
        )
        elapsed = time.perf_counter() - t0
        return result.get("corrected", raw_asr), result.get("mode", "PP_ERROR"), elapsed
    except Exception as exc:
        logger.warning("Post-processing failed: %s", exc)
        return raw_asr, "PP_ERROR", 0.0


def _get_model():
    """Load and cache the Amchi ASR model (once per worker process)."""
    global _MODEL
    if _MODEL is None:
        from scripts.amchi_inference import load_model_from_ckpt
        path = _resolve_checkpoint_path()
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                "No checkpoint found. Set CHECKPOINT_PATH (local file) "
                "or CHECKPOINT_URL (e.g. R2 public URL)."
            )
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Amchi ASR model from %s on %s …", path, device)
        _MODEL = load_model_from_ckpt(path, device=device)
        logger.info("Model ready on %s", device)
    return _MODEL


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(
    job: dict,
    *,
    model_factory=None,
    transcribe_fn=None,
    postprocess_sample_fn=None,
    genai_client_factory=None,
    dict_loader=None,
) -> dict:
    """Process one RunPod inference job.

    The extra keyword arguments are test seams — production callers should
    only pass ``job``. Tests pass mocks for the heavy dependencies.

    Returns
    -------
    dict
        ``{raw, corrected, transcription, mode, latency_ms}`` on success;
        ``{error: str}`` on failure.
    """
    job_id = job.get("id", "?") if isinstance(job, dict) else "?"
    inp = (job.get("input") if isinstance(job, dict) else None) or {}

    # 1. Decode audio
    wav_bytes, err = _decode_audio_input(inp)
    if err is not None:
        return err

    # 2. Load model + dictionary (cached after first call)
    try:
        model = (model_factory or _get_model)()
        dict_words = (dict_loader or _load_dictionary_once)()
    except FileNotFoundError as exc:
        logger.exception("Resource load failed for job %s", job_id)
        return {"error": str(exc)}
    except Exception as exc:
        logger.exception("Model load failed for job %s", job_id)
        return {"error": f"Model load failed: {exc}"}

    # 3. ASR
    try:
        if transcribe_fn is None:
            from scripts.amchi_inference import transcribe_audio_bytes
            transcribe_fn = transcribe_audio_bytes
        t0 = time.perf_counter()
        raw_asr = transcribe_fn(model, wav_bytes) or ""
        asr_sec = time.perf_counter() - t0
    except Exception as exc:
        logger.exception("Inference failed for job %s", job_id)
        return {"error": f"ASR failed: {exc}"}

    # 4. Optional Gemini post-processing
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    corrected, mode, pp_sec = _apply_amchi_postprocess(
        raw_asr, dict_words, gemini_key,
        postprocess_sample_fn=postprocess_sample_fn,
        genai_client_factory=genai_client_factory,
    )

    total_sec = asr_sec + pp_sec
    return {
        "raw": raw_asr,
        "corrected": corrected,
        # Alias kept so scripts/test_runpod_endpoint.py keeps working unchanged.
        "transcription": corrected,
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
    import runpod  # type: ignore[import-not-found]
    runpod.serverless.start({"handler": handler})
