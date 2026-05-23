#!/usr/bin/env python3
"""
Pre-warm one or both RunPod ASR endpoints with a tiny silence sample.

RunPod serverless workers cold-start in ~30–60s (model load + checkpoint
download). For a live demo this is unacceptable, so we fire a no-op
inference request ~30 seconds before walking on stage. The worker spins up,
loads the model, and stays warm for `idle_timeout` (5 min by default in the
endpoint settings) — long enough for the demo.

Usage
-----
Pre-warm both endpoints (typical demo-day):

    export RUNPOD_API_KEY="<your key>"
    export RUNPOD_AMCHI_ENDPOINT_ID="<amchi endpoint id>"
    export RUNPOD_DEAF_ENDPOINT_ID="<deaf endpoint id>"
    python3 scripts/prewarm_runpod.py

Pre-warm just one (faster, if you're only demoing one model):

    python3 scripts/prewarm_runpod.py --only amchi
    python3 scripts/prewarm_runpod.py --only deaf

Override endpoint IDs or API key on the command line (overrides env vars):

    python3 scripts/prewarm_runpod.py --amchi-endpoint xxx --api-key yyy

Exit codes
----------
0   All requested endpoints responded successfully.
1   Missing required credentials (RUNPOD_API_KEY or any requested endpoint ID).
2   At least one endpoint returned an error or timed out.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUNPOD_BASE_URL = "https://api.runpod.ai/v2"

# A pre-warm request must complete within this time, or we treat it as failed.
# RunPod's /runsync endpoint can hold the connection while waiting for a worker.
DEFAULT_TIMEOUT_SECONDS = 120

# Length of the silence clip we send. Short enough to be cheap, long enough
# to be a valid WAV NeMo can process (>= 0.1s).
SILENCE_SECONDS = 1.0
SAMPLE_RATE_HZ = 16_000


# ---------------------------------------------------------------------------
# Pure helpers — small, testable, no I/O
# ---------------------------------------------------------------------------

def generate_silence_wav(duration_seconds: float = SILENCE_SECONDS,
                         sample_rate: int = SAMPLE_RATE_HZ) -> bytes:
    """Build a valid 16-bit mono PCM WAV containing silence.

    Parameters
    ----------
    duration_seconds
        Length of the silence in seconds. Must be > 0.
    sample_rate
        Sample rate in Hz. Must be > 0. Defaults to 16 kHz (NeMo expects this).

    Returns
    -------
    bytes
        Complete WAV file bytes (header + data).

    Raises
    ------
    ValueError
        If ``duration_seconds`` or ``sample_rate`` is not positive.
    """
    if duration_seconds <= 0:
        raise ValueError(f"duration_seconds must be > 0, got {duration_seconds}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sample_rate}")

    num_samples = int(duration_seconds * sample_rate)
    bytes_per_sample = 2  # 16-bit mono
    data_size = num_samples * bytes_per_sample
    chunk_size = 36 + data_size  # RIFF chunk minus the 8-byte "RIFF" header

    # WAV header (44 bytes for PCM)
    header = b"".join([
        b"RIFF",
        struct.pack("<I", chunk_size),
        b"WAVE",
        b"fmt ",
        struct.pack("<I", 16),                   # PCM fmt chunk size
        struct.pack("<H", 1),                    # audio format = PCM
        struct.pack("<H", 1),                    # channels = mono
        struct.pack("<I", sample_rate),          # sample rate
        struct.pack("<I", sample_rate * bytes_per_sample),  # byte rate
        struct.pack("<H", bytes_per_sample),     # block align
        struct.pack("<H", 16),                   # bits per sample
        b"data",
        struct.pack("<I", data_size),
    ])
    silence_data = b"\x00" * data_size
    return header + silence_data


def encode_audio_for_runpod(wav_bytes: bytes) -> str:
    """Base64-encode WAV bytes so they can be sent in the JSON body.

    Parameters
    ----------
    wav_bytes
        Raw WAV file contents.

    Returns
    -------
    str
        ASCII base64 string (no line breaks).

    Raises
    ------
    TypeError
        If ``wav_bytes`` is not bytes-like.
    """
    if not isinstance(wav_bytes, (bytes, bytearray, memoryview)):
        raise TypeError(f"wav_bytes must be bytes-like, got {type(wav_bytes).__name__}")
    return base64.b64encode(bytes(wav_bytes)).decode("ascii")


# ---------------------------------------------------------------------------
# I/O — call out to RunPod (mockable via the ``opener`` parameter)
# ---------------------------------------------------------------------------

def prewarm_endpoint(
    api_key: str,
    endpoint_id: str,
    audio_b64: str,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    opener=None,
) -> dict:
    """Send a single /runsync request to the given RunPod endpoint.

    Parameters
    ----------
    api_key
        RunPod API key. Must be non-empty.
    endpoint_id
        RunPod endpoint ID. Must be non-empty.
    audio_b64
        Base64-encoded WAV (typically from ``encode_audio_for_runpod``).
    timeout_seconds
        Per-request timeout. The RunPod /runsync endpoint can hold the
        connection for a while if a worker is cold.
    opener
        Test seam — function that takes a ``urllib.request.Request`` and
        returns a context-manager response object. Defaults to
        ``urllib.request.urlopen``.

    Returns
    -------
    dict
        ``{"ok": True, "elapsed_seconds": float, "response": <runpod-json>}``
        on success, or ``{"ok": False, "error": "<message>"}`` on failure.
        Never raises.
    """
    if not api_key:
        return {"ok": False, "error": "Missing api_key"}
    if not endpoint_id:
        return {"ok": False, "error": "Missing endpoint_id"}

    url = f"{RUNPOD_BASE_URL}/{endpoint_id}/runsync"
    body = json.dumps({"input": {"audio_base64": audio_b64}}).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")

    if opener is None:
        opener = urllib.request.urlopen

    start = time.perf_counter()
    try:
        with opener(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return {"ok": False, "error": f"HTTP {exc.code}: {exc.reason}"}
    except urllib.error.URLError as exc:
        return {"ok": False, "error": f"Network error: {exc.reason}"}
    except Exception as exc:  # timeouts, ssl issues, etc.
        return {"ok": False, "error": f"Request failed: {exc}"}
    elapsed = time.perf_counter() - start

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"ok": False, "error": f"Non-JSON response: {raw[:200]}"}

    # RunPod returns {"status": "COMPLETED", "output": {...}} on success.
    status = (payload.get("status") or "").upper()
    if status and status != "COMPLETED":
        return {"ok": False, "error": f"RunPod status {status}: {payload}"}

    return {"ok": True, "elapsed_seconds": elapsed, "response": payload}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def prewarm_selected(
    api_key: str,
    amchi_endpoint_id: Optional[str],
    deaf_endpoint_id: Optional[str],
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    opener=None,
) -> dict:
    """Pre-warm every endpoint whose ID is provided.

    Returns
    -------
    dict
        Map ``{"amchi": <prewarm_endpoint result>, "deaf": ...}``. Endpoints
        not requested are omitted from the map.
    """
    silence = generate_silence_wav()
    audio_b64 = encode_audio_for_runpod(silence)

    results: dict = {}
    if amchi_endpoint_id:
        results["amchi"] = prewarm_endpoint(
            api_key, amchi_endpoint_id, audio_b64,
            timeout_seconds=timeout_seconds, opener=opener,
        )
    if deaf_endpoint_id:
        results["deaf"] = prewarm_endpoint(
            api_key, deaf_endpoint_id, audio_b64,
            timeout_seconds=timeout_seconds, opener=opener,
        )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-warm one or both RunPod ASR endpoints with a silence sample.",
    )
    parser.add_argument("--only", choices=("amchi", "deaf"), default=None,
                       help="Pre-warm just one endpoint instead of both.")
    parser.add_argument("--api-key", default=None,
                       help="RunPod API key. Defaults to $RUNPOD_API_KEY.")
    parser.add_argument("--amchi-endpoint", default=None,
                       help="Amchi endpoint ID. Defaults to $RUNPOD_AMCHI_ENDPOINT_ID.")
    parser.add_argument("--deaf-endpoint", default=None,
                       help="Deaf endpoint ID. Defaults to $RUNPOD_DEAF_ENDPOINT_ID.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS,
                       help=f"Per-request timeout in seconds (default {DEFAULT_TIMEOUT_SECONDS}).")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None, *, opener=None) -> int:
    args = _parse_args(argv)

    api_key = args.api_key or os.environ.get("RUNPOD_API_KEY", "")
    amchi_id = args.amchi_endpoint or os.environ.get("RUNPOD_AMCHI_ENDPOINT_ID", "")
    deaf_id = args.deaf_endpoint or os.environ.get("RUNPOD_DEAF_ENDPOINT_ID", "")

    # Apply --only filter
    if args.only == "amchi":
        deaf_id = ""
    elif args.only == "deaf":
        amchi_id = ""

    # Credential checks
    if not api_key:
        print("ERROR: Missing RUNPOD_API_KEY (env var or --api-key).", file=sys.stderr)
        return 1
    if not amchi_id and not deaf_id:
        print("ERROR: At least one endpoint ID is required "
              "(RUNPOD_AMCHI_ENDPOINT_ID / RUNPOD_DEAF_ENDPOINT_ID or --amchi-endpoint / --deaf-endpoint).",
              file=sys.stderr)
        return 1

    results = prewarm_selected(
        api_key,
        amchi_id or None,
        deaf_id or None,
        timeout_seconds=args.timeout,
        opener=opener,
    )

    any_failed = False
    for label, result in results.items():
        if result.get("ok"):
            secs = result["elapsed_seconds"]
            print(f"  [OK]  {label} endpoint warmed in {secs:.2f}s")
        else:
            any_failed = True
            print(f"  [FAIL] {label} endpoint: {result.get('error', 'unknown error')}",
                  file=sys.stderr)
    return 2 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
