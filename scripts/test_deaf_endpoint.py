#!/usr/bin/env python3
"""
Test the RunPod Serverless Deaf Speech endpoint.

Sends one audio file (local WAV or Railway audio URL) to the endpoint,
prints the response, and optionally computes WER against a reference.

Usage:
  # From a local WAV file:
  export RUNPOD_API_KEY="..."
  export RUNPOD_ENDPOINT_ID="..."
  python3 scripts/test_deaf_endpoint.py --audio data/deaf_speech/audio/131.wav

  # From a Railway audio URL (no local file needed):
  python3 scripts/test_deaf_endpoint.py \
      --audio_url "https://deafspeechcollector-production.up.railway.app/api/recordings/131/audio"

  # With a reference to compute WER:
  python3 scripts/test_deaf_endpoint.py \
      --audio data/deaf_speech/audio/131.wav \
      --reference "दूध किती आहे?"

  # Run against all 124 test samples (uses demo_data/samples.json):
  python3 scripts/test_deaf_endpoint.py --all_samples

  # Test locally (without RunPod) — runs handler_deaf.py directly:
  python3 scripts/test_deaf_endpoint.py --audio data/deaf_speech/audio/131.wav --local
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.request

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RAILWAY_BASE = "https://deafspeechcollector-production.up.railway.app"
RUNPOD_API_BASE = "https://api.runpod.ai/v2"


def load_wav_as_base64(wav_path: str) -> str:
    with open(wav_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def load_url_as_base64(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return base64.b64encode(resp.read()).decode("ascii")


def call_runpod(endpoint_id: str, api_key: str, payload: dict, timeout: int = 120) -> dict:
    """POST to /runsync and return the parsed response."""
    url = f"{RUNPOD_API_BASE}/{endpoint_id}/runsync?wait={timeout * 1000}"
    body = json.dumps({"input": payload}).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout + 10) as resp:
        return json.loads(resp.read())


def call_local(payload: dict) -> dict:
    """Run handler_deaf.py directly (no RunPod, for local smoke testing)."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)
    from runpod.handler_deaf import handler  # noqa: PLC0415
    return handler({"id": "local-test", "input": payload})


def compute_wer(reference: str, hypothesis: str) -> float:
    try:
        from jiwer import wer
        return wer(reference, hypothesis)
    except ImportError:
        return float("nan")


def print_result(result: dict, reference: str | None = None):
    sep = "─" * 64
    print(sep)
    if "error" in result:
        print(f"  ERROR     : {result['error']}")
        print(sep)
        return

    raw = result.get("raw", "")
    corrected = result.get("corrected", "")
    mode = result.get("mode", "?")
    lat = result.get("latency_ms", {})

    print(f"  Raw ASR   : {raw or '(empty)'}")
    print(f"  Corrected : {corrected or '(empty)'}  [{mode}]")
    if reference:
        wer_raw = compute_wer(reference, raw)
        wer_corr = compute_wer(reference, corrected)
        print(f"  Reference : {reference}")
        print(f"  WER raw   : {wer_raw:.1%}")
        print(f"  WER corr  : {wer_corr:.1%}  ({'↓ improved' if wer_corr < wer_raw else '─ same' if wer_corr == wer_raw else '↑ worse'})")
    asr_ms = lat.get("asr", 0)
    pp_ms = lat.get("postprocess", 0)
    total_ms = lat.get("total", 0)
    print(f"  Latency   : ASR {asr_ms}ms | Post-process {pp_ms}ms | Total {total_ms}ms")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test the RunPod Deaf Speech serverless endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--audio", help="Path to local 16 kHz mono WAV file")
    parser.add_argument("--audio_url", help="URL to a 16 kHz WAV (e.g. Railway recording URL)")
    parser.add_argument("--reference", help="Reference transcription (optional, for WER)")
    parser.add_argument("--all_samples", action="store_true",
                        help="Run all 124 samples from demo_data/samples.json")
    parser.add_argument("--local", action="store_true",
                        help="Run handler locally (no RunPod) for smoke testing")
    parser.add_argument("--endpoint_id",
                        default=os.environ.get("RUNPOD_ENDPOINT_ID", ""),
                        help="RunPod endpoint ID (or set RUNPOD_ENDPOINT_ID env var)")
    parser.add_argument("--api_key",
                        default=os.environ.get("RUNPOD_API_KEY", ""),
                        help="RunPod API key (or set RUNPOD_API_KEY env var)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Request timeout in seconds (default: 120)")
    args = parser.parse_args()

    if not args.local and not args.endpoint_id:
        print("ERROR: Provide --endpoint_id or set RUNPOD_ENDPOINT_ID env var", file=sys.stderr)
        sys.exit(1)
    if not args.local and not args.api_key:
        print("ERROR: Provide --api_key or set RUNPOD_API_KEY env var", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Mode: all samples
    # ------------------------------------------------------------------
    if args.all_samples:
        samples_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "demo_data", "samples.json"
        )
        with open(samples_path, encoding="utf-8") as f:
            data = json.load(f)

        samples = data["samples"]
        print(f"Running {len(samples)} samples …\n")
        errors = 0
        for i, s in enumerate(samples):
            print(f"[{i+1}/{len(samples)}] ID={s['id']}  ref={s['reference']}")
            try:
                audio_b64 = load_url_as_base64(s["audio_url"])
                payload = {"audio_base64": audio_b64}
                if args.local:
                    result = call_local(payload)
                else:
                    response = call_runpod(args.endpoint_id, args.api_key, payload, args.timeout)
                    result = response.get("output", response)
                print_result(result, reference=s["reference"])
            except Exception as exc:
                print(f"  FAILED: {exc}")
                errors += 1
        print(f"\nDone. Errors: {errors}/{len(samples)}")
        return

    # ------------------------------------------------------------------
    # Mode: single sample
    # ------------------------------------------------------------------
    if not args.audio and not args.audio_url:
        print("ERROR: Provide --audio, --audio_url, or --all_samples", file=sys.stderr)
        sys.exit(1)

    print("Encoding audio …")
    if args.audio:
        audio_b64 = load_wav_as_base64(args.audio)
        print(f"  Source: {args.audio}  ({len(audio_b64) * 3 // 4 // 1024} KB)")
    else:
        audio_b64 = load_url_as_base64(args.audio_url)
        print(f"  Source: {args.audio_url}")

    payload = {"audio_base64": audio_b64}

    print("Calling endpoint …")
    t0 = time.perf_counter()
    if args.local:
        result = call_local(payload)
    else:
        response = call_runpod(args.endpoint_id, args.api_key, payload, args.timeout)
        result = response.get("output", response)
    wall_sec = time.perf_counter() - t0
    print(f"  Round-trip: {wall_sec:.1f}s\n")

    print_result(result, reference=args.reference)


if __name__ == "__main__":
    main()
