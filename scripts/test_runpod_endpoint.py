#!/usr/bin/env python3
"""
Test the Amchi ASR RunPod Serverless endpoint with sample audio.

Usage:
  # Single file (env vars for API key and endpoint ID)
  export RUNPOD_API_KEY=...
  export RUNPOD_ENDPOINT_ID=...
  python scripts/test_runpod_endpoint.py --audio data/amchi/test/audio/570.wav

  # With reference text (prints WER)
  python scripts/test_runpod_endpoint.py --audio data/amchi/test/audio/570.wav --reference "रोहन होड ज़ाल्लो!"

  # All test samples from manifest (prints each transcription and mean WER)
  python scripts/test_runpod_endpoint.py --manifest data/amchi/test/manifest.jsonl
"""
import argparse
import base64
import json
import re
import sys
import urllib.request
from pathlib import Path

# Runsync: wait up to 2 minutes for first request (cold start)
RUNSYNC_WAIT_MS = 120000
RUNPOD_RUNSYNC_URL = "https://api.runpod.ai/v2/{endpoint_id}/runsync?wait={wait_ms}"


def _normalize(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"[^\w\s]", "", t)
    return " ".join(t.split())


def _wer(ref: str, hyp: str) -> float:
    r, h = _normalize(ref).split(), _normalize(hyp).split()
    m, n = len(r), len(h)
    if m == 0:
        return 0.0 if n == 0 else 1.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n] / m


def call_runsync(endpoint_id: str, api_key: str, audio_base64: str, wait_ms: int = RUNSYNC_WAIT_MS) -> dict:
    """POST to RunPod runsync; returns the parsed JSON response."""
    url = RUNPOD_RUNSYNC_URL.format(endpoint_id=endpoint_id, wait_ms=wait_ms)
    body = json.dumps({"input": {"audio_base64": audio_base64}}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def extract_transcription(response: dict) -> tuple:
    """
    Returns (transcription, error_message).
    RunPod may return output as the handler return value or inside a list.
    """
    out = response.get("output")
    if out is None:
        return "", response.get("error", "No output in response")
    if isinstance(out, list) and out:
        out = out[0]
    if isinstance(out, dict):
        if "error" in out:
            return "", out["error"]
        return out.get("transcription", ""), None
    return "", str(out)


def main():
    ap = argparse.ArgumentParser(description="Test RunPod Amchi ASR serverless endpoint")
    ap.add_argument("--endpoint-id", default=None, help="RunPod endpoint ID (or set RUNPOD_ENDPOINT_ID)")
    ap.add_argument("--api-key", default=None, help="RunPod API key (or set RUNPOD_API_KEY)")
    ap.add_argument("--audio", help="Path to a single 16 kHz mono WAV file")
    ap.add_argument("--manifest", help="Path to test manifest JSONL (audio_filepath, text)")
    ap.add_argument("--reference", help="Reference text for single-file WER (optional)")
    ap.add_argument("--wait-ms", type=int, default=RUNSYNC_WAIT_MS, help="Runsync wait timeout in ms")
    args = ap.parse_args()

    endpoint_id = args.endpoint_id or __import__("os").environ.get("RUNPOD_ENDPOINT_ID")
    api_key = args.api_key or __import__("os").environ.get("RUNPOD_API_KEY")
    if not endpoint_id or not api_key:
        print("Error: set --endpoint-id and --api-key, or RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY", file=sys.stderr)
        sys.exit(1)

    if args.audio and args.manifest:
        print("Error: use either --audio or --manifest, not both", file=sys.stderr)
        sys.exit(1)
    if not args.audio and not args.manifest:
        print("Error: provide --audio or --manifest", file=sys.stderr)
        sys.exit(1)

    # Collect (audio_path, reference_text) list
    if args.audio:
        path = Path(args.audio)
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        ref = args.reference or ""
        items = [(str(path), ref)]
    else:
        manifest = Path(args.manifest)
        if not manifest.exists():
            print(f"Error: manifest not found: {manifest}", file=sys.stderr)
            sys.exit(1)
        items = []
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                path = rec.get("audio_filepath") or rec.get("audio_path")
                ref = rec.get("text", "")
                if path:
                    items.append((path, ref))

    if not items:
        print("Error: no audio paths to test", file=sys.stderr)
        sys.exit(1)

    wers = []
    for i, (audio_path, ref) in enumerate(items):
        if not Path(audio_path).exists():
            print(f"[{i+1}/{len(items)}] Skip (missing): {audio_path}")
            continue
        with open(audio_path, "rb") as f:
            wav_bytes = f.read()
        b64 = base64.b64encode(wav_bytes).decode("ascii")
        print(f"[{i+1}/{len(items)}] Sending {audio_path} ({len(wav_bytes)} bytes)...", flush=True)
        try:
            resp = call_runsync(endpoint_id, api_key, b64, wait_ms=args.wait_ms)
        except Exception as e:
            print(f"  Error: {e}")
            continue
        status = resp.get("status", "")
        trans, err = extract_transcription(resp)
        if err:
            print(f"  Error: {err}")
            continue
        print(f"  Transcription: {trans}")
        if ref:
            w = _wer(ref, trans)
            wers.append(w)
            print(f"  Reference:     {ref}")
            print(f"  WER:          {w:.2%}")
        print()

    if wers:
        print(f"Mean WER ({len(wers)} samples): {sum(wers) / len(wers):.2%}")


if __name__ == "__main__":
    main()
