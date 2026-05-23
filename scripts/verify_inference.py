#!/usr/bin/env python3
"""End-to-end inference verification for the Amchi or Deaf Speech ASR.

Runs on any GPU host (Google Colab, RunPod GPU pod, local NVIDIA workstation).
Downloads the checkpoint from R2, downloads a handful of held-out test
audio samples from the Railway data API, runs the production inference
pipeline against them, and compares each transcription to the prediction
stored in ``final_test_results.json``.

If the fresh transcription closely matches the stored one (CER ≤
``--tolerance``) for every sample, the inference pipeline is verified
working — safe to bake into a Docker image and deploy to RunPod.

Usage
-----
::

    # On Colab / GPU host with NeMo installed and the conv_asr patch applied:
    pip install jiwer
    export GEMINI_API_KEY=...    # optional; enables post-processing
    python scripts/verify_inference.py --variant amchi    # uses default R2 URL
    python scripts/verify_inference.py --variant deaf

    # Override settings:
    python scripts/verify_inference.py \\
        --variant amchi \\
        --checkpoint-url <r2 url> \\
        --num-samples 5 \\
        --tolerance 0.05

Exit codes
----------
0   Every sample's fresh transcription matched the stored prediction
    within tolerance — inference pipeline verified.
1   At least one sample drifted beyond tolerance — pipeline regression.
2   Missing prerequisite (no GPU, NeMo not installed, no checkpoint, etc.)
3   Bad CLI argument.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.error
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINTS = {
    "amchi": (
        "https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev/"
        "results/run_c_stratified_split/checkpoints/"
        "konkani_asr-epoch=88-val_wer=0.334.ckpt"
    ),
    "deaf": (
        "https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev/"
        "results/deaf_speech_dsd/checkpoints/"
        "konkani_asr-epoch=96-val_wer=0.269.ckpt"
    ),
}

DEFAULT_GOLDEN = {
    "amchi": (
        REPO_ROOT
        / "results/run_c_stratified_split/experiments/"
        / "20260307_181510/final_test_results.json"
    ),
    "deaf": (
        REPO_ROOT
        / "results/deaf_speech_dsd/experiments/"
        / "20260307_224737/final_test_results.json"
    ),
}

# Railway data sources for downloading the held-out test audio.
RAILWAY_AUDIO_URLS = {
    "amchi": "https://konkanicollector-production.up.railway.app/api/recordings/{id}/audio",
    "deaf":  "https://deafspeechcollector-production.up.railway.app/api/recordings/{id}/audio",
}

DEFAULT_TOLERANCE = 0.05
DEFAULT_NUM_SAMPLES = 5


# ---------------------------------------------------------------------------
# Pure helpers (testable without network / model)
# ---------------------------------------------------------------------------

def extract_recording_id(audio_path: str) -> str:
    """Return the recording id from an audio_path like ``data/.../audio/551.wav``.

    The id is the basename minus the ``.wav`` extension. Used to build the
    Railway audio download URL.

    Parameters
    ----------
    audio_path
        Path string as stored in ``final_test_results.json`` (e.g.
        ``"data/amchi/test/audio/592.wav"``).

    Returns
    -------
    str
        The recording id (e.g. ``"592"``).

    Raises
    ------
    ValueError
        If the path's basename does not end in ``.wav``.
    """
    if not audio_path:
        raise ValueError("audio_path must be a non-empty string")
    name = Path(audio_path).name
    if not name.lower().endswith(".wav"):
        raise ValueError(f"Expected a .wav file, got {name!r}")
    return name[:-4]


def select_samples(per_sample: list, num: int) -> list:
    """Pick the first ``num`` samples that have all required fields.

    Filters out malformed entries (missing audio / prediction / reference).
    Stable order — preserves the order in the source JSON so the test set
    is reproducible.
    """
    if num <= 0:
        raise ValueError("num must be > 0")
    out = []
    for entry in per_sample:
        if not isinstance(entry, dict):
            continue
        if not (entry.get("audio") and isinstance(entry["audio"], str)):
            continue
        if entry.get("prediction") is None or entry.get("reference") is None:
            continue
        out.append(entry)
        if len(out) >= num:
            break
    return out


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Compute character-level edit distance / len(reference).

    Handles edge cases jiwer raises on:
      - Both strings empty → CER 0.0 (identical)
      - Only one empty → CER 1.0 (totally wrong)
      - Otherwise delegates to jiwer.cer
    """
    ref = (reference or "").strip()
    hyp = (hypothesis or "").strip()
    if not ref and not hyp:
        return 0.0
    if not ref or not hyp:
        return 1.0
    from jiwer import cer  # local import — jiwer is heavy
    return cer(ref, hyp)


def format_sample_row(*, idx: int, sample: dict, fresh: str, cer_value: float,
                     tolerance: float) -> str:
    """Build a one-line human-readable result row for the summary table."""
    status = "OK  " if cer_value <= tolerance else "FAIL"
    audio = sample.get("audio", "?")
    return (
        f"  [{status}] sample {idx} cer={cer_value:.4f}  audio={audio}\n"
        f"           stored: {sample.get('prediction', '')[:80]}\n"
        f"           fresh:  {fresh[:80]}"
    )


# ---------------------------------------------------------------------------
# Network / model — runs only on a real GPU host
# ---------------------------------------------------------------------------

def download_checkpoint(url: str, dest_path: str, *, progress_every_mb: int = 50) -> int:
    """Download the checkpoint from R2 with a sensible User-Agent.

    Streams to disk; prints progress every ``progress_every_mb`` megabytes.

    Returns
    -------
    int
        Bytes downloaded.
    """
    from scripts.runpod_http import download_url_to_path

    print(f"  downloading {url}")
    last_mb_printed = 0

    def progress(downloaded: int, total: int):
        nonlocal last_mb_printed
        mb = downloaded // (1024 * 1024)
        if mb >= last_mb_printed + progress_every_mb:
            if total > 0:
                pct = 100.0 * downloaded / total
                print(f"    {mb} MB / {total // (1024*1024)} MB ({pct:.0f}%)")
            else:
                print(f"    {mb} MB")
            last_mb_printed = mb

    n = download_url_to_path(url, dest_path, progress_callback=progress)
    print(f"  done: {n / (1024*1024):.1f} MB → {dest_path}")
    return n


def download_audio_sample(recording_id: str, dest_path: str, variant: str) -> int:
    """Download one test audio file from the Railway data API.

    Returns bytes downloaded; raises on network/HTTP error.
    """
    from scripts.runpod_http import download_url_to_path

    url = RAILWAY_AUDIO_URLS[variant].format(id=recording_id)
    return download_url_to_path(url, dest_path, timeout=120)


def load_model(checkpoint_path: str, variant: str):
    """Load the model using the production inference pipeline.

    Delegates to ``scripts/amchi_inference.py`` or
    ``scripts/deaf_speech_inference.py`` so we exercise the same code path
    the RunPod handler uses.
    """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. Inference on CPU is too slow for this test."
        )
    device = "cuda"

    if variant == "amchi":
        from scripts.amchi_inference import load_model_from_ckpt
        return load_model_from_ckpt(checkpoint_path, device=device)
    if variant == "deaf":
        from scripts.deaf_speech_inference import load_model as _load_deaf
        return _load_deaf(checkpoint_path, device=device)
    raise ValueError(f"Unknown variant: {variant!r}")


def transcribe_sample(model, audio_path: str, variant: str) -> str:
    """Transcribe a single WAV file via the production inference function."""
    if variant == "amchi":
        from scripts.amchi_inference import transcribe_audio_bytes
        wav_bytes = Path(audio_path).read_bytes()
        return (transcribe_audio_bytes(model, wav_bytes) or "").strip()
    if variant == "deaf":
        from scripts.deaf_speech_inference import transcribe_wav
        text, _elapsed = transcribe_wav(model, audio_path)
        return (text or "").strip()
    raise ValueError(f"Unknown variant: {variant!r}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def verify(
    *,
    variant: str,
    checkpoint_url: str,
    golden_path: Path,
    num_samples: int,
    tolerance: float,
    workdir: Path,
) -> int:
    """Run the full end-to-end verification and return an exit code."""

    # 1. Load golden predictions (from the repo — committed to git)
    if not golden_path.is_file():
        print(f"ERROR: golden file not found: {golden_path}", file=sys.stderr)
        return 2
    with golden_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    samples = select_samples(data.get("per_sample", []), num_samples)
    if not samples:
        print(f"ERROR: no usable samples in {golden_path}", file=sys.stderr)
        return 2

    print(f"==> {variant} verification — {len(samples)} samples, tolerance CER ≤ {tolerance}")

    # 2. Download checkpoint
    ckpt_path = workdir / f"{variant}_checkpoint.ckpt"
    if ckpt_path.is_file():
        print(f"==> Reusing cached checkpoint at {ckpt_path}")
    else:
        print("==> Downloading checkpoint from R2 …")
        try:
            download_checkpoint(checkpoint_url, str(ckpt_path))
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            print(f"ERROR: failed to download checkpoint: {exc}", file=sys.stderr)
            return 2

    # 3. Download audio samples
    audio_dir = workdir / f"{variant}_audio"
    audio_dir.mkdir(exist_ok=True)
    print(f"==> Downloading {len(samples)} test audio samples to {audio_dir}")
    sample_audio_paths: list[Path] = []
    for entry in samples:
        rid = extract_recording_id(entry["audio"])
        dest = audio_dir / f"{rid}.wav"
        if not dest.is_file():
            try:
                download_audio_sample(rid, str(dest), variant)
            except (urllib.error.HTTPError, urllib.error.URLError) as exc:
                print(f"ERROR: failed to download audio {rid}: {exc}", file=sys.stderr)
                return 2
        sample_audio_paths.append(dest)

    # 4. Load model
    print("==> Loading model …")
    t0 = time.perf_counter()
    try:
        model = load_model(str(ckpt_path), variant)
    except Exception as exc:
        print(f"ERROR: model load failed: {exc}", file=sys.stderr)
        return 2
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    # 5. Transcribe + compare each sample
    print("==> Running inference + comparing to stored predictions …")
    failures = 0
    print()
    for i, (entry, audio_path) in enumerate(zip(samples, sample_audio_paths)):
        fresh = transcribe_sample(model, str(audio_path), variant)
        stored = entry["prediction"]
        cer = character_error_rate(stored, fresh)
        passed = cer <= tolerance
        if not passed:
            failures += 1
        print(format_sample_row(
            idx=i, sample=entry, fresh=fresh, cer_value=cer, tolerance=tolerance,
        ))
        print()

    # 6. Summary
    print(f"==> Result: {len(samples) - failures}/{len(samples)} samples within "
          f"CER ≤ {tolerance}")
    if failures:
        print(f"   {failures} sample(s) drifted from stored predictions — pipeline regression.")
        return 1
    print("   Inference pipeline verified. Safe to bake into Docker.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end inference verification for the fine-tuned ASR.",
    )
    p.add_argument("--variant", choices=("amchi", "deaf"), required=True,
                  help="Which model to verify.")
    p.add_argument("--checkpoint-url", default=None,
                  help="Override the default R2 URL (mainly for testing).")
    p.add_argument("--golden", default=None,
                  help="Path to final_test_results.json (default: looks up by variant).")
    p.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
                  help=f"How many samples to run. Default {DEFAULT_NUM_SAMPLES}.")
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                  help=f"Max CER per sample. Default {DEFAULT_TOLERANCE}.")
    p.add_argument("--workdir", default=tempfile.gettempdir(),
                  help="Where to cache downloads. Default the system temp dir.")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)

    checkpoint_url = args.checkpoint_url or DEFAULT_CHECKPOINTS[args.variant]
    golden_path = Path(args.golden) if args.golden else DEFAULT_GOLDEN[args.variant]
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    if args.num_samples <= 0:
        print("ERROR: --num-samples must be > 0", file=sys.stderr)
        return 3
    if not 0 <= args.tolerance <= 1:
        print("ERROR: --tolerance must be in [0, 1]", file=sys.stderr)
        return 3

    return verify(
        variant=args.variant,
        checkpoint_url=checkpoint_url,
        golden_path=golden_path,
        num_samples=args.num_samples,
        tolerance=args.tolerance,
        workdir=workdir,
    )


if __name__ == "__main__":
    sys.exit(main())
