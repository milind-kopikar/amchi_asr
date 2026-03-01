#!/usr/bin/env python3
"""
Deaf Speech ASR Inference Script

Loads the fine-tuned deaf speech checkpoint, transcribes a WAV file via CTC,
optionally runs Gemini post-processing, and reports results with latency.

Usage:
  python3 scripts/deaf_speech_inference.py \\
      --checkpoint nemo_experiments/deaf_speech_story4_50epoch/checkpoints/konkani_asr-epoch=21-val_wer=0.720.ckpt \\
      --audio data/deaf_speech/audio/131.wav \\
      --gemini_key <YOUR_GEMINI_KEY>

  # ASR only (no post-processing):
  python3 scripts/deaf_speech_inference.py \\
      --checkpoint <ckpt> --audio <wav> --no_postprocess
"""

import argparse
import logging
import os
import sys
import time

import torch
from omegaconf import DictConfig, OmegaConf

# Silence NeMo / PyTorch Lightning verbose logs before importing nemo
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for _name in ("nemo", "nemo_logger", "pytorch_lightning", "lightning", "lightning.pytorch"):
    logging.getLogger(_name).setLevel(logging.ERROR)

import nemo.collections.asr as nemo_asr  # noqa: E402 — must come after logging config

# Import post-processing helpers from sibling module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from postprocess_asr import (  # noqa: E402
    GEMINI_MODEL,
    normalize_for_wer,
    postprocess_sample,
)

logger = logging.getLogger("deaf_infer")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: str = "cuda"):
    """Load fine-tuned checkpoint using the config-patch + strict=False pattern.

    The checkpoint config has two problems that prevent direct loading:
      1. loss_name='ctc' — the Hybrid RNNT validator rejects this.
      2. train_ds / validation_ds / test_ds point to paths that don't exist
         in the inference environment.

    Solution (from REPRODUCTION_NOTES.md §9):
      - Patch loss_name → 'default'
      - Remove dataset configs before __init__
      - Re-add empty stubs after __init__ (transcribe() needs them)
      - Load state dict with strict=False
      - Force CTC decoding strategy
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "hyper_parameters" not in ckpt or "cfg" not in ckpt["hyper_parameters"]:
        raise RuntimeError(
            f"Checkpoint does not contain 'hyper_parameters.cfg': {ckpt_path}"
        )

    cfg = ckpt["hyper_parameters"]["cfg"]
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    # Patch 1: loss_name 'ctc' → 'default' (RNNT validator requires RNNT loss names)
    OmegaConf.set_struct(cfg, False)
    if "loss" in cfg and cfg.loss.get("loss_name") == "ctc":
        cfg.loss.loss_name = "default"

    # Patch 2: Remove dataset configs to avoid file-not-found during __init__
    for key in ("train_ds", "validation_ds", "test_ds"):
        if key in cfg:
            del cfg[key]
    OmegaConf.set_struct(cfg, True)

    # Instantiate model shell from patched config
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel(cfg=cfg)

    # Restore empty dataset stubs — transcribe() checks for their existence
    OmegaConf.set_struct(model.cfg, False)
    model.cfg.validation_ds = OmegaConf.create({})
    model.cfg.test_ds = OmegaConf.create({})
    OmegaConf.set_struct(model.cfg, True)

    # Load weights; strict=False because the hybrid shell has more params than
    # the CTC-only checkpoint (RNNT joint/decoder weights are missing — expected)
    model.load_state_dict(ckpt["state_dict"], strict=False)

    model.to(device)
    model.eval()

    # Force CTC decoding (the model was trained CTC-only)
    model.change_decoding_strategy(decoder_type="ctc")

    return model


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_wav(model, audio_path: str):
    """Run ASR on a single WAV file. Returns (prediction_text, latency_seconds)."""
    t0 = time.perf_counter()
    predictions = model.transcribe(audio=[audio_path])
    latency = time.perf_counter() - t0

    pred_text = ""
    if predictions:
        p = predictions[0]
        pred_text = p if isinstance(p, str) else p.text

    return pred_text.strip(), latency


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def run_postprocess(api_key: str, raw_asr: str):
    """
    Call Gemini post-processing on the raw ASR output.

    Note: In inference mode we don't have a reference transcription, so the
    safety-valve (revert-if-worse) used in batch processing is not applied.
    We trust Gemini's output directly.

    Returns (corrected_text, mode, latency_seconds).
    """
    import google.genai as genai

    client = genai.Client(api_key=api_key)

    t0 = time.perf_counter()
    # original_wer=1.0 → not perfect, so all modes are eligible (FILL/RECONSTRUCT)
    result = postprocess_sample(client, raw_asr, original_wer=1.0)
    latency = time.perf_counter() - t0

    return result["corrected"], result["mode"], latency


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    default_ckpt = (
        "nemo_experiments/deaf_speech_story4_50epoch/checkpoints/"
        "konkani_asr-epoch=21-val_wer=0.720.ckpt"
    )

    parser = argparse.ArgumentParser(
        description="Deaf Speech ASR inference with optional Gemini post-processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        default=default_ckpt,
        help="Path to fine-tuned .ckpt file (default: best epoch-21 checkpoint)",
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to input WAV file (must be 16kHz mono)",
    )
    parser.add_argument(
        "--gemini_key",
        default=os.environ.get("GEMINI_API_KEY", ""),
        help="Gemini API key for post-processing (or set GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--no_postprocess",
        action="store_true",
        help="Skip Gemini post-processing (output raw ASR only)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use: cuda or cpu (default: auto-detect)",
    )
    args = parser.parse_args()

    # --- Validate inputs ---
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.audio):
        print(f"ERROR: Audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    do_postprocess = not args.no_postprocess and bool(args.gemini_key)
    if not args.no_postprocess and not args.gemini_key:
        print(
            "WARNING: No --gemini_key provided. "
            "Skipping post-processing. Pass --no_postprocess to suppress this warning.\n",
            file=sys.stderr,
        )

    # --- Step 1: Load model ---
    print(f"\n[1/3] Loading model on {args.device}...")
    t_load = time.perf_counter()
    model = load_model(args.checkpoint, device=args.device)
    load_time = time.perf_counter() - t_load
    print(f"      Model ready  ({load_time:.1f}s)")

    # --- Step 2: Transcribe ---
    print(f"\n[2/3] Transcribing: {os.path.basename(args.audio)}")
    raw_asr, asr_latency = transcribe_wav(model, args.audio)
    print(f"      ASR done      ({asr_latency:.2f}s)")

    # --- Step 3: Post-process ---
    if do_postprocess:
        print(f"\n[3/3] Post-processing with Gemini ({GEMINI_MODEL})...")
        corrected, mode, pp_latency = run_postprocess(args.gemini_key, raw_asr)
        total_latency = asr_latency + pp_latency
        print(f"      Post-process done ({pp_latency:.2f}s)  mode={mode}")
    else:
        corrected = normalize_for_wer(raw_asr)
        mode = "SKIPPED"
        pp_latency = 0.0
        total_latency = asr_latency

    # --- Results ---
    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  Audio     : {args.audio}")
    print(f"  Raw ASR   : {raw_asr if raw_asr else '(empty)'}")
    if mode != "SKIPPED":
        print(f"  Corrected : {corrected}  [{mode}]")
    if mode != "SKIPPED":
        print(
            f"  Latency   : ASR {asr_latency:.2f}s"
            f" | Post-process {pp_latency:.2f}s"
            f" | Total {total_latency:.2f}s"
        )
    else:
        print(f"  Latency   : ASR {asr_latency:.2f}s")
    print(sep)


if __name__ == "__main__":
    main()
