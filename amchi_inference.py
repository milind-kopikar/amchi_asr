#!/usr/bin/env python3
"""
Shared inference for Amchi ASR: load fine-tuned .ckpt and transcribe audio.
Used by: smoke_test_inference (CLI), RunPod handler, demo_test_set_wer.
Expects 16 kHz mono WAV; returns raw transcription (no post-processing).
"""
import os
import tempfile
import logging
from typing import List, Union

import torch
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger("amchi_inference")

# Lazy NeMo import so CLI can fail fast if not installed
_nemo_asr = None

def _get_nemo_asr():
    global _nemo_asr
    if _nemo_asr is None:
        import nemo.collections.asr as nemo_asr
        _nemo_asr = nemo_asr
    return _nemo_asr


def load_model_from_ckpt(checkpoint_path: str, device: str = "cuda") -> "torch.nn.Module":
    """Load the Amchi ASR model from a fine-tuned .ckpt (same logic as smoke_test_inference)."""
    nemo_asr = _get_nemo_asr()
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "hyper_parameters" not in ckpt or "cfg" not in ckpt["hyper_parameters"]:
        raise ValueError("Checkpoint missing hyper_parameters.cfg")

    cfg = ckpt["hyper_parameters"]["cfg"]
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    if "loss" in cfg and cfg.loss.get("loss_name") == "ctc":
        cfg.loss.loss_name = "default"
    OmegaConf.set_struct(cfg, False)
    for key in ["train_ds", "validation_ds", "test_ds"]:
        if key in cfg:
            del cfg[key]
    OmegaConf.set_struct(cfg, True)

    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel(cfg=cfg)
    OmegaConf.set_struct(model.cfg, False)
    model.cfg.validation_ds = OmegaConf.create({})
    model.cfg.test_ds = OmegaConf.create({})
    OmegaConf.set_struct(model.cfg, True)

    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device)
    model.eval()

    try:
        model.change_decoding_strategy(decoder_type="ctc")
    except Exception:
        model.cfg.decoder_type = "ctc"

    return model


def _extract_text(pred) -> str:
    if pred is None:
        return ""
    if isinstance(pred, str):
        return pred
    if hasattr(pred, "text"):
        return getattr(pred, "text", "") or ""
    return str(pred)


def transcribe_audio(model: "torch.nn.Module", audio: Union[str, List[str]]) -> List[str]:
    """
    Transcribe one or more 16 kHz mono WAV paths.
    Returns list of transcription strings (same order as input).
    """
    paths = [audio] if isinstance(audio, str) else list(audio)
    if not paths:
        return []
    # NeMo Hybrid model expects transcribe(audio=paths)
    predictions = model.transcribe(audio=paths)
    if not isinstance(predictions, list):
        predictions = [predictions]
    return [_extract_text(p) for p in predictions]


def transcribe_audio_bytes(model: "torch.nn.Module", wav_bytes: bytes) -> str:
    """Transcribe a single in-memory WAV (e.g. from base64). Writes to temp file and calls transcribe."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        path = f.name
    try:
        out = transcribe_audio(model, path)
        return out[0] if out else ""
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass
