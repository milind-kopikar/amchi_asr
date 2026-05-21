"""Golden transcript regression tests for the Deaf Speech ASR (DS-D).

Mirror of ``test_amchi_golden.py`` but for the Deaf Speech model. See that
file for the rationale + tolerance discussion.

Skipped by default. Enable with::

    export RUN_GOLDEN_TESTS=1
    export CHECKPOINT_URL="https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev/results/deaf_speech_dsd/checkpoints/konkani_asr-epoch=96-val_wer=0.269.ckpt"
    pytest tests/integration/test_deaf_golden.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.integration.conftest import (  # noqa: E402
    audio_path_or_skip,
    load_golden_samples,
)

GOLDEN_CER_TOLERANCE = 0.05


@pytest.fixture(scope="module")
def deaf_model():
    """Load the fine-tuned Deaf Speech model once for the module."""
    import tempfile
    import urllib.request

    from scripts.deaf_speech_inference import load_model

    ckpt_path = os.environ.get("CHECKPOINT_PATH", "")
    ckpt_url = os.environ.get("CHECKPOINT_URL", "")

    if ckpt_path and Path(ckpt_path).is_file():
        local = ckpt_path
    elif ckpt_url:
        local = os.path.join(tempfile.gettempdir(), "deaf_checkpoint.ckpt")
        if not Path(local).is_file():
            urllib.request.urlretrieve(ckpt_url, local)
    else:
        pytest.skip("Set CHECKPOINT_PATH or CHECKPOINT_URL to enable golden tests")

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pytest.skip("PyTorch not installed")
    if device == "cpu":
        pytest.skip("Golden tests require a CUDA GPU (CPU inference is too slow)")

    return load_model(local, device=device)


@pytest.fixture(scope="module")
def golden_samples():
    """5 representative test samples from the official DS-D evaluation."""
    return load_golden_samples("deaf", n=5)


def _character_error_rate(reference: str, hypothesis: str) -> float:
    from jiwer import cer
    return cer(reference or " ", hypothesis or " ")


class TestDeafGoldenTranscriptions:
    """One test per sample — easier to debug than one big batch test."""

    def test_pipeline_loads_model(self, deaf_model):
        assert deaf_model is not None
        assert hasattr(deaf_model, "transcribe")

    def test_sample_0(self, deaf_model, golden_samples):
        self._assert_close(deaf_model, golden_samples[0])

    def test_sample_1(self, deaf_model, golden_samples):
        self._assert_close(deaf_model, golden_samples[1])

    def test_sample_2(self, deaf_model, golden_samples):
        self._assert_close(deaf_model, golden_samples[2])

    def test_sample_3(self, deaf_model, golden_samples):
        self._assert_close(deaf_model, golden_samples[3])

    def test_sample_4(self, deaf_model, golden_samples):
        self._assert_close(deaf_model, golden_samples[4])

    @staticmethod
    def _assert_close(model, sample: dict) -> None:
        from scripts.deaf_speech_inference import transcribe_wav

        audio_path = audio_path_or_skip(sample["audio"])
        stored_prediction = sample["prediction"]

        # The deaf inference API takes a file path (not bytes).
        fresh_prediction, _elapsed = transcribe_wav(model, str(audio_path))
        fresh_prediction = (fresh_prediction or "").strip()

        print()
        print(f"audio:           {sample['audio']}")
        print(f"reference (ref): {sample.get('reference', '')[:80]}")
        print(f"stored (golden): {stored_prediction[:80]}")
        print(f"fresh:           {fresh_prediction[:80]}")

        cer = _character_error_rate(stored_prediction, fresh_prediction)
        print(f"CER (fresh vs stored): {cer:.4f}  (tolerance {GOLDEN_CER_TOLERANCE})")
        assert cer <= GOLDEN_CER_TOLERANCE, (
            f"Fresh transcription drifted from stored prediction (CER={cer:.4f}). "
            f"Inference pipeline may have regressed.\n"
            f"  stored: {stored_prediction!r}\n"
            f"  fresh:  {fresh_prediction!r}"
        )
