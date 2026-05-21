"""Golden transcript regression tests for the Amchi Konkani ASR (Run S).

Loads the fine-tuned checkpoint, transcribes 5 representative samples from
the held-out test set, and verifies the fresh transcription closely matches
the stored prediction. Catches regressions in:

  - the checkpoint-loading patch sequence
    (``cfg.loss.loss_name = 'default'``, delete train/dev/test_ds, strict=False)
  - audio decoding / sample-rate handling
  - the CTC decoding strategy
  - changes to ``scripts/amchi_inference.py``

These tests do NOT check WER against the reference — that catches model
quality regressions, which is a separate concern. They check that the
*same model* produces *the same output* it produced when officially
evaluated. A real change in output (CER > 5% from the stored prediction)
means the inference pipeline has shifted.

Skipped by default. Enable via::

    export RUN_GOLDEN_TESTS=1
    export CHECKPOINT_URL="https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev/results/run_c_stratified_split/checkpoints/konkani_asr-epoch=88-val_wer=0.334.ckpt"
    pytest tests/integration/test_amchi_golden.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# conftest.py already added the repo root to sys.path; re-add defensively.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.integration.conftest import (  # noqa: E402
    audio_path_or_skip,
    load_golden_samples,
)

# Tolerance: a fresh transcription must be within this CER of the stored one.
# Set high enough to absorb minor non-determinism (e.g. CUDA float ordering),
# low enough to catch a real pipeline change.
GOLDEN_CER_TOLERANCE = 0.05


# ---------------------------------------------------------------------------
# Module-scoped fixtures — the model load is the expensive part (~30s),
# so we share it across every sample test in this file.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def amchi_model():
    """Load the fine-tuned Amchi Konkani model once for the module.

    Resolves the checkpoint from CHECKPOINT_PATH or CHECKPOINT_URL the same
    way the production handler does, then applies the loading-patch
    sequence and forces CTC decoding.
    """
    import tempfile
    import urllib.request

    from scripts.amchi_inference import load_model_from_ckpt

    ckpt_path = os.environ.get("CHECKPOINT_PATH", "")
    ckpt_url = os.environ.get("CHECKPOINT_URL", "")

    if ckpt_path and Path(ckpt_path).is_file():
        local = ckpt_path
    elif ckpt_url:
        local = os.path.join(tempfile.gettempdir(), "amchi_checkpoint.ckpt")
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

    model = load_model_from_ckpt(local, device=device)
    return model


@pytest.fixture(scope="module")
def golden_samples():
    """5 representative test samples from the official Run S evaluation."""
    return load_golden_samples("amchi", n=5)


# ---------------------------------------------------------------------------
# Pure utilities (no model) — these can be tested independently if needed
# ---------------------------------------------------------------------------

def _read_audio_bytes(audio_path: Path) -> bytes:
    """Read the WAV file as bytes (what the handler receives)."""
    return audio_path.read_bytes()


def _character_error_rate(reference: str, hypothesis: str) -> float:
    """Compute character-level edit distance / len(reference). 0 = identical."""
    from jiwer import cer
    return cer(reference or " ", hypothesis or " ")


# ---------------------------------------------------------------------------
# Golden tests
# ---------------------------------------------------------------------------

class TestAmchiGoldenTranscriptions:
    """One test per representative sample — pytest reports each one's status."""

    def test_pipeline_loads_model(self, amchi_model):
        """Smoke: the module-scoped fixture itself succeeded."""
        assert amchi_model is not None
        assert hasattr(amchi_model, "transcribe")

    def test_sample_0_matches_stored_prediction(self, amchi_model, golden_samples):
        self._assert_close(amchi_model, golden_samples[0])

    def test_sample_1_matches_stored_prediction(self, amchi_model, golden_samples):
        self._assert_close(amchi_model, golden_samples[1])

    def test_sample_2_matches_stored_prediction(self, amchi_model, golden_samples):
        self._assert_close(amchi_model, golden_samples[2])

    def test_sample_3_matches_stored_prediction(self, amchi_model, golden_samples):
        self._assert_close(amchi_model, golden_samples[3])

    def test_sample_4_matches_stored_prediction(self, amchi_model, golden_samples):
        self._assert_close(amchi_model, golden_samples[4])

    # ---------------------------------------------------------------------
    # Shared body
    # ---------------------------------------------------------------------

    @staticmethod
    def _assert_close(model, sample: dict) -> None:
        """Run the inference pipeline against one sample and assert CER tolerance."""
        from scripts.amchi_inference import transcribe_audio_bytes

        audio_path = audio_path_or_skip(sample["audio"])
        stored_prediction = sample["prediction"]

        wav_bytes = _read_audio_bytes(audio_path)
        fresh_prediction = (transcribe_audio_bytes(model, wav_bytes) or "").strip()

        # Print a side-by-side so a failing assertion is immediately readable.
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
