"""Common fixtures + skip-guards for the integration test suite.

These tests load the real fine-tuned checkpoint and run actual ASR. They
take 30s–2 min each and require:
  - a CUDA GPU (or they fall back to CPU and run very slowly)
  - the checkpoint downloadable from R2 (or already at CHECKPOINT_PATH)
  - the held-out test audio under ``data/<variant>/test/audio/`` (or
    similar — script-specific path is in each test file)

By default the tests are SKIPPED unless ``RUN_GOLDEN_TESTS=1`` is set. This
keeps the unit test suite fast and CI-friendly.

To enable locally::

    export RUN_GOLDEN_TESTS=1
    pytest tests/integration/ -v -k amchi   # just amchi
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Make the repo root importable so tests can ``from scripts import ...``.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_collection_modifyitems(config, items):  # noqa: D401
    """Auto-skip integration tests in this folder unless ``RUN_GOLDEN_TESTS`` is set.

    Important: only skip items whose path is under ``tests/integration/``.
    Pytest invokes this hook for every collected item across the whole suite,
    so an unconditional skip-all would silence the unit tests too.
    """
    if os.environ.get("RUN_GOLDEN_TESTS"):
        return
    skip = pytest.mark.skip(reason="Set RUN_GOLDEN_TESTS=1 to enable golden tests")
    integration_dir = Path(__file__).resolve().parent
    for item in items:
        item_path = Path(str(item.fspath)).resolve()
        try:
            item_path.relative_to(integration_dir)
        except ValueError:
            continue  # not under tests/integration/ — leave it alone
        item.add_marker(skip)


# ---------------------------------------------------------------------------
# Resource locator helpers — keep filesystem assumptions in one place
# ---------------------------------------------------------------------------

def golden_predictions_path(variant: str) -> Path:
    """Return the path to the stored ``final_test_results.json`` for a variant.

    These files are the source of truth for "what the model produced on the
    held-out test set when we officially evaluated it". Golden tests compare
    a fresh inference to the ``prediction`` field of each entry.
    """
    if variant == "amchi":
        return REPO_ROOT / (
            "results/run_c_stratified_split/experiments/"
            "20260307_181510/final_test_results.json"
        )
    if variant == "deaf":
        return REPO_ROOT / (
            "results/deaf_speech_dsd/experiments/"
            "20260307_224737/final_test_results.json"
        )
    raise ValueError(f"Unknown variant: {variant!r}")


def load_golden_samples(variant: str, n: int = 5) -> list[dict]:
    """Load the first ``n`` samples from the official ``final_test_results.json``.

    Each returned dict has at least ``audio``, ``reference``, ``prediction``,
    ``wer`` (per the format used in both Amchi and Deaf result files).

    Skips the test if the file is not present locally — these files ARE in
    git so missing-file usually means the repo wasn't fully checked out.
    """
    path = golden_predictions_path(variant)
    if not path.is_file():
        pytest.skip(f"Golden results not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    samples = data.get("per_sample", [])
    if not samples:
        pytest.skip(f"No per_sample entries in {path}")
    return samples[:n]


def audio_path_or_skip(repo_relative_path: str) -> Path:
    """Resolve an audio path relative to the repo root; skip if missing.

    The held-out audio files are NOT in git (too large) so on a fresh checkout
    we skip rather than fail.
    """
    p = REPO_ROOT / repo_relative_path
    if not p.is_file():
        pytest.skip(f"Audio not present: {repo_relative_path} (download via scripts/download_data_from_railway.py)")
    return p
