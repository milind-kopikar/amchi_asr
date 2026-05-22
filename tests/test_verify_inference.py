"""Unit tests for the pure helpers in ``scripts/verify_inference.py``.

The script's heavy dependencies (NeMo, network downloads, GPU) are NOT
covered here — they need a real GPU environment to run. These tests focus
on the I/O-free logic:

  - Recording-ID extraction from audio paths
  - Sample selection / filtering
  - CER computation (delegates to jiwer)
  - Result-row formatting
  - CLI argument validation
  - Exit codes for the early-return error paths

Run from the repo root::

    pytest tests/test_verify_inference.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from scripts.verify_inference import (
    DEFAULT_CHECKPOINTS,
    DEFAULT_GOLDEN,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_TOLERANCE,
    character_error_rate,
    extract_recording_id,
    format_sample_row,
    main,
    select_samples,
    verify,
)


# ---------------------------------------------------------------------------
# extract_recording_id
# ---------------------------------------------------------------------------

class TestExtractRecordingId:
    """Path-parsing logic — must be robust to the variations in
    ``final_test_results.json``'s audio field."""

    def test_typical_path(self):
        assert extract_recording_id("data/amchi/test/audio/592.wav") == "592"

    def test_train_split_path(self):
        """Some samples reference data/.../train/audio (the stratified split
        intentionally pulls audio from multiple original folders)."""
        assert extract_recording_id("data/amchi/train/audio/551.wav") == "551"

    def test_deaf_path(self):
        assert extract_recording_id("data/deaf_speech/audio/130.wav") == "130"

    def test_relative_path_without_dirs(self):
        assert extract_recording_id("100.wav") == "100"

    def test_path_with_extra_dots(self):
        """Names with dots in them are tolerated."""
        assert extract_recording_id("audio/foo.bar.wav") == "foo.bar"

    def test_empty_path_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            extract_recording_id("")

    def test_non_wav_extension_raises(self):
        with pytest.raises(ValueError, match=".wav"):
            extract_recording_id("file.mp3")

    def test_uppercase_wav_accepted(self):
        """Some audio dumps use .WAV — accept either case."""
        assert extract_recording_id("foo.WAV") == "foo"


# ---------------------------------------------------------------------------
# select_samples
# ---------------------------------------------------------------------------

class TestSelectSamples:
    """Filter + take-first-N from the per_sample list."""

    def test_typical(self):
        per = [
            {"audio": "a.wav", "prediction": "x", "reference": "x"},
            {"audio": "b.wav", "prediction": "y", "reference": "y"},
            {"audio": "c.wav", "prediction": "z", "reference": "z"},
        ]
        assert len(select_samples(per, 2)) == 2

    def test_skips_entries_missing_audio(self):
        per = [
            {"prediction": "x", "reference": "x"},  # no audio
            {"audio": "ok.wav", "prediction": "y", "reference": "y"},
        ]
        out = select_samples(per, 5)
        assert len(out) == 1
        assert out[0]["audio"] == "ok.wav"

    def test_skips_entries_with_non_string_audio(self):
        per = [
            {"audio": 42, "prediction": "x", "reference": "x"},
            {"audio": "ok.wav", "prediction": "y", "reference": "y"},
        ]
        out = select_samples(per, 5)
        assert len(out) == 1

    def test_skips_entries_missing_prediction(self):
        per = [
            {"audio": "a.wav", "reference": "x"},  # no prediction
            {"audio": "b.wav", "prediction": "y", "reference": "y"},
        ]
        out = select_samples(per, 5)
        assert len(out) == 1

    def test_skips_non_dict_entries(self):
        per = [
            "not a dict",
            None,
            {"audio": "ok.wav", "prediction": "y", "reference": "y"},
        ]
        out = select_samples(per, 5)
        assert len(out) == 1

    def test_empty_input(self):
        assert select_samples([], 5) == []

    def test_zero_num_raises(self):
        with pytest.raises(ValueError, match="num must be > 0"):
            select_samples([], 0)

    def test_negative_num_raises(self):
        with pytest.raises(ValueError, match="num must be > 0"):
            select_samples([], -1)

    def test_returns_at_most_num_even_if_more_available(self):
        per = [{"audio": f"{i}.wav", "prediction": "x", "reference": "x"} for i in range(10)]
        out = select_samples(per, 3)
        assert len(out) == 3

    def test_preserves_order(self):
        per = [{"audio": f"{i}.wav", "prediction": "x", "reference": "x"} for i in range(5)]
        out = select_samples(per, 3)
        assert [s["audio"] for s in out] == ["0.wav", "1.wav", "2.wav"]


# ---------------------------------------------------------------------------
# character_error_rate (thin wrapper over jiwer.cer)
# ---------------------------------------------------------------------------

class TestCharacterErrorRate:

    def test_identical_strings_zero_cer(self):
        assert character_error_rate("hello", "hello") == 0.0

    def test_completely_different(self):
        # 5-character reference, 5-character hypothesis, all different
        # CER = edit distance / len(reference)
        cer = character_error_rate("hello", "world")
        assert cer > 0.5

    def test_both_empty_returns_zero(self):
        """Both empty == identical → CER 0.0."""
        assert character_error_rate("", "") == 0.0

    def test_only_reference_empty_returns_one(self):
        """Empty reference + non-empty hypothesis == totally wrong → CER 1.0."""
        assert character_error_rate("", "something") == 1.0

    def test_only_hypothesis_empty_returns_one(self):
        """Non-empty reference + empty hypothesis == nothing recognised → 1.0."""
        assert character_error_rate("something", "") == 1.0

    def test_one_character_change(self):
        # 5-char ref, 1 substitution → CER 0.2
        cer = character_error_rate("hello", "hellx")
        assert cer == pytest.approx(0.2, abs=0.01)


# ---------------------------------------------------------------------------
# format_sample_row
# ---------------------------------------------------------------------------

class TestFormatSampleRow:

    def test_passing_row_shows_ok(self):
        sample = {"audio": "data/foo/1.wav", "prediction": "hello world"}
        row = format_sample_row(
            idx=0, sample=sample, fresh="hello world",
            cer_value=0.0, tolerance=0.05,
        )
        assert "[OK" in row
        assert "0.0000" in row
        assert "data/foo/1.wav" in row

    def test_failing_row_shows_fail(self):
        sample = {"audio": "data/foo/1.wav", "prediction": "hello"}
        row = format_sample_row(
            idx=0, sample=sample, fresh="world",
            cer_value=0.9, tolerance=0.05,
        )
        assert "[FAIL" in row

    def test_truncates_long_text(self):
        long_text = "x" * 200
        sample = {"audio": "a.wav", "prediction": long_text}
        row = format_sample_row(
            idx=0, sample=sample, fresh=long_text,
            cer_value=0.0, tolerance=0.05,
        )
        # Should not contain 200 x's in a row — truncated to 80
        assert "x" * 100 not in row


# ---------------------------------------------------------------------------
# verify — orchestrator. We test the early-return error paths only;
# the full happy path requires a real GPU.
# ---------------------------------------------------------------------------

class TestVerifyEarlyExits:
    """Exit codes for the no-network / no-model error paths."""

    def test_missing_golden_returns_2(self, tmp_path):
        rc = verify(
            variant="amchi",
            checkpoint_url="http://x.test/file",
            golden_path=tmp_path / "does_not_exist.json",
            num_samples=5,
            tolerance=0.05,
            workdir=tmp_path,
        )
        assert rc == 2

    def test_empty_per_sample_returns_2(self, tmp_path):
        golden = tmp_path / "g.json"
        golden.write_text(json.dumps({"per_sample": []}), encoding="utf-8")
        rc = verify(
            variant="amchi",
            checkpoint_url="http://x.test/file",
            golden_path=golden,
            num_samples=5,
            tolerance=0.05,
            workdir=tmp_path,
        )
        assert rc == 2


# ---------------------------------------------------------------------------
# main — CLI exit codes for bad arguments
# ---------------------------------------------------------------------------

class TestMainCli:
    """argparse-side validation."""

    def test_negative_num_samples_returns_3(self, tmp_path):
        rc = main([
            "--variant", "amchi",
            "--num-samples", "0",
            "--workdir", str(tmp_path),
        ])
        assert rc == 3

    def test_tolerance_above_one_returns_3(self, tmp_path):
        rc = main([
            "--variant", "amchi",
            "--tolerance", "1.5",
            "--workdir", str(tmp_path),
        ])
        assert rc == 3

    def test_invalid_variant_raises_systemexit(self):
        """argparse rejects unknown --variant values with SystemExit(2)."""
        with pytest.raises(SystemExit):
            main(["--variant", "klingon"])

    def test_defaults_resolve_for_amchi(self, tmp_path):
        """When required args supplied and amchi defaults exist, main reaches
        ``verify()``. We force-skip the verify by patching it."""
        with mock.patch(
            "scripts.verify_inference.verify", return_value=0
        ) as mocked:
            rc = main([
                "--variant", "amchi",
                "--workdir", str(tmp_path),
                "--num-samples", "1",
            ])
        assert rc == 0
        assert mocked.called
        kwargs = mocked.call_args.kwargs
        assert kwargs["variant"] == "amchi"
        assert kwargs["checkpoint_url"] == DEFAULT_CHECKPOINTS["amchi"]
        assert kwargs["golden_path"] == DEFAULT_GOLDEN["amchi"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestDefaults:

    def test_default_checkpoints_exist(self):
        assert "amchi" in DEFAULT_CHECKPOINTS
        assert "deaf" in DEFAULT_CHECKPOINTS
        for url in DEFAULT_CHECKPOINTS.values():
            assert url.startswith("https://")
            assert ".r2.dev" in url

    def test_default_tolerance_is_sensible(self):
        # 5% CER tolerance — strict enough to catch pipeline regressions,
        # loose enough to absorb stochastic CUDA noise.
        assert 0.01 <= DEFAULT_TOLERANCE <= 0.1
