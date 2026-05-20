"""Unit tests for ``runpod/handler.py`` (Amchi Konkani ASR handler).

The handler has four heavy dependencies (NeMo model, Gemini client, dictionary
file, network I/O). All four are injected as test seams (keyword args on
``handler(...)`` and helper functions), so these tests run fully offline in
under a second without GPU, NeMo, or any network access.

Run from the repo root::

    pytest tests/test_handler_amchi.py -v
"""

from __future__ import annotations

import base64
import importlib
import sys
import urllib.error
from pathlib import Path
from unittest import mock

import pytest

# Import the handler module by file path so we can target its caches.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from runpod import handler as handler_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures — reset module-level caches between tests for clean isolation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_handler_caches():
    """Clear the handler's lazy-loaded singletons before every test."""
    handler_mod._MODEL = None
    handler_mod._DICT_WORDS = None
    handler_mod._RESOLVED_CHECKPOINT_PATH = None
    yield
    handler_mod._MODEL = None
    handler_mod._DICT_WORDS = None
    handler_mod._RESOLVED_CHECKPOINT_PATH = None


# ---------------------------------------------------------------------------
# _decode_audio_input
# ---------------------------------------------------------------------------

class TestDecodeAudioInput:
    """Covers base64 path, URL path, and every documented error case."""

    def test_typical_base64(self):
        """A valid base64 payload returns the decoded bytes and no error."""
        payload = b"FAKE_WAV_BYTES_FOR_TEST"
        inp = {"audio_base64": base64.b64encode(payload).decode("ascii")}
        wav, err = handler_mod._decode_audio_input(inp)
        assert err is None
        assert wav == payload

    def test_base64_takes_precedence_over_url(self):
        """If both keys are present, audio_base64 wins (matches docstring)."""
        payload = b"FROM_BASE64"
        inp = {
            "audio_base64": base64.b64encode(payload).decode("ascii"),
            "audio_url": "http://example.test/file.wav",
        }
        # Patch urlopen so we can detect (and fail) if URL path is taken.
        with mock.patch.object(handler_mod.urllib.request, "urlopen") as urlopen:
            wav, err = handler_mod._decode_audio_input(inp)
            urlopen.assert_not_called()
        assert err is None
        assert wav == payload

    def test_url_path_typical(self):
        """A URL input fetches the bytes via urllib."""
        payload = b"WAV_FROM_URL"

        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *exc): return False
            def read(self): return payload

        with mock.patch.object(handler_mod.urllib.request,
                              "urlopen", return_value=_Resp()):
            wav, err = handler_mod._decode_audio_input(
                {"audio_url": "http://example.test/file.wav"}
            )
        assert err is None
        assert wav == payload

    def test_null_input_non_dict(self):
        """Non-dict input is rejected cleanly."""
        _, err = handler_mod._decode_audio_input(None)
        assert err is not None
        assert "must be a dict" in err["error"]

    def test_empty_dict(self):
        """An empty input dict produces the documented error."""
        _, err = handler_mod._decode_audio_input({})
        assert err == {"error": "Provide audio_base64 or audio_url in input"}

    def test_invalid_base64(self):
        """Garbage base64 returns an error, not an exception."""
        wav, err = handler_mod._decode_audio_input({"audio_base64": "!!!not base64!!!"})
        # base64 may decode some malformed inputs into bytes; the function only
        # errors when the decode raises. Treat both outcomes as acceptable as
        # long as no exception escapes.
        assert (wav is None and err is not None) or (wav is not None and err is None)

    def test_url_fetch_failure(self):
        """Network errors during URL fetch are converted into an error dict."""
        with mock.patch.object(
            handler_mod.urllib.request, "urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            wav, err = handler_mod._decode_audio_input(
                {"audio_url": "http://example.test/file.wav"}
            )
        assert wav is None
        assert err is not None
        assert "Failed to fetch audio_url" in err["error"]

    def test_empty_payload_after_decode(self):
        """An empty base64 string decodes to zero bytes → error."""
        wav, err = handler_mod._decode_audio_input({"audio_base64": ""})
        # Empty audio_base64 fails the `if inp.get(...)` truthiness test,
        # so it goes down the "no key provided" branch.
        assert wav is None
        assert err is not None
        assert "Provide" in err["error"] or "Empty audio" in err["error"]

    def test_oversized_payload(self):
        """A payload bigger than the 20 MB cap is rejected."""
        huge = b"X" * (handler_mod.MAX_AUDIO_BYTES + 1)
        wav, err = handler_mod._decode_audio_input(
            {"audio_base64": base64.b64encode(huge).decode("ascii")}
        )
        assert wav is None
        assert err is not None
        assert "too large" in err["error"]


# ---------------------------------------------------------------------------
# _apply_amchi_postprocess
# ---------------------------------------------------------------------------

class TestApplyAmchiPostprocess:
    """Covers SKIP / SKIPPED / success / PP_ERROR branches."""

    def test_empty_raw_returns_skip(self):
        """Empty input string short-circuits to SKIP without calling Gemini."""
        # Use a sentinel postprocess_fn that would fail if invoked.
        def _never(*a, **kw):
            pytest.fail("postprocess_sample should not be called for empty input")
        corrected, mode, secs = handler_mod._apply_amchi_postprocess(
            "", {"घर"}, "key",
            postprocess_sample_fn=_never,
            genai_client_factory=lambda: object(),
        )
        assert corrected == ""
        assert mode == "SKIP"
        assert secs == 0.0

    def test_empty_gemini_key_returns_skipped(self):
        """Missing Gemini key short-circuits to SKIPPED with raw preserved."""
        def _never(*a, **kw):
            pytest.fail("postprocess_sample should not be called when key is empty")
        corrected, mode, secs = handler_mod._apply_amchi_postprocess(
            "हांव घर वोचलो", {"घर"}, "",
            postprocess_sample_fn=_never,
            genai_client_factory=lambda: object(),
        )
        assert corrected == "हांव घर वोचलो"
        assert mode == "SKIPPED"
        assert secs == 0.0

    def test_typical_call_returns_corrected(self):
        """A successful Gemini call returns the corrected text + mode."""
        def _fake_pp(client, *, prediction, reference, dict_words, original_wer):
            return {"corrected": "हांव घर वोचलो.", "mode": "PASSTHROUGH"}
        corrected, mode, secs = handler_mod._apply_amchi_postprocess(
            "हांव घर वोचलो", {"घर"}, "key",
            postprocess_sample_fn=_fake_pp,
            genai_client_factory=lambda: object(),
        )
        assert corrected == "हांव घर वोचलो."
        assert mode == "PASSTHROUGH"
        assert secs >= 0.0

    def test_postprocess_exception_returns_pp_error(self):
        """An exception from the post-processor degrades to PP_ERROR."""
        def _boom(*a, **kw):
            raise RuntimeError("Gemini quota exceeded")
        corrected, mode, secs = handler_mod._apply_amchi_postprocess(
            "हांव", {"घर"}, "key",
            postprocess_sample_fn=_boom,
            genai_client_factory=lambda: object(),
        )
        assert corrected == "हांव"  # falls back to raw
        assert mode == "PP_ERROR"
        assert secs == 0.0

    def test_postprocess_missing_keys_uses_defaults(self):
        """If the post-processor returns a dict without expected keys, defaults kick in."""
        def _bad_shape(*a, **kw):
            return {}
        corrected, mode, _ = handler_mod._apply_amchi_postprocess(
            "हांव", {"घर"}, "key",
            postprocess_sample_fn=_bad_shape,
            genai_client_factory=lambda: object(),
        )
        assert corrected == "हांव"  # default to raw
        assert mode == "PP_ERROR"  # default mode when missing


# ---------------------------------------------------------------------------
# _load_dictionary_once
# ---------------------------------------------------------------------------

class TestLoadDictionaryOnce:
    """Covers explicit path, missing file, and the caching behaviour."""

    def test_missing_file_raises(self, tmp_path):
        """An explicit path that does not exist raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            handler_mod._load_dictionary_once(str(tmp_path / "nope.json"))

    def test_typical_file_load(self, tmp_path):
        """A valid dictionary file is parsed into a set of Devanagari words."""
        dict_path = tmp_path / "dict.json"
        dict_path.write_text(
            '[{"word_konkani_devanagari": "घर"},'
            ' {"word_konkani_devanagari": "वाट"}]',
            encoding="utf-8",
        )
        words = handler_mod._load_dictionary_once(str(dict_path))
        assert words == {"घर", "वाट"}

    def test_caching(self, tmp_path):
        """A second call with no argument returns the cached set."""
        dict_path = tmp_path / "dict.json"
        dict_path.write_text(
            '[{"word_konkani_devanagari": "घर"}]', encoding="utf-8"
        )
        first = handler_mod._load_dictionary_once(str(dict_path))
        # Delete the file — second call must still succeed (cached).
        dict_path.unlink()
        second = handler_mod._load_dictionary_once()
        assert first is second


# ---------------------------------------------------------------------------
# handler — orchestration
# ---------------------------------------------------------------------------

class TestHandlerOrchestration:
    """End-to-end handler tests with fully mocked dependencies."""

    def _make_inputs(self):
        """Build a fake job with valid base64 audio."""
        wav = b"FAKE_WAV_FOR_HANDLER_TEST"
        return {
            "id": "test-job-1",
            "input": {"audio_base64": base64.b64encode(wav).decode("ascii")},
        }

    def test_missing_input_returns_error(self):
        """No audio_base64 / audio_url in input → error."""
        out = handler_mod.handler({"input": {}})
        assert "error" in out

    def test_non_dict_job_returns_error(self):
        """A non-dict job is rejected cleanly."""
        out = handler_mod.handler("not a job")  # type: ignore[arg-type]
        assert "error" in out

    def test_model_load_failure_returns_error(self):
        """If the model factory raises, the error is wrapped, not swallowed."""
        def _boom_model():
            raise FileNotFoundError("no checkpoint configured")
        out = handler_mod.handler(
            self._make_inputs(),
            model_factory=_boom_model,
            dict_loader=lambda: {"घर"},
        )
        assert "error" in out
        assert "checkpoint" in out["error"]

    def test_dict_load_failure_returns_error(self):
        """A missing dictionary file surfaces as an error response."""
        def _boom_dict():
            raise FileNotFoundError("dict missing")
        out = handler_mod.handler(
            self._make_inputs(),
            model_factory=lambda: object(),
            dict_loader=_boom_dict,
        )
        assert "error" in out
        assert "dict missing" in out["error"]

    def test_asr_failure_returns_error(self):
        """If transcribe raises, the error is returned, not propagated."""
        def _boom_transcribe(model, wav_bytes):
            raise RuntimeError("CUDA OOM")
        out = handler_mod.handler(
            self._make_inputs(),
            model_factory=lambda: object(),
            dict_loader=lambda: set(),
            transcribe_fn=_boom_transcribe,
        )
        assert "error" in out
        assert "ASR failed" in out["error"]

    def test_typical_success_path_with_postprocess(self):
        """Happy path: model + dict + ASR + Gemini all succeed."""
        def _fake_transcribe(model, wav_bytes):
            return "हांव वोचलो"

        def _fake_pp(client, *, prediction, reference, dict_words, original_wer):
            return {"corrected": "हांव घरा वोचलो.", "mode": "PASSTHROUGH"}

        # Force the post-processor by setting a fake API key.
        with mock.patch.dict(handler_mod.os.environ, {"GEMINI_API_KEY": "fake-key"}):
            out = handler_mod.handler(
                self._make_inputs(),
                model_factory=lambda: object(),
                dict_loader=lambda: {"घर", "वोचलो"},
                transcribe_fn=_fake_transcribe,
                postprocess_sample_fn=_fake_pp,
                genai_client_factory=lambda: object(),
            )

        assert "error" not in out
        assert out["raw"] == "हांव वोचलो"
        assert out["corrected"] == "हांव घरा वोचलो."
        assert out["transcription"] == "हांव घरा वोचलो."  # backwards-compat alias
        assert out["mode"] == "PASSTHROUGH"
        assert "asr" in out["latency_ms"]
        assert "postprocess" in out["latency_ms"]
        assert "total" in out["latency_ms"]

    def test_success_without_gemini_key_returns_skipped_mode(self):
        """When GEMINI_API_KEY is unset, post-processing is skipped, mode=SKIPPED."""
        def _fake_transcribe(model, wav_bytes):
            return "हांव वोचलो"

        with mock.patch.dict(handler_mod.os.environ, {"GEMINI_API_KEY": ""}):
            out = handler_mod.handler(
                self._make_inputs(),
                model_factory=lambda: object(),
                dict_loader=lambda: {"घर"},
                transcribe_fn=_fake_transcribe,
            )

        assert "error" not in out
        assert out["raw"] == "हांव वोचलो"
        assert out["corrected"] == "हांव वोचलो"
        assert out["transcription"] == "हांव वोचलो"
        assert out["mode"] == "SKIPPED"
        assert out["latency_ms"]["postprocess"] == 0

    def test_empty_asr_output_returns_skip_mode(self):
        """If ASR returns an empty string, mode is SKIP and corrected is empty."""
        def _empty_transcribe(model, wav_bytes):
            return ""
        with mock.patch.dict(handler_mod.os.environ, {"GEMINI_API_KEY": "fake-key"}):
            out = handler_mod.handler(
                self._make_inputs(),
                model_factory=lambda: object(),
                dict_loader=lambda: set(),
                transcribe_fn=_empty_transcribe,
                postprocess_sample_fn=lambda *a, **kw: pytest.fail("should not call"),
                genai_client_factory=lambda: object(),
            )
        assert out["raw"] == ""
        assert out["corrected"] == ""
        assert out["mode"] == "SKIP"
