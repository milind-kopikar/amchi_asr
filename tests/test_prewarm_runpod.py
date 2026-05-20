"""Unit tests for ``scripts/prewarm_runpod.py``.

Covers the pure helpers (silence WAV generation, base64 encoding) plus the
network-mocked endpoint call and the CLI orchestrator's exit codes. No real
HTTP requests are made.

Run from the repo root::

    pytest tests/test_prewarm_runpod.py -v
"""

from __future__ import annotations

import json
import struct
import urllib.error
from unittest import mock

import pytest

from scripts.prewarm_runpod import (
    DEFAULT_TIMEOUT_SECONDS,
    SAMPLE_RATE_HZ,
    encode_audio_for_runpod,
    generate_silence_wav,
    main,
    prewarm_endpoint,
    prewarm_selected,
)


# ---------------------------------------------------------------------------
# generate_silence_wav — pure
# ---------------------------------------------------------------------------

class TestGenerateSilenceWav:
    """Verify the WAV header is well-formed and the data section is silence."""

    def test_typical_output_is_valid_wav(self):
        """A 1-second 16 kHz mono WAV has 32044 bytes (header 44 + 2*16000 data)."""
        wav = generate_silence_wav(1.0, SAMPLE_RATE_HZ)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        assert wav[12:16] == b"fmt "
        # PCM fmt chunk = 16 bytes; 4+4+4+4+4+2+2+4+4+2+2+4+4 = 44 header total
        expected_data_size = SAMPLE_RATE_HZ * 2  # 16-bit mono
        assert len(wav) == 44 + expected_data_size

    def test_data_section_is_all_zeros(self):
        """Silence means every sample byte is 0x00."""
        wav = generate_silence_wav(0.5)
        # data starts after the 44-byte PCM header
        data = wav[44:]
        assert data == b"\x00" * len(data)

    def test_sample_rate_in_header(self):
        """The sample rate field is little-endian uint32 at offset 24."""
        wav = generate_silence_wav(0.5, sample_rate=16000)
        sr = struct.unpack("<I", wav[24:28])[0]
        assert sr == 16000

    def test_channels_and_bits_per_sample(self):
        """Mono (channels=1) and 16-bit per sample."""
        wav = generate_silence_wav(0.5)
        channels = struct.unpack("<H", wav[22:24])[0]
        bits = struct.unpack("<H", wav[34:36])[0]
        assert channels == 1
        assert bits == 16

    def test_zero_duration_rejected(self):
        """Zero or negative durations raise ValueError."""
        with pytest.raises(ValueError, match="duration_seconds must be > 0"):
            generate_silence_wav(0)
        with pytest.raises(ValueError, match="duration_seconds must be > 0"):
            generate_silence_wav(-1.0)

    def test_zero_sample_rate_rejected(self):
        """Zero or negative sample rate raises ValueError."""
        with pytest.raises(ValueError, match="sample_rate must be > 0"):
            generate_silence_wav(0.5, sample_rate=0)


# ---------------------------------------------------------------------------
# encode_audio_for_runpod — pure
# ---------------------------------------------------------------------------

class TestEncodeAudioForRunpod:
    """Base64 round-trip + type-safety."""

    def test_typical_round_trip(self):
        """Encoded → decoded matches the original bytes."""
        import base64
        sample = b"\x00\x01\x02\xfe\xff"
        encoded = encode_audio_for_runpod(sample)
        assert base64.b64decode(encoded) == sample

    def test_empty_bytes(self):
        """Empty input is allowed and round-trips to empty string."""
        assert encode_audio_for_runpod(b"") == ""

    def test_bytearray_accepted(self):
        """bytearray (mutable) inputs are accepted, not only bytes."""
        encoded = encode_audio_for_runpod(bytearray(b"hello"))
        import base64
        assert base64.b64decode(encoded) == b"hello"

    def test_string_rejected(self):
        """str input is rejected (would silently decode incorrectly)."""
        with pytest.raises(TypeError, match="must be bytes-like"):
            encode_audio_for_runpod("not bytes")  # type: ignore[arg-type]

    def test_none_rejected(self):
        """None input is rejected."""
        with pytest.raises(TypeError):
            encode_audio_for_runpod(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# prewarm_endpoint — mocked HTTP
# ---------------------------------------------------------------------------

def _opener_returning(body: dict | str, *, http_status: int = 200):
    """Build a fake opener that returns ``body`` as the response payload."""
    raw = json.dumps(body) if isinstance(body, dict) else body

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return raw.encode("utf-8")

    return mock.MagicMock(return_value=_Resp())


class TestPrewarmEndpoint:
    """Network-mocked tests for the per-endpoint prewarm call."""

    def test_typical_success(self):
        """A COMPLETED RunPod response is reported as ok=True."""
        opener = _opener_returning({"status": "COMPLETED", "output": {"raw": ""}})
        result = prewarm_endpoint("key", "endpoint-123", "QUFB", opener=opener)
        assert result["ok"] is True
        assert "elapsed_seconds" in result
        assert result["response"]["status"] == "COMPLETED"

    def test_missing_api_key(self):
        """An empty api_key is rejected without a network call."""
        # Use an opener that would fail if called.
        opener = mock.MagicMock(side_effect=AssertionError("should not call"))
        result = prewarm_endpoint("", "endpoint-123", "QUFB", opener=opener)
        assert result["ok"] is False
        assert "api_key" in result["error"]

    def test_missing_endpoint_id(self):
        """An empty endpoint_id is rejected without a network call."""
        opener = mock.MagicMock(side_effect=AssertionError("should not call"))
        result = prewarm_endpoint("key", "", "QUFB", opener=opener)
        assert result["ok"] is False
        assert "endpoint_id" in result["error"]

    def test_http_error(self):
        """4xx/5xx HTTP errors are reported with the status code."""
        def _raises(*a, **kw):
            raise urllib.error.HTTPError(
                "http://x", 401, "Unauthorized", hdrs={}, fp=None
            )
        result = prewarm_endpoint("key", "endpoint-123", "QUFB", opener=_raises)
        assert result["ok"] is False
        assert "HTTP 401" in result["error"]

    def test_network_error(self):
        """DNS / TCP errors are reported."""
        def _raises(*a, **kw):
            raise urllib.error.URLError("connection refused")
        result = prewarm_endpoint("key", "endpoint-123", "QUFB", opener=_raises)
        assert result["ok"] is False
        assert "Network error" in result["error"]

    def test_runpod_non_completed_status(self):
        """If RunPod returns status=FAILED, we mark it as failed."""
        opener = _opener_returning({"status": "FAILED", "error": "OOM"})
        result = prewarm_endpoint("key", "endpoint-123", "QUFB", opener=opener)
        assert result["ok"] is False
        assert "FAILED" in result["error"]

    def test_non_json_response(self):
        """A non-JSON response is reported as malformed."""
        opener = _opener_returning("not json at all")
        result = prewarm_endpoint("key", "endpoint-123", "QUFB", opener=opener)
        assert result["ok"] is False
        assert "Non-JSON response" in result["error"]

    def test_request_body_shape(self):
        """The Request body is JSON containing audio_base64 under 'input'."""
        captured = {}

        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *exc): return False
            def read(self): return b'{"status":"COMPLETED"}'

        def _opener(request, timeout):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["auth"] = request.headers.get("Authorization")
            return _Resp()

        prewarm_endpoint("my-key", "endpoint-xyz", "QUFB", opener=_opener)
        assert captured["url"].endswith("/endpoint-xyz/runsync")
        assert captured["body"]["input"]["audio_base64"] == "QUFB"
        assert captured["auth"] == "Bearer my-key"


# ---------------------------------------------------------------------------
# prewarm_selected — multi-endpoint orchestration
# ---------------------------------------------------------------------------

class TestPrewarmSelected:
    """Verifies the multi-endpoint orchestrator."""

    def test_both_endpoints(self):
        """When both IDs are given, both are pre-warmed."""
        opener = _opener_returning({"status": "COMPLETED"})
        results = prewarm_selected("key", "amchi-id", "deaf-id", opener=opener)
        assert set(results.keys()) == {"amchi", "deaf"}
        assert results["amchi"]["ok"] is True
        assert results["deaf"]["ok"] is True

    def test_only_amchi(self):
        """When deaf_endpoint_id is None, only amchi is in the result."""
        opener = _opener_returning({"status": "COMPLETED"})
        results = prewarm_selected("key", "amchi-id", None, opener=opener)
        assert "amchi" in results
        assert "deaf" not in results

    def test_only_deaf(self):
        """Symmetric to test_only_amchi."""
        opener = _opener_returning({"status": "COMPLETED"})
        results = prewarm_selected("key", None, "deaf-id", opener=opener)
        assert "deaf" in results
        assert "amchi" not in results

    def test_neither(self):
        """If both IDs are None, the result dict is empty (caller checks)."""
        opener = mock.MagicMock(side_effect=AssertionError("should not call"))
        results = prewarm_selected("key", None, None, opener=opener)
        assert results == {}


# ---------------------------------------------------------------------------
# main — CLI exit codes
# ---------------------------------------------------------------------------

class TestMain:
    """Exit-code contract for the CLI entry point."""

    def test_missing_api_key_returns_1(self, monkeypatch, capsys):
        """No api key (env or arg) → exit 1."""
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        rc = main(["--amchi-endpoint", "amchi-id"])
        assert rc == 1
        assert "Missing RUNPOD_API_KEY" in capsys.readouterr().err

    def test_missing_endpoint_returns_1(self, monkeypatch, capsys):
        """No endpoint IDs at all → exit 1."""
        monkeypatch.setenv("RUNPOD_API_KEY", "key")
        monkeypatch.delenv("RUNPOD_AMCHI_ENDPOINT_ID", raising=False)
        monkeypatch.delenv("RUNPOD_DEAF_ENDPOINT_ID", raising=False)
        rc = main([])
        assert rc == 1
        assert "endpoint ID is required" in capsys.readouterr().err

    def test_both_endpoints_succeed_returns_0(self, monkeypatch, capsys):
        """All endpoints OK → exit 0."""
        monkeypatch.setenv("RUNPOD_API_KEY", "key")
        opener = _opener_returning({"status": "COMPLETED"})
        rc = main(
            ["--amchi-endpoint", "a", "--deaf-endpoint", "d"],
            opener=opener,
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "amchi endpoint warmed" in out
        assert "deaf endpoint warmed" in out

    def test_one_endpoint_fails_returns_2(self, monkeypatch):
        """Any failure → exit 2."""
        monkeypatch.setenv("RUNPOD_API_KEY", "key")

        def _opener(request, timeout):
            if "amchi-id" in request.full_url:
                raise urllib.error.URLError("amchi down")
            class _OK:
                def __enter__(self): return self
                def __exit__(self, *exc): return False
                def read(self): return b'{"status":"COMPLETED"}'
            return _OK()

        rc = main(
            ["--amchi-endpoint", "amchi-id", "--deaf-endpoint", "deaf-id"],
            opener=_opener,
        )
        assert rc == 2

    def test_only_amchi_filter(self, monkeypatch, capsys):
        """--only amchi pre-warms only the amchi endpoint."""
        monkeypatch.setenv("RUNPOD_API_KEY", "key")
        monkeypatch.setenv("RUNPOD_DEAF_ENDPOINT_ID", "deaf-id")
        opener = _opener_returning({"status": "COMPLETED"})
        rc = main(
            ["--only", "amchi", "--amchi-endpoint", "amchi-id"],
            opener=opener,
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "amchi endpoint warmed" in out
        assert "deaf endpoint" not in out
