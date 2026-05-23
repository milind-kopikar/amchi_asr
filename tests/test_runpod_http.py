"""Unit tests for ``scripts/runpod_http.py``.

All HTTP calls are mocked via the ``opener`` test seam. No real network is
touched. Verifies:

  - The User-Agent header is actually set on every Request
  - Streaming download writes the expected bytes to disk
  - Errors propagate (the helpers do not swallow them)
  - The progress callback fires for each chunk
  - Defaults are sensible

Run from the repo root::

    pytest tests/test_runpod_http.py -v
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import pytest

from scripts.runpod_http import (
    DEFAULT_USER_AGENT,
    download_url_to_path,
    fetch_url_bytes,
    open_url,
)


# ---------------------------------------------------------------------------
# Helpers — build fake openers
# ---------------------------------------------------------------------------

class _FakeResponse:
    """Context-manager response that yields the given body in chunks."""

    def __init__(self, body: bytes, *, content_length: int | None = None):
        self._body = body
        self._read_offset = 0
        if content_length is None:
            content_length = len(body)
        self.headers = {"Content-Length": str(content_length)} if content_length >= 0 else {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size: int | None = None) -> bytes:
        remaining = self._body[self._read_offset:]
        if size is None or size <= 0:
            self._read_offset = len(self._body)
            return remaining
        chunk = remaining[:size]
        self._read_offset += len(chunk)
        return chunk


def _opener_returning(body: bytes, *, captured: dict | None = None,
                     content_length: int | None = None):
    """Build a fake opener that returns ``body`` and optionally captures the request."""

    def fn(request, timeout):
        if captured is not None:
            captured["url"] = request.full_url
            captured["headers"] = dict(request.headers)
            captured["timeout"] = timeout
        return _FakeResponse(body, content_length=content_length)

    return fn


def _opener_raising(exc: Exception):
    def fn(request, timeout):
        raise exc
    return fn


# ---------------------------------------------------------------------------
# open_url
# ---------------------------------------------------------------------------

class TestOpenUrl:
    """Verifies the User-Agent injection and pass-through behaviour."""

    def test_default_user_agent_is_set(self):
        captured: dict = {}
        with open_url("http://x.test/file", opener=_opener_returning(b"hi", captured=captured)) as r:
            r.read()
        # urllib normalises header names to title-case: "User-agent"
        assert captured["headers"].get("User-agent") == DEFAULT_USER_AGENT \
            or captured["headers"].get("User-Agent") == DEFAULT_USER_AGENT

    def test_custom_user_agent(self):
        captured: dict = {}
        with open_url("http://x.test/file", user_agent="custom/9.9",
                     opener=_opener_returning(b"hi", captured=captured)):
            pass
        assert "custom/9.9" in captured["headers"].values()

    def test_timeout_is_forwarded(self):
        captured: dict = {}
        with open_url("http://x.test/file", timeout=42,
                     opener=_opener_returning(b"hi", captured=captured)):
            pass
        assert captured["timeout"] == 42

    def test_url_is_forwarded(self):
        captured: dict = {}
        with open_url("http://example.test/path?q=1",
                     opener=_opener_returning(b"hi", captured=captured)):
            pass
        assert captured["url"] == "http://example.test/path?q=1"

    def test_http_error_propagates(self):
        err = urllib.error.HTTPError("http://x", 403, "Forbidden", hdrs={}, fp=None)
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            with open_url("http://x.test/file", opener=_opener_raising(err)):
                pass
        assert exc_info.value.code == 403

    def test_url_error_propagates(self):
        err = urllib.error.URLError("network down")
        with pytest.raises(urllib.error.URLError):
            with open_url("http://x.test/file", opener=_opener_raising(err)):
                pass

    def test_default_opener_is_urlopen(self):
        """When opener=None, the function defers to urllib.request.urlopen."""
        with mock.patch.object(
            urllib.request, "urlopen",
            return_value=_FakeResponse(b"hi"),
        ) as mocked:
            with open_url("http://x.test/file") as r:
                r.read()
        assert mocked.called


# ---------------------------------------------------------------------------
# download_url_to_path
# ---------------------------------------------------------------------------

class TestDownloadUrlToPath:
    """Streaming download to a file."""

    def test_typical_small_file(self, tmp_path):
        payload = b"PK" + b"\x00" * 999  # mimic a small .ckpt magic prefix
        dest = tmp_path / "out.ckpt"
        n = download_url_to_path(
            "http://x.test/big.ckpt", str(dest),
            opener=_opener_returning(payload),
        )
        assert n == len(payload)
        assert dest.read_bytes() == payload

    def test_multi_chunk_streaming(self, tmp_path):
        """A payload larger than chunk_size still ends up correct on disk."""
        payload = b"X" * 100_000  # 100 KB
        dest = tmp_path / "out.bin"
        n = download_url_to_path(
            "http://x.test/big.bin", str(dest),
            chunk_size=4096,
            opener=_opener_returning(payload),
        )
        assert n == 100_000
        assert dest.read_bytes() == payload

    def test_progress_callback_fires_per_chunk(self, tmp_path):
        payload = b"Y" * 10_000
        dest = tmp_path / "out.bin"
        calls: list[tuple[int, int]] = []

        def cb(downloaded: int, total: int) -> None:
            calls.append((downloaded, total))

        download_url_to_path(
            "http://x.test/big.bin", str(dest),
            chunk_size=1024,
            opener=_opener_returning(payload),
            progress_callback=cb,
        )
        # With 10 KB payload and 1 KiB chunk size, expect ~10 callbacks.
        assert len(calls) >= 9
        # The last call's downloaded count must equal payload length.
        assert calls[-1][0] == 10_000
        # Total is reported from Content-Length.
        assert calls[-1][1] == 10_000

    def test_missing_content_length_uses_zero(self, tmp_path):
        payload = b"abc"
        dest = tmp_path / "out.bin"
        calls = []
        download_url_to_path(
            "http://x.test/file", str(dest),
            opener=_opener_returning(payload, content_length=-1),  # triggers our "no header" branch
            progress_callback=lambda d, t: calls.append((d, t)),
        )
        # When Content-Length missing, total is 0 (caller can detect "unknown size")
        assert calls[-1][1] == 0

    def test_empty_response_writes_zero_bytes(self, tmp_path):
        dest = tmp_path / "empty.bin"
        n = download_url_to_path(
            "http://x.test/empty", str(dest),
            opener=_opener_returning(b""),
        )
        assert n == 0
        assert dest.read_bytes() == b""

    def test_user_agent_is_set(self, tmp_path):
        captured: dict = {}
        dest = tmp_path / "out.bin"
        download_url_to_path(
            "http://x.test/big.bin", str(dest),
            opener=_opener_returning(b"hi", captured=captured),
        )
        # urllib title-cases header keys
        ua = captured["headers"].get("User-agent") or captured["headers"].get("User-Agent")
        assert ua == DEFAULT_USER_AGENT

    def test_network_error_propagates(self, tmp_path):
        dest = tmp_path / "out.bin"
        with pytest.raises(urllib.error.URLError):
            download_url_to_path(
                "http://x.test/file", str(dest),
                opener=_opener_raising(urllib.error.URLError("dns")),
            )

    def test_http_error_propagates(self, tmp_path):
        dest = tmp_path / "out.bin"
        err = urllib.error.HTTPError("http://x", 403, "Forbidden", hdrs={}, fp=None)
        with pytest.raises(urllib.error.HTTPError):
            download_url_to_path(
                "http://x.test/file", str(dest),
                opener=_opener_raising(err),
            )


# ---------------------------------------------------------------------------
# fetch_url_bytes
# ---------------------------------------------------------------------------

class TestFetchUrlBytes:
    """In-memory fetcher for small payloads."""

    def test_typical(self):
        body = b'{"key": "value"}'
        out = fetch_url_bytes("http://x.test/api", opener=_opener_returning(body))
        assert out == body

    def test_empty(self):
        out = fetch_url_bytes("http://x.test/api", opener=_opener_returning(b""))
        assert out == b""

    def test_user_agent_is_set(self):
        captured: dict = {}
        fetch_url_bytes(
            "http://x.test/api",
            opener=_opener_returning(b"ok", captured=captured),
        )
        ua = captured["headers"].get("User-agent") or captured["headers"].get("User-Agent")
        assert ua == DEFAULT_USER_AGENT

    def test_custom_user_agent_overrides_default(self):
        captured: dict = {}
        fetch_url_bytes(
            "http://x.test/api",
            user_agent="custom-ua",
            opener=_opener_returning(b"ok", captured=captured),
        )
        assert "custom-ua" in captured["headers"].values()

    def test_http_error_propagates(self):
        err = urllib.error.HTTPError("http://x", 404, "Not Found", hdrs={}, fp=None)
        with pytest.raises(urllib.error.HTTPError):
            fetch_url_bytes("http://x.test/api", opener=_opener_raising(err))


# ---------------------------------------------------------------------------
# Cross-cutting: default UA constant
# ---------------------------------------------------------------------------

def test_default_user_agent_format():
    """The default UA should identify the project and contain a +URL."""
    assert "konkani-asr" in DEFAULT_USER_AGENT
    assert "+http" in DEFAULT_USER_AGENT  # well-known +url convention
