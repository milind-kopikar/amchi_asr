"""HTTP helpers used by RunPod handlers and the local inference verifier.

Why this exists
---------------
Cloudflare's anti-bot protection on the R2 public domain (``*.r2.dev``)
rejects requests whose User-Agent is the default ``Python-urllib/3.x``
(HTTP 403). The RunPod workers and the local Colab verifier both need to
download multi-hundred-MB checkpoints from R2, so every urllib request to
external hosts must carry a sensible User-Agent header.

This module centralises that in three small functions so the same fix is
not duplicated across both handlers and the verification scripts.

Inputs / outputs (test-friendly contracts)
------------------------------------------
``open_url(url, *, timeout=30, user_agent=DEFAULT, opener=None) -> response context manager``
``download_url_to_path(url, dest, *, timeout=600, chunk_size=1<<20, ...) -> int (bytes downloaded)``
``fetch_url_bytes(url, *, timeout=30, ...) -> bytes``

Every function accepts an ``opener`` test seam — a callable with the same
signature as ``urllib.request.urlopen``. Tests inject a fake opener that
returns a context manager wrapping pre-built bytes; no real HTTP happens.

Failure modes
-------------
The helpers do NOT swallow exceptions. Callers should catch
``urllib.error.HTTPError`` and ``urllib.error.URLError`` (or just
``Exception``) and decide what to do — return an error response,
retry, fail the build, etc. — based on context.
"""

from __future__ import annotations

import urllib.request
from typing import Callable, Optional

# Default UA. Identifies the project + a reachable repo URL so a server
# admin who sees suspicious traffic can find us. ``+url`` is the convention
# used by major crawlers (Googlebot, etc.).
DEFAULT_USER_AGENT = "konkani-asr/1.0 (+https://github.com/milind-kopikar/amchi_asr)"


def open_url(
    url: str,
    *,
    timeout: int = 30,
    user_agent: str = DEFAULT_USER_AGENT,
    opener: Optional[Callable] = None,
):
    """Open a URL with a custom User-Agent header.

    Parameters
    ----------
    url
        HTTP(S) URL.
    timeout
        Per-request timeout in seconds.
    user_agent
        Sent as the ``User-Agent`` request header. Defaults to the project UA.
    opener
        Test seam — a callable matching ``urllib.request.urlopen``'s signature
        ``(Request, timeout=...)``. Defaults to ``urllib.request.urlopen``.

    Returns
    -------
    Response context manager — use as ``with open_url(...) as resp: ...``.

    Raises
    ------
    urllib.error.HTTPError, urllib.error.URLError
        On HTTP errors / DNS / connection failures. Caller decides.
    """
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    fn = opener if opener is not None else urllib.request.urlopen
    return fn(request, timeout=timeout)


def download_url_to_path(
    url: str,
    dest_path: str,
    *,
    timeout: int = 600,
    chunk_size: int = 1 << 20,  # 1 MiB
    user_agent: str = DEFAULT_USER_AGENT,
    opener: Optional[Callable] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> int:
    """Stream a URL to a local file with a custom User-Agent.

    Suitable for multi-hundred-MB checkpoint downloads (memory-efficient —
    streams in ``chunk_size`` chunks, never builds the whole thing in RAM).

    Parameters
    ----------
    url
        HTTP(S) URL to download.
    dest_path
        Local filesystem path to write to. Created/overwritten.
    timeout
        Per-connection timeout. Default 600s (10 min) for large checkpoints.
    chunk_size
        Streaming read buffer size. Default 1 MiB.
    user_agent
        Header value.
    opener
        Test seam.
    progress_callback
        Optional callable ``(bytes_so_far, total_bytes)`` invoked after each
        chunk. ``total_bytes`` is 0 if the server didn't send Content-Length.

    Returns
    -------
    int
        Total bytes downloaded.

    Raises
    ------
    urllib.error.HTTPError, urllib.error.URLError, OSError
    """
    with open_url(url, timeout=timeout, user_agent=user_agent, opener=opener) as response:
        total = int(response.headers.get("Content-Length", "0") or 0)
        downloaded = 0
        with open(dest_path, "wb") as fh:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                if progress_callback is not None:
                    progress_callback(downloaded, total)
    return downloaded


def fetch_url_bytes(
    url: str,
    *,
    timeout: int = 30,
    user_agent: str = DEFAULT_USER_AGENT,
    opener: Optional[Callable] = None,
) -> bytes:
    """Fetch a URL and return all bytes in memory.

    Use for small payloads (audio files under ~20 MB, JSON, manifests). For
    large checkpoints prefer ``download_url_to_path`` which streams.

    Parameters
    ----------
    url
        HTTP(S) URL.
    timeout
        Per-request timeout in seconds.
    user_agent
        Header value.
    opener
        Test seam.

    Returns
    -------
    bytes
        Full response body.

    Raises
    ------
    urllib.error.HTTPError, urllib.error.URLError
    """
    with open_url(url, timeout=timeout, user_agent=user_agent, opener=opener) as response:
        return response.read()
