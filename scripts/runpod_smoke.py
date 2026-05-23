#!/usr/bin/env python3
"""
RunPod container smoke test.

Runs INSIDE the Docker image at build time (and any time you want to
debug a misbehaving image with ``docker run --rm <image> python scripts/runpod_smoke.py``).

The smoke test is intentionally **staged** — each component is its own
check, and the script can either run them all in sequence (``--check all``)
or just one (``--check imports``). When a check fails, the script exits
non-zero with the specific component name in the error, which makes a
failed Docker build point at exactly which `RUN` command broke.

The script DOES NOT try to load the checkpoint, run inference, or call
Gemini. That all requires GPU / model / network and belongs in the
golden transcript tests (``tests/integration/``).

Usage
-----
Run all checks in order (default)::

    python scripts/runpod_smoke.py
    python scripts/runpod_smoke.py --check all

Run one specific check::

    python scripts/runpod_smoke.py --check imports
    python scripts/runpod_smoke.py --check patch
    python scripts/runpod_smoke.py --check dictionary
    python scripts/runpod_smoke.py --check handler

Select the handler variant under test (defaults to ``amchi``)::

    python scripts/runpod_smoke.py --variant amchi
    python scripts/runpod_smoke.py --variant deaf

Exit codes
----------
0  — every requested check passed
1  — at least one check failed (the script prints which one and why)
2  — invalid command-line arguments

Coverage
--------
- ``imports``    — can we ``import nemo``, ``import torch``, and the small
                   list of project-specific modules without error?
- ``patch``      — has the conv_asr patch been applied to NeMo? (compares
                   the installed file's checksum with the local patch).
- ``dictionary`` — for the amchi variant: is the dictionary JSON file
                   present and parseable into a non-empty set of words?
                   (skipped for the deaf variant.)
- ``handler``    — does the handler module import, and does the
                   ``handler()`` function return ``{error: ...}`` when
                   given an empty input (i.e. it short-circuits cleanly
                   without trying to load a model)?
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DICT_PATH = "/app/data/amchi_konkani_dict.json"
DEFAULT_DICT_FALLBACK = str(REPO_ROOT / "data" / "amchi_konkani_dict.json")

# Order matters: each check assumes the prior ones have already passed.
CHECK_ORDER = ("imports", "patch", "dictionary", "handler")

# Map variant → which handler module / what checks to skip.
#
# `handler_module` is the dotted name we'd ideally use for import_module.
# `handler_path` is the absolute filesystem path to the handler script,
# used by check_handler() when the dotted import is blocked by the
# `runpod` PyPI SDK shadowing the local `runpod/` directory (both want
# the top-level package name `runpod`, and the installed SDK wins because
# it's a regular package while the local directory is an implicit
# namespace package).
VARIANT_CONFIG = {
    "amchi": {
        "handler_module": "runpod.handler",
        "handler_path": str(REPO_ROOT / "runpod" / "handler.py"),
        "needs_dictionary": True,
    },
    "deaf": {
        "handler_module": "runpod.handler_deaf",
        "handler_path": str(REPO_ROOT / "runpod" / "handler_deaf.py"),
        "needs_dictionary": False,
    },
}


# ---------------------------------------------------------------------------
# Check implementations
# ---------------------------------------------------------------------------

def check_imports() -> None:
    """Verify the foundational Python imports succeed.

    Raises
    ------
    RuntimeError
        If any required import fails — the message names the missing module.
    """
    # The Docker image installs these via separate pip layers. Each one
    # being importable proves the corresponding layer was built correctly.
    required = [
        ("torch", "PyTorch (CUDA layer)"),
        ("nemo", "NeMo toolkit"),
        ("librosa", "librosa (audio loading)"),
        ("omegaconf", "OmegaConf (NeMo config)"),
        ("runpod", "RunPod SDK"),
    ]
    for name, label in required:
        try:
            importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - exercised in container
            raise RuntimeError(f"Failed to import {label} ({name!r}): {exc}") from exc


def check_patch() -> None:
    """Verify the conv_asr patch has been applied to NeMo.

    The patch is mandatory for loading fine-tuned hybrid CTC/RNNT
    checkpoints. Compares the SHA-256 of the patch file with the
    installed conv_asr module file.
    """
    import hashlib

    patch_path = REPO_ROOT / "patches" / "conv_asr_fixed.py"
    if not patch_path.is_file():
        raise RuntimeError(f"Patch source missing: {patch_path}")

    try:
        import nemo.collections.asr.modules.conv_asr as installed_mod
    except Exception as exc:
        raise RuntimeError(f"Cannot import nemo.collections.asr.modules.conv_asr: {exc}") from exc

    installed_path = Path(installed_mod.__file__)
    if not installed_path.is_file():
        raise RuntimeError(f"Installed NeMo conv_asr file missing: {installed_path}")

    src_hash = hashlib.sha256(patch_path.read_bytes()).hexdigest()
    installed_hash = hashlib.sha256(installed_path.read_bytes()).hexdigest()

    if src_hash != installed_hash:
        raise RuntimeError(
            f"conv_asr patch is NOT applied — installed file hash {installed_hash[:12]}… "
            f"does not match patch source hash {src_hash[:12]}…. "
            f"Run the patch step in the Dockerfile."
        )


def check_dictionary(dict_path: str | None = None) -> None:
    """Verify the Konkani dictionary JSON is present and parseable.

    Parameters
    ----------
    dict_path
        Explicit path. If None, tries ``AMCHI_DICT_PATH`` env, then
        the in-image default ``/app/data/amchi_konkani_dict.json``, then
        the local-dev fallback ``<repo>/data/amchi_konkani_dict.json``.
    """
    path = dict_path or os.environ.get("AMCHI_DICT_PATH")
    if not path:
        path = DEFAULT_DICT_PATH if Path(DEFAULT_DICT_PATH).is_file() else DEFAULT_DICT_FALLBACK

    if not Path(path).is_file():
        raise RuntimeError(
            f"Dictionary JSON not found at {path}. "
            f"Run 'python scripts/build_amchi_dict.py' before building the Docker image."
        )

    try:
        with open(path, encoding="utf-8") as fh:
            entries = json.load(fh)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse dictionary JSON {path}: {exc}") from exc

    if not isinstance(entries, list):
        raise RuntimeError(
            f"Dictionary JSON must be a list of entry-dicts; "
            f"got {type(entries).__name__}"
        )

    word_count = sum(
        1 for e in entries
        if isinstance(e, dict)
        and isinstance(e.get("word_konkani_devanagari"), str)
        and e["word_konkani_devanagari"].strip()
    )
    if word_count == 0:
        raise RuntimeError(
            f"Dictionary JSON has zero usable entries with a non-empty "
            f"'word_konkani_devanagari' field."
        )


def check_handler(variant: str = "amchi") -> None:
    """Verify the handler module loads and gracefully handles empty input.

    Calling ``handler({})`` should return a dict containing an ``"error"`` key,
    NOT raise an exception. This guarantees the handler's early-validation
    path is wired up correctly without needing a model or audio.

    The handler is loaded by absolute file path via
    ``importlib.util.spec_from_file_location`` rather than by dotted name —
    the dotted name ``runpod.handler_deaf`` collides with the installed
    ``runpod`` PyPI SDK (the SDK is a regular package and shadows our local
    namespace-package ``runpod/`` directory). The dotted name is preserved
    in the ``module_name`` argument to spec_from_file_location so any
    introspection / error messages still show ``runpod.handler_*``.
    """
    if variant not in VARIANT_CONFIG:
        raise RuntimeError(f"Unknown variant {variant!r}; expected one of {list(VARIANT_CONFIG)}")
    cfg = VARIANT_CONFIG[variant]
    module_name = cfg["handler_module"]
    handler_path = cfg["handler_path"]

    if not os.path.isfile(handler_path):
        raise RuntimeError(
            f"Handler source file not found at {handler_path}. "
            "Was the repo COPYed into /app correctly?"
        )

    import importlib.util  # local import — only needed in this check
    try:
        spec = importlib.util.spec_from_file_location(module_name, handler_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"spec_from_file_location returned None for {handler_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {module_name} from {handler_path}: {exc}"
        ) from exc

    if not hasattr(mod, "handler") or not callable(mod.handler):
        raise RuntimeError(f"{module_name} does not export a callable 'handler' function")

    try:
        result = mod.handler({"input": {}})
    except Exception as exc:
        raise RuntimeError(
            f"Handler raised an exception on empty input; should have returned "
            f"an error-dict instead: {exc!r}"
        ) from exc

    if not isinstance(result, dict):
        raise RuntimeError(
            f"Handler returned {type(result).__name__}, expected dict"
        )
    if "error" not in result:
        raise RuntimeError(
            f"Handler did not produce an 'error' for empty input; got {result!r}"
        )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def get_check_function(name: str, variant: str) -> Callable[[], None]:
    """Return a zero-arg callable for the named check + variant."""
    if name == "imports":
        return check_imports
    if name == "patch":
        return check_patch
    if name == "dictionary":
        if not VARIANT_CONFIG[variant]["needs_dictionary"]:
            # The deaf variant doesn't use the dictionary; turn this into a no-op.
            return lambda: None
        return check_dictionary
    if name == "handler":
        return lambda: check_handler(variant)
    raise ValueError(f"Unknown check: {name!r}")


def run_checks(checks: list[str], variant: str, *, verbose: bool = True) -> int:
    """Run the named checks in order. Returns 0 on full success, 1 otherwise."""
    failures: list[tuple[str, str]] = []
    total = len(checks)
    for index, name in enumerate(checks, start=1):
        if verbose:
            print(f"[{index}/{total}] {name:11s} ", end="", flush=True)
        try:
            get_check_function(name, variant)()
        except Exception as exc:
            failures.append((name, str(exc)))
            if verbose:
                print(f"FAIL: {exc}")
        else:
            if verbose:
                print("PASS")

    if verbose and failures:
        print()
        print("Summary:")
        for name, msg in failures:
            print(f"  [{name}] {msg}")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RunPod container smoke test — staged component checks.",
    )
    parser.add_argument(
        "--check",
        choices=("all", *CHECK_ORDER),
        default="all",
        help="Which check to run (default: all).",
    )
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANT_CONFIG),
        default="amchi",
        help="Which handler variant the image embeds.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-check PASS/FAIL output (still exits with proper code).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    checks = list(CHECK_ORDER) if args.check == "all" else [args.check]
    return run_checks(checks, args.variant, verbose=not args.quiet)


if __name__ == "__main__":
    sys.exit(main())
