"""Shared pytest setup for the unit-test suite.

Adds the repository root to ``sys.path`` so that ``from scripts.X import Y``
and ``from runpod import handler`` resolve in CI environments where the
project is not pip-installed and pytest is invoked without a wrapper.

``scripts/`` and ``runpod/`` have no ``__init__.py`` — they are implicit
namespace packages — so they are only importable when their parent
directory (the repo root) is on ``sys.path``. Locally this happens to
work because of editable installs or a pytest invocation from the repo
root; in the clean GitHub Actions runner it does not.

This file replaces the per-test ``sys.path.insert(...)`` boilerplate
that several test modules had been carrying. Leaving the per-file
blocks in place is harmless (they no-op when the path is already set).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
