#!/usr/bin/env bash
set -euo pipefail

echo "Running full preflight checks and unit tests..."

# Activate canonical venv if present so tests run in the same env as provisioning
if [ -d "venv_py311" ]; then
  echo "Activating venv_py311 for reproducible test runs"
  # shellcheck disable=SC1091
  source venv_py311/bin/activate
fi

# Run preflight script
python3 scripts/preflight_checks.py || { echo "Preflight checks failed"; exit 2; }

# Run selected pytest tests (tokenizer and preflight units)
pytest -q tests/test_unit_preflight.py tests/test_tokenizer_nemo_consistency.py || { echo "Unit tests failed"; exit 3; }

if [ "${RUN_MICRO_OVERFIT:-0}" = "1" ]; then
  echo "RUN_MICRO_OVERFIT=1 detected — running micro overfit check (this may take a few minutes)..."
  python3 scripts/run_micro_overfit.py || { echo "Micro-overfit check failed"; exit 4; }
fi

echo "All preflight checks and unit tests passed."