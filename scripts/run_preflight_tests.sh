#!/usr/bin/env bash
set -euo pipefail

echo "Running full preflight checks and unit tests..."

# Run preflight script
python3 scripts/preflight_checks.py || { echo "Preflight checks failed"; exit 2; }

# Run selected pytest tests (tokenizer and preflight units)
pytest -q tests/test_unit_preflight.py tests/test_tokenizer_nemo_consistency.py || { echo "Unit tests failed"; exit 3; }

echo "All preflight checks and unit tests passed."