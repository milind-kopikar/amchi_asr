#!/usr/bin/env bash
set -euo pipefail

echo "Running full preflight checks and unit tests..."

# Activate canonical venv if present so tests run in the same env as provisioning
if [ -d "venv_py311" ]; then
  echo "Activating venv_py311 for reproducible test runs"
  # shellcheck disable=SC1091
  source venv_py311/bin/activate
fi

# Ensure our conv_asr runtime patch env var is set so preflight verifies the patched codepath
export APPLY_CONV_PATCH=${APPLY_CONV_PATCH:-1}

# Run preflight script
python3 scripts/preflight_checks.py || { echo "Preflight checks failed"; exit 2; }

# Run selected pytest tests (tokenizer and preflight units)
# Ensure the repo root is on PYTHONPATH so `import scripts` works during tests
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
pytest -q tests/test_unit_preflight.py tests/test_tokenizer_nemo_consistency.py || { echo "Unit tests failed"; exit 3; }

if [ "${RUN_MICRO_OVERFIT:-0}" = "1" ]; then
  echo "RUN_MICRO_OVERFIT=1 detected — cleaning old artifacts and running micro overfit check (this may take a while)..."
  # Run cleanup (will fail if free space < PREFLIGHT_MIN_DISK_GB or MIN_GB)
  MIN_GB=${MIN_GB:-25}
  echo "Ensuring at least ${MIN_GB}GB free (cleanup may remove old artifacts)..."
  MIN_GB=${MIN_GB} ./scripts/cleanup_old_artifacts.sh || { echo "Cleanup failed or insufficient disk space"; exit 4; }
  python3 scripts/run_micro_overfit.py || { echo "Micro-overfit check failed"; exit 5; }
fi

echo "All preflight checks and unit tests passed."