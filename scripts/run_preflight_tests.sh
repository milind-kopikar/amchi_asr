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
# Ensure the repo root is first on PYTHONPATH so `import scripts` resolves to local files
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
pytest -q tests/test_unit_preflight.py tests/test_tokenizer_nemo_consistency.py tests/test_char_metric.py || { echo "Unit tests failed"; exit 3; }

# Allow micro-overfit to be run as part of preflight via RUN_MICRO_OVERFIT=1 or PREFLIGHT_RUN_MICRO=1
if [ -z "${RUN_MICRO_OVERFIT:-}" ] && [ "${PREFLIGHT_RUN_MICRO:-0}" = "1" ]; then
  export RUN_MICRO_OVERFIT=1
fi

if [ "${RUN_MICRO_OVERFIT:-0}" = "1" ]; then
  echo "RUN_MICRO_OVERFIT=1 detected — cleaning old artifacts and running micro overfit check (this may take a while)..."
  # Run cleanup (will fail if free space < PREFLIGHT_MIN_DISK_GB or MIN_GB)
  MIN_GB=${MIN_GB:-25}
  echo "Ensuring at least ${MIN_GB}GB free (cleanup may remove old artifacts)..."
  MIN_GB=${MIN_GB} ./scripts/cleanup_old_artifacts.sh || { echo "Cleanup failed or insufficient disk space"; exit 4; }
  # Allow quick runs via MAX_MICRO_EPOCHS env var (e.g., 3 for CI fast check)
  python3 scripts/run_micro_overfit.py || { echo "Micro-overfit check failed"; exit 5; }

  # After a successful micro-overfit, ask user whether to remove experiment outputs unless CLEANUP_AFTER_MICRO=1
  if [ "${CLEANUP_AFTER_MICRO:-0}" = "1" ]; then
    echo "CLEANUP_AFTER_MICRO=1 -> removing micro-overfit experiment outputs and checkpoints"
    rm -rf results/experiments/* results/checkpoints/* || true
  else
    # interactive prompt (skip in non-TTY or CI)
    if [ -t 0 ]; then
      read -p "Micro-overfit passed. Remove experiment outputs to save space? [y/N] " resp || true
      if [[ "${resp,,}" = "y" || "${resp,,}" = "yes" ]]; then
        echo "Removing experiment outputs and checkpoints..."
        rm -rf results/experiments/* results/checkpoints/* || true
      else
        echo "Keeping experiment outputs under results/experiments/"
      fi
    else
      echo "Micro-overfit passed. To auto-clean outputs set CLEANUP_AFTER_MICRO=1 before running tests."
    fi
  fi
fi

echo "All preflight checks and unit tests passed."