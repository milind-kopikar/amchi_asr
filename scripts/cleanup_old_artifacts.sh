#!/usr/bin/env bash
set -euo pipefail

# Cleanup old experiments and checkpoints to free disk space.
# Default: keep 2 newest experiment dirs and 5 newest checkpoints.
KEEP_EXPS=${KEEP_EXPS:-2}
KEEP_CKPTS=${KEEP_CKPTS:-5}

echo "Cleanup: keeping ${KEEP_EXPS} newest experiments and ${KEEP_CKPTS} newest checkpoints"

# Remove old experiment dirs
if [ -d results/experiments ]; then
  cd results/experiments
  to_remove=$(ls -1t | tail -n +$((KEEP_EXPS+1)) || true)
  if [ -n "$to_remove" ]; then
    echo "Removing experiments: $to_remove"
    echo "$to_remove" | xargs -r -I{} rm -rf {}
  else
    echo "No old experiments to remove"
  fi
  cd - >/dev/null
fi

# Remove old checkpoint files
if [ -d results/checkpoints ]; then
  cd results/checkpoints
  to_remove=$(ls -1t | tail -n +$((KEEP_CKPTS+1)) || true)
  if [ -n "$to_remove" ]; then
    echo "Removing checkpoints: $to_remove"
    echo "$to_remove" | xargs -r -I{} rm -f {}
  else
    echo "No old checkpoints to remove"
  fi
  cd - >/dev/null
fi

# Print disk free and compare to minimum
MIN_GB=${MIN_GB:-25}
free_gb=$(python3 - <<'PY'
import shutil,sys
free=shutil.disk_usage('.').free/(1024**3)
print(int(free))
PY
)

echo "Disk free after cleanup: ${free_gb} GB"
if [ "$free_gb" -lt "$MIN_GB" ]; then
  echo "ERROR: free space ${free_gb}GB is less than required ${MIN_GB}GB" >&2
  exit 2
fi

echo "Cleanup completed; ${free_gb} GB available."