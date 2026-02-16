import os
import subprocess
import sys
import pytest


@pytest.mark.skipif(os.environ.get("RUN_MICRO_OVERFIT", "0") != "1", reason="Micro-overfit is opt-in and skipped by default")
def test_micro_overfit_runs_and_passes():
    """Runs the micro overfit script and expects it to exit 0 (pass)"""
    ret = subprocess.run([sys.executable, "scripts/run_micro_overfit.py"], check=False)
    assert ret.returncode == 0, f"micro-overfit script failed with exit {ret.returncode}"
