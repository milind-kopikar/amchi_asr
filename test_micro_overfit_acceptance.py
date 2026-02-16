import os
import sys
import subprocess
import json
from pathlib import Path
import pytest

PY = sys.executable

@pytest.fixture(autouse=True)
def clean_results(tmp_path, monkeypatch):
    # Use a temporary results dir to avoid stomping repository results
    repo_root = Path.cwd()
    tmp_results = tmp_path / 'results'
    tmp_results.mkdir()
    monkeypatch.chdir(repo_root)
    # Redirect RESULTS via env var used by script (it uses relative paths so we change CWD)
    yield


@pytest.mark.skipif(os.environ.get("RUN_MICRO_OVERFIT", "0") != "1", reason="Micro-overfit is opt-in and skipped by default")
def test_micro_overfit_acceptance_pass(tmp_path):
    # Create fake experiment with final_test_results.json and epoch_metrics.csv that satisfy pass criteria
    exp_dir = Path('results') / 'experiments' / 'fake_exp_pass'
    exp_dir.mkdir(parents=True, exist_ok=True)
    final = exp_dir / 'final_test_results.json'
    final.write_text(json.dumps({"per_sample": [{"reference": "hello world", "prediction": "hello world"}]}), encoding='utf-8')
    csvp = exp_dir / 'epoch_metrics.csv'
    csvp.write_text('\n'.join([
        'epoch,train_loss,val_loss',
        '1,100,50',
        '5,40,20',
    ]), encoding='utf-8')

    env = os.environ.copy()
    env.update({'RUN_MICRO_OVERFIT': '1', 'SKIP_MICRO_PREFLIGHT': '1', 'SKIP_MICRO_TRAIN': '1'})
    ret = subprocess.run([PY, 'scripts/run_micro_overfit.py'], env=env, check=False)
    assert ret.returncode == 0, f"expected script to pass, got exit {ret.returncode}"


@pytest.mark.skipif(os.environ.get("RUN_MICRO_OVERFIT", "0") != "1", reason="Micro-overfit is opt-in and skipped by default")
def test_micro_overfit_acceptance_fail(tmp_path):
    # Create fake experiment that does NOT meet pass criteria
    exp_dir = Path('results') / 'experiments' / 'fake_exp_fail'
    exp_dir.mkdir(parents=True, exist_ok=True)
    final = exp_dir / 'final_test_results.json'
    final.write_text(json.dumps({"per_sample": [{"reference": "abc", "prediction": "xyz"}]}), encoding='utf-8')
    csvp = exp_dir / 'epoch_metrics.csv'
    csvp.write_text('\n'.join([
        'epoch,train_loss,val_loss',
        '1,1.0,0.9',
        '2,0.9,0.8',
    ]), encoding='utf-8')

    env = os.environ.copy()
    env.update({'RUN_MICRO_OVERFIT': '1', 'SKIP_MICRO_PREFLIGHT': '1', 'SKIP_MICRO_TRAIN': '1'})
    ret = subprocess.run([PY, 'scripts/run_micro_overfit.py'], env=env, check=False)
    # Expect code 7 (neither criterion met)
    assert ret.returncode == 7, f"expected script to fail with exit 7, got {ret.returncode}"
