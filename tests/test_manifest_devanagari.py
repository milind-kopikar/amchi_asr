import pytest
from pathlib import Path
import subprocess


def test_train_manifest_has_devanagari():
    manifest = Path('data/train/manifest.jsonl')
    assert manifest.exists(), 'train manifest missing'
    # call the checker script
    res = subprocess.run(['python','scripts/check_manifest_devanagari.py', str(manifest)])
    assert res.returncode == 0


def test_dev_manifest_has_devanagari():
    manifest = Path('data/dev/manifest.jsonl')
    assert manifest.exists(), 'dev manifest missing'
    res = subprocess.run(['python','scripts/check_manifest_devanagari.py', str(manifest)])
    assert res.returncode == 0


def test_test_manifest_has_devanagari():
    manifest = Path('data/test/manifest.jsonl')
    assert manifest.exists(), 'test manifest missing'
    res = subprocess.run(['python','scripts/check_manifest_devanagari.py', str(manifest)])
    assert res.returncode == 0
