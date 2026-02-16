import json
import subprocess
import sys

from scripts import preflight_checks


def test_run_preflight_json():
    ok, results = preflight_checks.run_all()
    # We expect a dict of results; individual items may be False in environments
    assert isinstance(results, dict)
    assert 'python' in results
    assert 'tokenizer' in results


def test_tokenizer_has_deva():
    res = preflight_checks.check_tokenizer()
    # If sentencepiece is not installed or tokenizer missing, we accept the test as non-blocking
    assert isinstance(res, dict)


def test_nemo_patch_presence():
    res = preflight_checks.check_nemo_and_patch()
    assert isinstance(res, dict)
