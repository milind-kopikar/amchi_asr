import pytest
from scripts.run_micro_overfit import _compute_char_distance


def test_char_distance_exact():
    assert _compute_char_distance("रोहन", "रोहन") == 0.0


def test_char_distance_diff():
    d = _compute_char_distance("रोहन", "रहन")
    assert d > 0


def test_char_distance_normalized():
    d = _compute_char_distance("abc", "")
    assert d == 1.0
