import math

import pytest

from agenteval.summary import _safe_mean, _safe_stderr


def test_safe_mean_all_numbers():
    assert _safe_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_safe_mean_single_number():
    assert _safe_mean([5.0]) == pytest.approx(5.0)


def test_safe_mean_mixed_with_none():
    assert _safe_mean([1.0, None, 3.0]) is None


def test_safe_mean_empty_list():
    assert _safe_mean([]) is None


def test_safe_stderr_all_numbers():
    expected = 1.0 / math.sqrt(3)
    assert _safe_stderr([1.0, 2.0, 3.0]) == pytest.approx(expected)


def test_safe_stderr_single_number():
    assert _safe_stderr([5.0]) is None


def test_safe_stderr_mixed_with_none():
    assert _safe_stderr([1.0, None, 3.0]) is None


def test_safe_stderr_empty_list():
    assert _safe_stderr([]) is None
