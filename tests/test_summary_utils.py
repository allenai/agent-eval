import math

import pytest

from agenteval.summary import _safe_mean, _safe_stderr


def test_safe_mean_all_numbers():
    assert _safe_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_safe_mean_single_number():
    assert _safe_mean([5.0]) == pytest.approx(5.0)


def test_safe_mean_mixed_with_none_score():
    assert _safe_mean([1.0, None, 2.0], is_score=True) == 1.0


def test_safe_mean_mixed_with_none_cost():
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


def test_safe_mean_with_equal_weights():
    """Test weighted mean with equal weights should equal regular mean."""
    values = [1.0, 2.0, 3.0]
    weights = [1.0, 1.0, 1.0]
    result = _safe_mean(values, is_score=True, weights=weights)
    assert result == pytest.approx(2.0)


def test_safe_mean_with_different_weights():
    """Test weighted mean with different weights."""
    values = [1.0, 2.0, 3.0]
    weights = [1.0, 2.0, 1.0]
    # (1*1 + 2*2 + 3*1) / (1+2+1) = 8/4 = 2.0
    result = _safe_mean(values, is_score=True, weights=weights)
    assert result == pytest.approx(2.0)


def test_safe_mean_weighted_score_with_none():
    """Test weighted mean for scores treats None as 0."""
    values = [1.0, None, 3.0]
    weights = [1.0, 2.0, 1.0]
    # (1*1 + 0*2 + 3*1) / (1+2+1) = 4/4 = 1.0
    result = _safe_mean(values, is_score=True, weights=weights)
    assert result == pytest.approx(1.0)


def test_safe_mean_weighted_cost_with_none():
    """Test weighted mean for costs returns None if any value is None."""
    values = [1.0, None, 3.0]
    weights = [1.0, 2.0, 1.0]
    result = _safe_mean(values, is_score=False, weights=weights)
    assert result is None


def test_safe_mean_weighted_empty():
    """Test weighted mean with empty lists."""
    result = _safe_mean([], is_score=True, weights=[])
    assert result is None


def test_safe_mean_weighted_mismatched_lengths():
    """Test weighted mean with mismatched value/weight lengths raises an error."""
    values = [1.0, 2.0]
    weights = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="Length mismatch"):
        _safe_mean(values, is_score=True, weights=weights)


def test_safe_mean_weighted_zero_weights():
    """Test weighted mean with zero total weight raises an error."""
    values = [1.0, 2.0, 3.0]
    weights = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="Total weight is zero"):
        _safe_mean(values, is_score=True, weights=weights)


def test_safe_mean_weighted_zero_weights_cost():
    """Test weighted mean for costs with zero total weight raises an error."""
    values = [1.0, 2.0, 3.0]
    weights = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="Total weight is zero"):
        _safe_mean(values, is_score=False, weights=weights)
