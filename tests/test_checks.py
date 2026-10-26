import random

import pytest

from reward_lens.checks import (assert_consistent, assert_correlated, assert_non_degenerate,
                                assert_strict_intervals, mae, pearson, rmse)

TRUTH = [float(i) for i in range(50)]
GOOD = [t + (0.5 if i % 2 else -0.5) for i, t in enumerate(TRUTH)]
ZEROS = [0.0] * 50
SHUFFLED = random.Random(0).sample(GOOD, len(GOOD))


def test_a_good_answer_passes_every_check():
    assert_non_degenerate(GOOD)
    assert_strict_intervals([g - 1 for g in GOOD], GOOD, [g + 1 for g in GOOD])
    assert_correlated(GOOD, TRUTH, min_r=0.9)
    assert_consistent({"rmse": 0.5, "mae": 0.5}, {"rmse": rmse(GOOD, TRUTH), "mae": mae(GOOD, TRUTH)})


def test_all_zeros_is_caught():
    with pytest.raises(AssertionError, match="distinct"):
        assert_non_degenerate(ZEROS)
    with pytest.raises(AssertionError, match="empty"):
        assert_strict_intervals(ZEROS, ZEROS, ZEROS)  # 0 <= 0 <= 0 is valid for a weak check, not this one
    with pytest.raises(AssertionError, match="correlation"):
        assert_correlated(ZEROS, TRUTH)


def test_shuffled_values_pass_format_level_checks_but_not_correlation():
    assert_non_degenerate(SHUFFLED)  # same distribution as a good answer...
    with pytest.raises(AssertionError, match="correlation"):
        assert_correlated(SHUFFLED, TRUTH, min_r=0.5)  # ...but the values are in the wrong rows


def test_invented_metrics_do_not_add_up():
    with pytest.raises(AssertionError, match="rmse: reported 0 vs recomputed"):
        assert_consistent({"rmse": 0.0, "mae": 0.5}, {"rmse": rmse(GOOD, TRUTH), "mae": mae(GOOD, TRUTH)})
    with pytest.raises(AssertionError, match="missing"):
        assert_consistent({}, {"rmse": 1.0})


def test_numeric_edge_cases():
    assert pearson([1, 2, 3], [5, 5, 5]) == 0.0
    with pytest.raises(AssertionError, match="NaN"):
        assert_non_degenerate([1.0, float("nan")])
    with pytest.raises(AssertionError, match="no values"):
        assert_non_degenerate([])
    with pytest.raises(AssertionError, match="length"):
        pearson([1, 2], [1, 2, 3])
    with pytest.raises(AssertionError, match="standard deviation"):
        assert_non_degenerate([1.0, 1.0 + 1e-12], min_std=1e-6)


def test_works_with_numpy_arrays():
    np = pytest.importorskip("numpy")
    assert_correlated(np.array(GOOD), np.array(TRUTH), min_r=0.9)


def test_truncated_outputs_are_length_mismatches():
    with pytest.raises(AssertionError, match="length mismatch"):
        rmse([1.0], [1.0, 100.0])
    with pytest.raises(AssertionError, match="length mismatch"):
        assert_strict_intervals([0, 0], [1, 1], [2])
