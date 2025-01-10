import numpy
import pytest
from beignet import maximum_mean_discrepancy
from numpy.testing import assert_almost_equal


def test_maximum_mean_discrepancy():
    # Test 1: Basic functionality - two clearly different distributions
    num_samples = 1024

    rng = numpy.random.RandomState(42)  # Fixed seed for reproducibility
    X = rng.normal(0, 1, (num_samples, 2))
    Y = rng.normal(5, 1, (num_samples, 2))
    mmd = maximum_mean_discrepancy(X, Y)
    assert mmd > 0, "MMD should be positive for different distributions"

    # Test 2: Identity - same distribution should give near-zero MMD
    X = rng.normal(0, 1, (num_samples, 2))
    mmd_same = maximum_mean_discrepancy(X, X.copy())
    assert (
        mmd_same < 0.1
    ), f"MMD should be close to 0 for identical distributions, got {mmd_same}"

    # Test 3: Symmetry property
    X = rng.normal(0, 1, (num_samples, 2))
    Y = rng.normal(3, 1, (num_samples, 2))
    split_rng = numpy.random.default_rng(42)
    mmd_xy = maximum_mean_discrepancy(X, Y, rng=split_rng)
    split_rng = numpy.random.default_rng(42)
    mmd_yx = maximum_mean_discrepancy(Y, X, rng=split_rng)
    assert_almost_equal(mmd_xy, mmd_yx, decimal=5, err_msg="MMD should be symmetric")

    # Test 4: Custom distance function
    def manhattan_distance(x, y):
        return numpy.abs(x - y).sum(axis=-1)

    mmd_custom = maximum_mean_discrepancy(X, Y, distance_fn=manhattan_distance)
    assert mmd_custom > 0, "MMD with custom distance should be positive"

    # Test 5: Custom kernel width
    mmd_width = maximum_mean_discrepancy(X, Y, kernel_width=1.0)
    assert mmd_width > 0, "MMD with custom kernel width should be positive"

    # Test 6: Edge cases
    # Single point
    X_single = numpy.array([[1.0, 1.0]])
    Y_single = numpy.array([[2.0, 2.0]])
    with pytest.raises(ValueError):
        _ = maximum_mean_discrepancy(X_single, Y_single)

    # Test 7: Different sample sizes
    X_large = rng.normal(0, 1, (150, 2))
    Y_small = rng.normal(0, 1, (50, 2))
    mmd_diff_size = maximum_mean_discrepancy(X_large, Y_small)
    assert numpy.isfinite(mmd_diff_size), "MMD should handle different sample sizes"

    # Test 8: Verify positive definiteness
    assert mmd >= 0, "MMD should be non-negative"
    assert mmd_width >= 0, "MMD should be non-negative with custom width"
    assert mmd_custom >= 0, "MMD should be non-negative with custom distance"
