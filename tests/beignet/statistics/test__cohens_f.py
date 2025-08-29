import hypothesis
import hypothesis.strategies
import torch

import beignet
import beignet.statistics

try:
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@hypothesis.given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=5),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_cohens_f(batch_size, dtype):
    """Test Cohen's f effect size calculation."""
    # Generate test parameters - create group means for 3 groups
    group_means = (
        torch.tensor(
            [[10.0, 12.0, 14.0], [5.0, 7.0, 9.0], [20.0, 22.0, 24.0]],
            dtype=dtype,
        )
        .repeat(batch_size, 1, 1)
        .view(-1, 3)
    )

    pooled_stds = (
        torch.tensor([2.0, 1.5, 3.0], dtype=dtype).repeat(batch_size).flatten()
    )

    # Test basic functionality
    result = beignet.statistics.cohens_f(group_means, pooled_stds)
    assert result.shape == pooled_stds.shape
    assert result.dtype == dtype
    assert torch.all(result >= 0.0)

    # Test with out parameter
    out = torch.empty_like(pooled_stds)
    result_out = beignet.statistics.cohens_f(group_means, pooled_stds, out=out)
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test that Cohen's f increases with larger spread in group means
    close_means = torch.tensor([10.0, 10.1, 10.2], dtype=dtype)
    spread_means = torch.tensor([10.0, 15.0, 20.0], dtype=dtype)
    pooled_std = torch.tensor(2.0, dtype=dtype)

    f_close = beignet.statistics.cohens_f(close_means, pooled_std)
    f_spread = beignet.statistics.cohens_f(spread_means, pooled_std)

    assert f_spread > f_close

    # Test that Cohen's f decreases with larger pooled standard deviation
    group_means_test = torch.tensor([10.0, 12.0, 14.0], dtype=dtype)
    small_std = torch.tensor(1.0, dtype=dtype)
    large_std = torch.tensor(5.0, dtype=dtype)

    f_small_std = beignet.statistics.cohens_f(group_means_test, small_std)
    f_large_std = beignet.statistics.cohens_f(group_means_test, large_std)

    assert f_small_std > f_large_std

    # Test gradient computation
    group_means_grad = group_means.clone().requires_grad_(True)
    pooled_stds_grad = pooled_stds.clone().requires_grad_(True)
    result_grad = beignet.statistics.cohens_f(group_means_grad, pooled_stds_grad)

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert group_means_grad.grad is not None
    assert pooled_stds_grad.grad is not None

    # Test torch.compile compatibility
    compiled_cohens_f = torch.compile(beignet.statistics.cohens_f, fullgraph=True)
    result_compiled = compiled_cohens_f(group_means, pooled_stds)
    assert torch.allclose(result, result_compiled, atol=1e-6)

    # Test zero standard deviation handling
    zero_std = torch.tensor(0.0, dtype=dtype)
    result_zero = beignet.statistics.cohens_f(group_means_test, zero_std)
    assert torch.isfinite(result_zero)  # Should not be inf/nan

    # Test Cohen's f against known values
    # Example from Cohen (1988)
    # Three groups with means [10, 12, 14] and pooled std = 2
    group_means_known = torch.tensor([10.0, 12.0, 14.0], dtype=dtype)
    pooled_std_known = torch.tensor(2.0, dtype=dtype)

    f = beignet.statistics.cohens_f(group_means_known, pooled_std_known)

    # Manual calculation: std([10, 12, 14]) = sqrt(8/3) ≈ 1.633
    # Cohen's f = 1.633 / 2 ≈ 0.8165
    expected = 1.633 / 2.0
    assert torch.abs(f - expected) < 0.01

    # Test interpretation guidelines
    # Small effect: f = 0.10
    small_means = torch.tensor([10.0, 10.2, 10.4], dtype=dtype)
    pooled_std_small = torch.tensor(2.0, dtype=dtype)
    f_small = beignet.statistics.cohens_f(small_means, pooled_std_small)
    assert f_small < 0.2  # Should be small effect

    # Large effect: f = 0.40
    large_means = torch.tensor([10.0, 15.0, 20.0], dtype=dtype)
    pooled_std_large = torch.tensor(5.0, dtype=dtype)
    f_large = beignet.statistics.cohens_f(large_means, pooled_std_large)
    assert f_large > 0.8  # Should be large effect

    # Test Cohen's f with manual calculation
    # Test case with known outcome
    group_means_manual = torch.tensor([8.0, 10.0, 12.0], dtype=dtype)
    pooled_std_manual = torch.tensor(2.0, dtype=dtype)

    # Manual calculation
    # Mean of means = (8 + 10 + 12) / 3 = 10
    # Variance of means = [(8-10)² + (10-10)² + (12-10)²] / 3 = [4 + 0 + 4] / 3 = 8/3
    # Std of means = sqrt(8/3) ≈ 1.6329931618554521
    # Cohen's f = 1.6329931618554521 / 2.0 ≈ 0.8164965809277261

    result_manual = beignet.statistics.cohens_f(group_means_manual, pooled_std_manual)
    expected_manual = 0.8164965809277261

    assert torch.abs(result_manual - expected_manual) < 1e-10

    # Test relationship between Cohen's f and Cohen's d for two groups
    # For two groups, Cohen's f = Cohen's d / 2
    # Create sample data with known means and standard deviation
    torch.manual_seed(42)
    group1 = torch.normal(10.0, 2.0, (50,), dtype=dtype)
    group2 = torch.normal(12.0, 2.0, (50,), dtype=dtype)

    # Calculate Cohen's d using actual group data
    cohens_d_value = beignet.statistics.cohens_d(group1, group2, pooled=True)

    # Calculate Cohen's f using group means and pooled standard deviation
    group1_mean = torch.mean(group1)
    group2_mean = torch.mean(group2)
    group_means_rel = torch.stack([group1_mean, group2_mean])

    # Calculate pooled standard deviation
    n1, n2 = group1.shape[0], group2.shape[0]
    var1 = torch.var(group1, unbiased=True)
    var2 = torch.var(group2, unbiased=True)
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_std_rel = torch.sqrt(pooled_var)

    cohens_f_value = beignet.statistics.cohens_f(group_means_rel, pooled_std_rel)

    # Check relationship: f = |d| / 2 for two groups
    expected_cohens_f = torch.abs(cohens_d_value) / 2.0

    assert torch.allclose(cohens_f_value, expected_cohens_f, atol=1e-2)

    # Test edge cases for Cohen's f calculation
    # Test with identical group means (should give f = 0)
    identical_means = torch.tensor([10.0, 10.0, 10.0], dtype=dtype)
    pooled_std_edge = torch.tensor(2.0, dtype=dtype)
    f_identical = beignet.statistics.cohens_f(identical_means, pooled_std_edge)
    assert torch.abs(f_identical) < 1e-10

    # Test with very small pooled std
    group_means_edge = torch.tensor([10.0, 12.0, 14.0], dtype=dtype)
    tiny_std = torch.tensor(1e-12, dtype=dtype)
    f_tiny_std = beignet.statistics.cohens_f(group_means_edge, tiny_std)
    assert torch.isfinite(f_tiny_std)
    assert f_tiny_std > 1000  # Should be very large

    # Test with single group (edge case)
    single_group = torch.tensor([10.0], dtype=dtype)
    f_single = beignet.statistics.cohens_f(single_group, pooled_std_edge)
    assert torch.abs(f_single) < 1e-10  # Standard deviation of single value is 0
