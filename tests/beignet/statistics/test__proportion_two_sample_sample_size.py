import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet.statistics

# from statsmodels.stats.power import proportions_ztest_power  # Function not available in current statsmodels version


@given(
    batch_size=st.integers(min_value=1, max_value=5),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_proportion_two_sample_sample_size(batch_size, dtype):
    """Test two-sample proportion sample size calculation."""
    # Generate test parameters
    p1_values = (
        torch.tensor([0.4, 0.5, 0.6], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    p2_values = (
        torch.tensor([0.5, 0.6, 0.7], dtype=dtype).repeat(batch_size, 1).flatten()
    )

    # Test basic functionality - two-sided test
    result = beignet.statistics.proportion_two_sample_sample_size(
        p1_values, p2_values, power=0.8, alpha=0.05, alternative="two-sided"
    )
    assert result.shape == p1_values.shape
    assert result.dtype == dtype
    assert torch.all(result >= 1.0)
    assert torch.all(result < 10000)  # Should be reasonable

    # Test one-sided tests
    result_greater = beignet.statistics.proportion_two_sample_sample_size(
        p1_values, p2_values, power=0.8, alpha=0.05, alternative="greater"
    )
    result_less = beignet.statistics.proportion_two_sample_sample_size(
        p1_values, p2_values, power=0.8, alpha=0.05, alternative="less"
    )

    assert torch.all(result_greater >= 1.0)
    assert torch.all(result_less >= 1.0)

    # One-sided tests should generally require smaller sample sizes than two-sided
    assert torch.all(result_greater <= result + 100)  # Allow some tolerance
    assert torch.all(result_less <= result + 100)

    # Test with different ratios
    result_ratio_2 = beignet.statistics.proportion_two_sample_sample_size(
        p1_values, p2_values, power=0.8, alpha=0.05, ratio=2.0
    )
    assert torch.all(result_ratio_2 >= 1.0)

    # Test with out parameter
    out = torch.empty_like(p1_values)
    result_out = beignet.statistics.proportion_two_sample_sample_size(
        p1_values, p2_values, power=0.8, alpha=0.05, alternative="two-sided", out=out
    )
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test sample size decreases with effect size
    small_effect = beignet.statistics.proportion_two_sample_sample_size(
        torch.tensor(0.5, dtype=dtype), torch.tensor(0.51, dtype=dtype), power=0.8
    )
    large_effect = beignet.statistics.proportion_two_sample_sample_size(
        torch.tensor(0.5, dtype=dtype), torch.tensor(0.6, dtype=dtype), power=0.8
    )

    assert large_effect < small_effect

    # Test sample size increases with power
    low_power = beignet.statistics.proportion_two_sample_sample_size(
        torch.tensor(0.5, dtype=dtype), torch.tensor(0.6, dtype=dtype), power=0.5
    )
    high_power = beignet.statistics.proportion_two_sample_sample_size(
        torch.tensor(0.5, dtype=dtype), torch.tensor(0.6, dtype=dtype), power=0.9
    )

    assert high_power > low_power

    # Test gradient computation
    p1_grad = p1_values.clone().requires_grad_(True)
    p2_grad = p2_values.clone().requires_grad_(True)
    result_grad = beignet.statistics.proportion_two_sample_sample_size(
        p1_grad, p2_grad, power=0.8
    )

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert p1_grad.grad is not None
    assert p2_grad.grad is not None

    # Test torch.compile compatibility
    compiled_proportion_two_sample_sample_size = torch.compile(
        beignet.statistics.proportion_two_sample_sample_size, fullgraph=True
    )
    result_compiled = compiled_proportion_two_sample_sample_size(
        p1_values, p2_values, power=0.8
    )
    assert torch.allclose(result, result_compiled, atol=1e-5)

    # Test invalid alternative
    try:
        beignet.statistics.proportion_two_sample_sample_size(
            torch.tensor(0.5, dtype=dtype),
            torch.tensor(0.6, dtype=dtype),
            alternative="invalid",
        )
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass

    # Test two-sample proportion sample size against known theoretical values
    # For moderate effect (0.5 vs 0.6) with power=0.8, sample size should be reasonable
    p1_known = torch.tensor(0.5, dtype=dtype)
    p2_known = torch.tensor(0.6, dtype=dtype)
    n = beignet.statistics.proportion_two_sample_sample_size(
        p1_known, p2_known, power=0.8, alpha=0.05, alternative="two-sided"
    )

    # Should be somewhere between 200 and 800 for these parameters
    assert 200 < n < 800

    # For large effect (0.5 vs 0.8), sample size should be smaller
    large_effect_n = beignet.statistics.proportion_two_sample_sample_size(
        p1_known, torch.tensor(0.8, dtype=dtype), power=0.8
    )
    assert large_effect_n < n

    # Test that sample size and power calculations are consistent
    # Calculate sample size for given power, then verify power with that sample size
    p1_consistent = torch.tensor(0.5, dtype=dtype)
    p2_consistent = torch.tensor(0.6, dtype=dtype)
    target_power = 0.8
    alpha = 0.05

    # Calculate required sample size
    n1 = beignet.statistics.proportion_two_sample_sample_size(
        p1_consistent, p2_consistent, power=target_power, alpha=alpha
    )
    n2 = n1  # Equal sample sizes

    # Calculate power with that sample size
    actual_power = beignet.statistics.proportion_two_sample_power(
        p1_consistent, p2_consistent, n1, n2, alpha=alpha
    )

    # Should be close to target power (within tolerance for rounding)
    assert abs(float(actual_power) - target_power) < 0.15

    # Test the effect of sample size ratio
    p1_ratio = torch.tensor(0.5, dtype=dtype)
    p2_ratio = torch.tensor(0.6, dtype=dtype)

    # Equal sample sizes (ratio = 1.0)
    n1_equal = beignet.statistics.proportion_two_sample_sample_size(
        p1_ratio, p2_ratio, power=0.8, ratio=1.0
    )

    # Unequal sample sizes (ratio = 2.0, so n2 = 2*n1)
    n1_unequal = beignet.statistics.proportion_two_sample_sample_size(
        p1_ratio, p2_ratio, power=0.8, ratio=2.0
    )

    # With more subjects in group 2, we should need fewer in group 1
    assert n1_unequal < n1_equal

    # But total sample size should be larger
    total_equal = 2 * n1_equal
    total_unequal = n1_unequal + 2 * n1_unequal  # n1 + n2 = n1 + 2*n1 = 3*n1
    assert total_unequal > total_equal

    # Test edge cases for two-sample proportion sample size calculation
    # Test with very small effect size
    tiny_effect = beignet.statistics.proportion_two_sample_sample_size(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(0.501, dtype=dtype),
        power=0.8,
    )
    assert tiny_effect > 1000  # Should require large sample size

    # Test with large effect size
    large_effect_edge = beignet.statistics.proportion_two_sample_sample_size(
        torch.tensor(0.2, dtype=dtype),
        torch.tensor(0.8, dtype=dtype),
        power=0.8,
    )
    assert large_effect_edge < 200  # Should require smaller sample size

    # Test with very high power
    high_power_edge = beignet.statistics.proportion_two_sample_sample_size(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(0.6, dtype=dtype),
        power=0.99,
    )
    normal_power_edge = beignet.statistics.proportion_two_sample_sample_size(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(0.6, dtype=dtype),
        power=0.8,
    )
    assert high_power_edge > normal_power_edge

    # Test with extreme ratio
    extreme_ratio = beignet.statistics.proportion_two_sample_sample_size(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(0.6, dtype=dtype),
        power=0.8,
        ratio=10.0,
    )
    assert extreme_ratio >= 1.0
