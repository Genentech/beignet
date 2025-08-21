import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet

try:
    from statsmodels.stats.power import proportions_ztest_power

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


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
    result = beignet.proportion_two_sample_sample_size(
        p1_values, p2_values, power=0.8, alpha=0.05, alternative="two-sided"
    )
    assert result.shape == p1_values.shape
    assert result.dtype == dtype
    assert torch.all(result >= 1.0)
    assert torch.all(result < 10000)  # Should be reasonable

    # Test one-sided tests
    result_greater = beignet.proportion_two_sample_sample_size(
        p1_values, p2_values, power=0.8, alpha=0.05, alternative="greater"
    )
    result_less = beignet.proportion_two_sample_sample_size(
        p1_values, p2_values, power=0.8, alpha=0.05, alternative="less"
    )

    assert torch.all(result_greater >= 1.0)
    assert torch.all(result_less >= 1.0)

    # One-sided tests should generally require smaller sample sizes than two-sided
    assert torch.all(result_greater <= result + 100)  # Allow some tolerance
    assert torch.all(result_less <= result + 100)

    # Test with different ratios
    result_ratio_2 = beignet.proportion_two_sample_sample_size(
        p1_values, p2_values, power=0.8, alpha=0.05, ratio=2.0
    )
    assert torch.all(result_ratio_2 >= 1.0)

    # Test with out parameter
    out = torch.empty_like(p1_values)
    result_out = beignet.proportion_two_sample_sample_size(
        p1_values, p2_values, power=0.8, alpha=0.05, alternative="two-sided", out=out
    )
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test sample size decreases with effect size
    small_effect = beignet.proportion_two_sample_sample_size(
        torch.tensor(0.5, dtype=dtype), torch.tensor(0.51, dtype=dtype), power=0.8
    )
    large_effect = beignet.proportion_two_sample_sample_size(
        torch.tensor(0.5, dtype=dtype), torch.tensor(0.6, dtype=dtype), power=0.8
    )

    assert large_effect < small_effect

    # Test sample size increases with power
    low_power = beignet.proportion_two_sample_sample_size(
        torch.tensor(0.5, dtype=dtype), torch.tensor(0.6, dtype=dtype), power=0.5
    )
    high_power = beignet.proportion_two_sample_sample_size(
        torch.tensor(0.5, dtype=dtype), torch.tensor(0.6, dtype=dtype), power=0.9
    )

    assert high_power > low_power

    # Test gradient computation
    p1_grad = p1_values.clone().requires_grad_(True)
    p2_grad = p2_values.clone().requires_grad_(True)
    result_grad = beignet.proportion_two_sample_sample_size(p1_grad, p2_grad, power=0.8)

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert p1_grad.grad is not None
    assert p2_grad.grad is not None

    # Test torch.compile compatibility
    compiled_proportion_two_sample_sample_size = torch.compile(
        beignet.proportion_two_sample_sample_size, fullgraph=True
    )
    result_compiled = compiled_proportion_two_sample_sample_size(
        p1_values, p2_values, power=0.8
    )
    assert torch.allclose(result, result_compiled, atol=1e-5)

    # Test invalid alternative
    try:
        beignet.proportion_two_sample_sample_size(
            torch.tensor(0.5, dtype=dtype),
            torch.tensor(0.6, dtype=dtype),
            alternative="invalid",
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_proportion_two_sample_sample_size_against_known_values():
    """Test two-sample proportion sample size against known theoretical values."""
    # For moderate effect (0.5 vs 0.6) with power=0.8, sample size should be reasonable
    p1 = torch.tensor(0.5, dtype=torch.float32)
    p2 = torch.tensor(0.6, dtype=torch.float32)
    n = beignet.proportion_two_sample_sample_size(
        p1, p2, power=0.8, alpha=0.05, alternative="two-sided"
    )

    # Should be somewhere between 200 and 800 for these parameters
    assert 200 < n < 800

    # For large effect (0.5 vs 0.8), sample size should be smaller
    large_effect_n = beignet.proportion_two_sample_sample_size(
        p1, torch.tensor(0.8, dtype=torch.float32), power=0.8
    )
    assert large_effect_n < n


def test_proportion_two_sample_sample_size_consistency():
    """Test that sample size and power calculations are consistent."""
    # Calculate sample size for given power, then verify power with that sample size
    p1 = torch.tensor(0.5, dtype=torch.float64)
    p2 = torch.tensor(0.6, dtype=torch.float64)
    target_power = 0.8
    alpha = 0.05

    # Calculate required sample size
    n1 = beignet.proportion_two_sample_sample_size(
        p1, p2, power=target_power, alpha=alpha
    )
    n2 = n1  # Equal sample sizes

    # Calculate power with that sample size
    actual_power = beignet.proportion_two_sample_power(p1, p2, n1, n2, alpha=alpha)

    # Should be close to target power (within tolerance for rounding)
    assert abs(float(actual_power) - target_power) < 0.15


def test_proportion_two_sample_sample_size_ratio_effects():
    """Test the effect of sample size ratio."""
    p1 = torch.tensor(0.5, dtype=torch.float64)
    p2 = torch.tensor(0.6, dtype=torch.float64)

    # Equal sample sizes (ratio = 1.0)
    n1_equal = beignet.proportion_two_sample_sample_size(p1, p2, power=0.8, ratio=1.0)

    # Unequal sample sizes (ratio = 2.0, so n2 = 2*n1)
    n1_unequal = beignet.proportion_two_sample_sample_size(p1, p2, power=0.8, ratio=2.0)

    # With more subjects in group 2, we should need fewer in group 1
    assert n1_unequal < n1_equal

    # But total sample size should be larger
    total_equal = 2 * n1_equal
    total_unequal = n1_unequal + 2 * n1_unequal  # n1 + n2 = n1 + 2*n1 = 3*n1
    assert total_unequal > total_equal


def test_proportion_two_sample_sample_size_edge_cases():
    """Test edge cases for two-sample proportion sample size calculation."""
    # Test with very small effect size
    tiny_effect = beignet.proportion_two_sample_sample_size(
        torch.tensor(0.5, dtype=torch.float64),
        torch.tensor(0.501, dtype=torch.float64),
        power=0.8,
    )
    assert tiny_effect > 1000  # Should require large sample size

    # Test with large effect size
    large_effect = beignet.proportion_two_sample_sample_size(
        torch.tensor(0.2, dtype=torch.float64),
        torch.tensor(0.8, dtype=torch.float64),
        power=0.8,
    )
    assert large_effect < 200  # Should require smaller sample size

    # Test with very high power
    high_power = beignet.proportion_two_sample_sample_size(
        torch.tensor(0.5, dtype=torch.float64),
        torch.tensor(0.6, dtype=torch.float64),
        power=0.99,
    )
    normal_power = beignet.proportion_two_sample_sample_size(
        torch.tensor(0.5, dtype=torch.float64),
        torch.tensor(0.6, dtype=torch.float64),
        power=0.8,
    )
    assert high_power > normal_power

    # Test with extreme ratio
    extreme_ratio = beignet.proportion_two_sample_sample_size(
        torch.tensor(0.5, dtype=torch.float64),
        torch.tensor(0.6, dtype=torch.float64),
        power=0.8,
        ratio=10.0,
    )
    assert extreme_ratio >= 1.0
