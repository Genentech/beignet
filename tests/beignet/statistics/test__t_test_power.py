import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet

try:
    import statsmodels.stats.power as smp

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


@given(
    batch_size=st.integers(min_value=1, max_value=5),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_t_test_power(batch_size, dtype):
    """Test one-sample/paired t-test power calculation."""
    # Generate test parameters
    effect_sizes = (
        torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    sample_sizes = (
        torch.tensor([20, 50, 100], dtype=dtype).repeat(batch_size, 1).flatten()
    )

    # Test basic functionality
    result = beignet.t_test_power(effect_sizes, sample_sizes, alpha=0.05)
    assert result.shape == effect_sizes.shape
    assert result.dtype == dtype
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)

    # Test with out parameter
    out = torch.empty_like(effect_sizes)
    result_out = beignet.t_test_power(effect_sizes, sample_sizes, alpha=0.05, out=out)
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test that power increases with effect size
    small_effect = beignet.t_test_power(
        torch.tensor(0.2, dtype=dtype), torch.tensor(30.0, dtype=dtype)
    )
    large_effect = beignet.t_test_power(
        torch.tensor(0.8, dtype=dtype), torch.tensor(30.0, dtype=dtype)
    )
    assert large_effect > small_effect

    # Test that power increases with sample size
    small_n = beignet.t_test_power(
        torch.tensor(0.5, dtype=dtype), torch.tensor(20.0, dtype=dtype)
    )
    large_n = beignet.t_test_power(
        torch.tensor(0.5, dtype=dtype), torch.tensor(100.0, dtype=dtype)
    )
    assert large_n > small_n

    # Test gradient computation
    effect_grad = effect_sizes.clone().requires_grad_(True)
    sample_grad = sample_sizes.clone().requires_grad_(True)
    result_grad = beignet.t_test_power(effect_grad, sample_grad)

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert effect_grad.grad is not None
    assert sample_grad.grad is not None

    # Test torch.compile compatibility
    compiled_ttest_power = torch.compile(beignet.t_test_power, fullgraph=True)
    result_compiled = compiled_ttest_power(effect_sizes, sample_sizes)
    assert torch.allclose(result, result_compiled, atol=1e-5)


def test_ttest_power_alternatives():
    """Test different alternative hypotheses."""
    effect_size = torch.tensor(0.5, dtype=torch.float64)
    sample_size = torch.tensor(30.0, dtype=torch.float64)

    # Two-sided test should have lower power than one-sided for same parameters
    power_two_sided = beignet.t_test_power(
        effect_size, sample_size, alternative="two-sided"
    )
    power_one_sided = beignet.t_test_power(
        effect_size, sample_size, alternative="one-sided"
    )

    assert power_two_sided < power_one_sided


def test_ttest_power_against_known_values():
    """Test against known power values."""
    # For effect size = 0.5, n = 30, alpha = 0.05, two-sided
    # Expected power ≈ 0.659 (from power analysis literature)
    effect_size = torch.tensor(0.5, dtype=torch.float64)
    sample_size = torch.tensor(30.0, dtype=torch.float64)

    power = beignet.t_test_power(
        effect_size, sample_size, alpha=0.05, alternative="two-sided"
    )
    expected = 0.659

    # Allow some tolerance for approximation differences
    assert torch.abs(power - expected) < 0.05


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not available")
def test_ttest_power_against_statsmodels():
    """Test one-sample t-test power against statsmodels."""

    effect_sizes = [0.2, 0.5, 0.8]
    sample_sizes = [20, 30, 50]

    for effect_size in effect_sizes:
        for sample_size in sample_sizes:
            # Test two-sided
            beignet_power = beignet.t_test_power(
                torch.tensor(effect_size, dtype=torch.float64),
                torch.tensor(float(sample_size), dtype=torch.float64),
                alpha=0.05,
                alternative="two-sided",
            )

            # Use TTestPower for one-sample t-test
            statsmodels_power = smp.TTestPower().solve_power(
                effect_size=effect_size,
                nobs=sample_size,
                alpha=0.05,
                power=None,
                alternative="two-sided",
            )

            # Allow reasonable tolerance for different approximations
            assert torch.abs(beignet_power - statsmodels_power) < 0.05, (
                f"Failed for effect_size={effect_size}, sample_size={sample_size}"
            )


def test_ttest_power_edge_cases():
    """Test edge cases for t-test power calculation."""
    # Zero effect size should give power ≈ alpha
    zero_effect = beignet.t_test_power(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(50.0, dtype=torch.float64),
        alpha=0.05,
    )
    assert torch.abs(zero_effect - 0.05) < 0.025

    # Very large effect size should give power ≈ 1
    large_effect = beignet.t_test_power(
        torch.tensor(3.0, dtype=torch.float64), torch.tensor(30.0, dtype=torch.float64)
    )
    assert large_effect > 0.99

    # Very small sample size (minimum is 2)
    small_n = beignet.t_test_power(
        torch.tensor(0.5, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),  # Will be clamped to 2
    )
    assert 0.0 <= small_n <= 1.0


def test_ttest_power_consistency():
    """Test consistency with sample size calculation."""
    effect_size = torch.tensor(0.5, dtype=torch.float64)
    target_power = 0.8

    # Calculate required sample size
    sample_size = beignet.t_test_sample_size(effect_size, power=target_power)

    # Calculate power with that sample size
    achieved_power = beignet.t_test_power(effect_size, sample_size)

    # Should achieve approximately the target power
    assert torch.abs(achieved_power - target_power) < 0.05
