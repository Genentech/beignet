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
def test_t_test_sample_size(batch_size, dtype):
    """Test one-sample/paired t-test sample size calculation."""
    # Generate test parameters
    effect_sizes = (
        torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
    )

    # Test basic functionality
    result = beignet.t_test_sample_size(effect_sizes, power=0.8, alpha=0.05)
    assert result.shape == effect_sizes.shape
    assert result.dtype == dtype
    assert torch.all(result >= 2.0)  # Minimum sample size is 2
    assert torch.all(result <= 100000.0)  # Reasonable upper bound

    # Test with out parameter
    out = torch.empty_like(effect_sizes)
    result_out = beignet.t_test_sample_size(
        effect_sizes, power=0.8, alpha=0.05, out=out
    )
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test that sample size decreases with effect size
    small_effect = beignet.t_test_sample_size(torch.tensor(0.2, dtype=dtype), power=0.8)
    large_effect = beignet.t_test_sample_size(torch.tensor(0.8, dtype=dtype), power=0.8)
    assert small_effect > large_effect

    # Test gradient computation
    effect_grad = effect_sizes.clone().requires_grad_(True)
    result_grad = beignet.t_test_sample_size(effect_grad, power=0.8)

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert effect_grad.grad is not None

    # Test torch.compile compatibility
    compiled_ttest_sample_size = torch.compile(
        beignet.t_test_sample_size, fullgraph=True
    )
    result_compiled = compiled_ttest_sample_size(effect_sizes, power=0.8)
    assert torch.allclose(
        result, result_compiled, atol=1e-0
    )  # Allow integer rounding differences


def test_ttest_sample_size_different_powers():
    """Test sample size calculation for different power levels."""
    effect_size = torch.tensor(0.5, dtype=torch.float64)

    # Higher power should require larger sample size
    n_low = beignet.t_test_sample_size(effect_size, power=0.7)
    n_high = beignet.t_test_sample_size(effect_size, power=0.9)

    assert n_high > n_low


def test_ttest_sample_size_alternatives():
    """Test different alternative hypotheses."""
    effect_size = torch.tensor(0.5, dtype=torch.float64)

    # Two-sided test should require larger sample size than one-sided
    n_two_sided = beignet.t_test_sample_size(
        effect_size, power=0.8, alternative="two-sided"
    )
    n_one_sided = beignet.t_test_sample_size(
        effect_size, power=0.8, alternative="one-sided"
    )

    assert n_two_sided > n_one_sided


def test_ttest_sample_size_against_known_values():
    """Test against known sample size values."""
    # For effect size = 0.5, power = 0.8, alpha = 0.05, two-sided
    # Expected sample size ≈ 34 (from power analysis literature)
    effect_size = torch.tensor(0.5, dtype=torch.float64)

    sample_size = beignet.t_test_sample_size(
        effect_size, power=0.8, alpha=0.05, alternative="two-sided"
    )
    expected = 34

    # Allow some tolerance for approximation differences
    assert torch.abs(sample_size - expected) < 10


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not available")
def test_ttest_sample_size_against_statsmodels():
    """Test one-sample t-test sample size against statsmodels."""

    effect_sizes = [0.2, 0.5, 0.8]
    powers = [0.7, 0.8, 0.9]

    for effect_size in effect_sizes:
        for power in powers:
            # Test two-sided
            beignet_n = beignet.t_test_sample_size(
                torch.tensor(effect_size, dtype=torch.float64),
                power=power,
                alpha=0.05,
                alternative="two-sided",
            )

            # Use TTestPower for one-sample t-test
            statsmodels_n = smp.TTestPower().solve_power(
                effect_size=effect_size,
                nobs=None,
                alpha=0.05,
                power=power,
                alternative="two-sided",
            )

            # Allow reasonable tolerance for different approximations and rounding
            assert torch.abs(beignet_n - statsmodels_n) < 5, (
                f"Failed for effect_size={effect_size}, power={power}"
            )


def test_ttest_sample_size_edge_cases():
    """Test edge cases for t-test sample size calculation."""
    # Very small effect size should require large sample size
    small_effect = beignet.t_test_sample_size(
        torch.tensor(0.01, dtype=torch.float64), power=0.8
    )
    assert small_effect > 1000

    # Very large effect size should require small sample size
    large_effect = beignet.t_test_sample_size(
        torch.tensor(2.0, dtype=torch.float64), power=0.8
    )
    assert large_effect < 10

    # Check minimum sample size constraint
    tiny_effect = beignet.t_test_sample_size(
        torch.tensor(1e-8, dtype=torch.float64), power=0.8
    )
    assert tiny_effect >= 2.0


def test_ttest_sample_size_consistency():
    """Test consistency with power calculation."""
    effect_size = torch.tensor(0.5, dtype=torch.float64)
    target_power = 0.8

    # Calculate required sample size
    sample_size = beignet.t_test_sample_size(effect_size, power=target_power)

    # Calculate power with that sample size
    achieved_power = beignet.t_test_power(effect_size, sample_size)

    # Should achieve approximately the target power
    assert torch.abs(achieved_power - target_power) < 0.05


def test_ttest_sample_size_different_alphas():
    """Test sample size calculation for different significance levels."""
    effect_size = torch.tensor(0.5, dtype=torch.float64)

    # Stricter alpha should require larger sample size
    n_strict = beignet.t_test_sample_size(effect_size, power=0.8, alpha=0.01)
    n_lenient = beignet.t_test_sample_size(effect_size, power=0.8, alpha=0.05)

    assert n_strict > n_lenient
