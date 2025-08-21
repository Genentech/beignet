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
def test_independent_t_test_power(batch_size, dtype):
    """Test independent samples t-test power calculation."""
    # Generate test parameters
    effect_sizes = (
        torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    nobs1_values = (
        torch.tensor([20, 30, 50], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    ratio_values = (
        torch.tensor([1.0, 1.5, 2.0], dtype=dtype).repeat(batch_size, 1).flatten()
    )

    # Test basic functionality
    result = beignet.independent_t_test_power(
        effect_sizes, nobs1_values, ratio_values, alpha=0.05
    )
    assert result.shape == effect_sizes.shape
    assert result.dtype == dtype
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)

    # Test with out parameter
    out = torch.empty_like(effect_sizes)
    result_out = beignet.independent_t_test_power(
        effect_sizes, nobs1_values, ratio_values, alpha=0.05, out=out
    )
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test with default ratio
    result_default = beignet.independent_t_test_power(
        effect_sizes, nobs1_values, alpha=0.05
    )
    assert result_default.shape == effect_sizes.shape

    # Test that power increases with effect size
    small_effect = beignet.independent_t_test_power(
        torch.tensor(0.2, dtype=dtype), torch.tensor(30.0, dtype=dtype)
    )
    large_effect = beignet.independent_t_test_power(
        torch.tensor(0.8, dtype=dtype), torch.tensor(30.0, dtype=dtype)
    )
    assert large_effect > small_effect

    # Test that power increases with sample size
    small_n = beignet.independent_t_test_power(
        torch.tensor(0.5, dtype=dtype), torch.tensor(20.0, dtype=dtype)
    )
    large_n = beignet.independent_t_test_power(
        torch.tensor(0.5, dtype=dtype), torch.tensor(80.0, dtype=dtype)
    )
    assert large_n > small_n

    # Test gradient computation
    effect_grad = effect_sizes.clone().requires_grad_(True)
    nobs1_grad = nobs1_values.clone().requires_grad_(True)
    ratio_grad = ratio_values.clone().requires_grad_(True)
    result_grad = beignet.independent_t_test_power(effect_grad, nobs1_grad, ratio_grad)

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert effect_grad.grad is not None
    assert nobs1_grad.grad is not None
    assert ratio_grad.grad is not None

    # Test torch.compile compatibility
    compiled_ttest_ind_power = torch.compile(
        beignet.independent_t_test_power, fullgraph=True
    )
    result_compiled = compiled_ttest_ind_power(effect_sizes, nobs1_values, ratio_values)
    assert torch.allclose(result, result_compiled, atol=1e-5)


def test_ttest_ind_power_alternatives():
    """Test different alternative hypotheses."""
    effect_size = torch.tensor(0.5, dtype=torch.float64)
    nobs1 = torch.tensor(30.0, dtype=torch.float64)

    # Two-sided test should have lower power than one-sided for same parameters
    power_two_sided = beignet.independent_t_test_power(
        effect_size, nobs1, alternative="two-sided"
    )
    power_larger = beignet.independent_t_test_power(
        effect_size, nobs1, alternative="larger"
    )
    power_smaller = beignet.independent_t_test_power(
        effect_size, nobs1, alternative="smaller"
    )

    assert power_two_sided <= power_larger
    # For positive effect size, "smaller" should have very low power
    assert power_smaller <= power_larger


def test_ttest_ind_power_against_known_values():
    """Test against known power values."""
    # For effect size = 0.5, n1 = n2 = 30, alpha = 0.05, two-sided
    # Expected power ≈ 0.47 (from power analysis literature)
    effect_size = torch.tensor(0.5, dtype=torch.float64)
    nobs1 = torch.tensor(30.0, dtype=torch.float64)

    power = beignet.independent_t_test_power(
        effect_size, nobs1, alpha=0.05, alternative="two-sided"
    )
    expected = 0.47

    # Allow some tolerance for approximation differences
    assert torch.abs(power - expected) < 0.1


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not available")
def test_ttest_ind_power_against_statsmodels():
    """Test independent samples t-test power against statsmodels."""

    effect_sizes = [0.2, 0.5, 0.8]
    nobs1_values = [20, 30, 50]
    ratios = [1.0, 1.5]

    for effect_size in effect_sizes:
        for nobs1 in nobs1_values:
            for ratio in ratios:
                # Test two-sided
                beignet_power = beignet.independent_t_test_power(
                    torch.tensor(effect_size, dtype=torch.float64),
                    torch.tensor(float(nobs1), dtype=torch.float64),
                    torch.tensor(ratio, dtype=torch.float64),
                    alpha=0.05,
                    alternative="two-sided",
                )

                # Use tt_ind_solve_power for independent samples t-test
                statsmodels_power = smp.tt_ind_solve_power(
                    effect_size=effect_size,
                    nobs1=nobs1,
                    alpha=0.05,
                    power=None,
                    ratio=ratio,
                    alternative="two-sided",
                )

                # Allow reasonable tolerance for different approximations
                assert torch.abs(beignet_power - statsmodels_power) < 0.1, (
                    f"Failed for effect_size={effect_size}, nobs1={nobs1}, ratio={ratio}"
                )


def test_ttest_ind_power_edge_cases():
    """Test edge cases for independent t-test power calculation."""
    # Zero effect size should give power ≈ alpha
    zero_effect = beignet.independent_t_test_power(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(50.0, dtype=torch.float64),
        alpha=0.05,
    )
    assert torch.abs(zero_effect - 0.05) < 0.03

    # Very large effect size should give power ≈ 1
    large_effect = beignet.independent_t_test_power(
        torch.tensor(2.0, dtype=torch.float64), torch.tensor(30.0, dtype=torch.float64)
    )
    assert large_effect > 0.9

    # Very small sample size (minimum is 2)
    small_n = beignet.independent_t_test_power(
        torch.tensor(0.5, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),  # Will be clamped to 2
    )
    assert 0.0 <= small_n <= 1.0


def test_ttest_ind_power_ratio_effects():
    """Test effects of different sample size ratios."""
    effect_size = torch.tensor(0.5, dtype=torch.float64)
    nobs1 = torch.tensor(30.0, dtype=torch.float64)

    # Balanced design (ratio=1) should generally have higher power than unbalanced
    power_balanced = beignet.independent_t_test_power(
        effect_size, nobs1, torch.tensor(1.0, dtype=torch.float64)
    )
    power_unbalanced = beignet.independent_t_test_power(
        effect_size, nobs1, torch.tensor(0.5, dtype=torch.float64)
    )

    # For same total sample size, balanced is more powerful
    assert power_balanced > power_unbalanced


def test_ttest_ind_power_consistency():
    """Test consistency with sample size calculation."""
    effect_size = torch.tensor(0.5, dtype=torch.float64)
    ratio = torch.tensor(1.0, dtype=torch.float64)
    target_power = 0.8

    # Calculate required sample size
    nobs1 = beignet.independent_t_test_sample_size(
        effect_size, ratio, power=target_power
    )

    # Calculate power with that sample size
    achieved_power = beignet.independent_t_test_power(effect_size, nobs1, ratio)

    # Should achieve approximately the target power
    assert torch.abs(achieved_power - target_power) < 0.1
