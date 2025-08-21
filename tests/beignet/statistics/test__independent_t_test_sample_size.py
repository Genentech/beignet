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
def test_independent_t_test_sample_size(batch_size, dtype):
    """Test independent samples t-test sample size calculation."""
    # Generate test parameters
    effect_sizes = (
        torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    ratio_values = (
        torch.tensor([1.0, 1.5, 2.0], dtype=dtype).repeat(batch_size, 1).flatten()
    )

    # Test basic functionality
    result = beignet.independent_t_test_sample_size(
        effect_sizes, ratio_values, power=0.8, alpha=0.05
    )
    assert result.shape == effect_sizes.shape
    assert result.dtype == dtype
    assert torch.all(result >= 2.0)  # Minimum sample size is 2
    assert torch.all(result <= 100000.0)  # Reasonable upper bound

    # Test with out parameter
    out = torch.empty_like(effect_sizes)
    result_out = beignet.independent_t_test_sample_size(
        effect_sizes, ratio_values, power=0.8, alpha=0.05, out=out
    )
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test with default ratio
    result_default = beignet.independent_t_test_sample_size(
        effect_sizes, power=0.8, alpha=0.05
    )
    assert result_default.shape == effect_sizes.shape

    # Test that sample size decreases with effect size
    small_effect = beignet.independent_t_test_sample_size(
        torch.tensor(0.2, dtype=dtype), power=0.8
    )
    large_effect = beignet.independent_t_test_sample_size(
        torch.tensor(0.8, dtype=dtype), power=0.8
    )
    assert small_effect > large_effect

    # Test gradient computation
    effect_grad = effect_sizes.clone().requires_grad_(True)
    ratio_grad = ratio_values.clone().requires_grad_(True)
    result_grad = beignet.independent_t_test_sample_size(
        effect_grad, ratio_grad, power=0.8
    )

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert effect_grad.grad is not None
    assert ratio_grad.grad is not None

    # Test torch.compile compatibility
    compiled_ttest_ind_sample_size = torch.compile(
        beignet.independent_t_test_sample_size, fullgraph=True
    )
    result_compiled = compiled_ttest_ind_sample_size(
        effect_sizes, ratio_values, power=0.8
    )
    assert torch.allclose(
        result, result_compiled, atol=1e-0
    )  # Allow integer rounding differences


def test_ttest_ind_sample_size_different_powers():
    """Test sample size calculation for different power levels."""
    effect_size = torch.tensor(0.5, dtype=torch.float64)
    ratio = torch.tensor(1.0, dtype=torch.float64)

    # Higher power should require larger sample size
    n_low = beignet.independent_t_test_sample_size(effect_size, ratio, power=0.7)
    n_high = beignet.independent_t_test_sample_size(effect_size, ratio, power=0.9)

    assert n_high > n_low


def test_ttest_ind_sample_size_alternatives():
    """Test different alternative hypotheses."""
    effect_size = torch.tensor(0.5, dtype=torch.float64)
    ratio = torch.tensor(1.0, dtype=torch.float64)

    # Two-sided test should require larger sample size than one-sided
    n_two_sided = beignet.independent_t_test_sample_size(
        effect_size, ratio, power=0.8, alternative="two-sided"
    )
    n_larger = beignet.independent_t_test_sample_size(
        effect_size, ratio, power=0.8, alternative="larger"
    )

    assert n_two_sided > n_larger


def test_ttest_ind_sample_size_against_known_values():
    """Test against known sample size values."""
    # For effect size = 0.5, power = 0.8, alpha = 0.05, two-sided, ratio = 1
    # Expected sample size per group ≈ 64 (from power analysis literature)
    effect_size = torch.tensor(0.5, dtype=torch.float64)
    ratio = torch.tensor(1.0, dtype=torch.float64)

    sample_size = beignet.independent_t_test_sample_size(
        effect_size, ratio, power=0.8, alpha=0.05, alternative="two-sided"
    )
    expected = 64

    # Allow some tolerance for approximation differences
    assert torch.abs(sample_size - expected) < 15


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not available")
def test_ttest_ind_sample_size_against_statsmodels():
    """Test independent samples t-test sample size against statsmodels."""

    effect_sizes = [0.2, 0.5, 0.8]
    powers = [0.7, 0.8, 0.9]
    ratios = [1.0, 1.5]

    for effect_size in effect_sizes:
        for power in powers:
            for ratio in ratios:
                # Test two-sided
                beignet_n = beignet.independent_t_test_sample_size(
                    torch.tensor(effect_size, dtype=torch.float64),
                    torch.tensor(ratio, dtype=torch.float64),
                    power=power,
                    alpha=0.05,
                    alternative="two-sided",
                )

                # Use tt_ind_solve_power for independent samples t-test
                statsmodels_n = smp.tt_ind_solve_power(
                    effect_size=effect_size,
                    nobs1=None,
                    alpha=0.05,
                    power=power,
                    ratio=ratio,
                    alternative="two-sided",
                )

                # Allow reasonable tolerance for different approximations and rounding
                assert torch.abs(beignet_n - statsmodels_n) < 10, (
                    f"Failed for effect_size={effect_size}, power={power}, ratio={ratio}"
                )


def test_ttest_ind_sample_size_edge_cases():
    """Test edge cases for independent t-test sample size calculation."""
    # Very small effect size should require large sample size
    small_effect = beignet.independent_t_test_sample_size(
        torch.tensor(0.01, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
        power=0.8,
    )
    assert small_effect > 1000

    # Very large effect size should require small sample size
    large_effect = beignet.independent_t_test_sample_size(
        torch.tensor(2.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
        power=0.8,
    )
    assert large_effect < 10

    # Check minimum sample size constraint
    tiny_effect = beignet.independent_t_test_sample_size(
        torch.tensor(1e-8, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
        power=0.8,
    )
    assert tiny_effect >= 2.0


def test_ttest_ind_sample_size_ratio_effects():
    """Test effects of different sample size ratios."""
    effect_size = torch.tensor(0.5, dtype=torch.float64)

    # Balanced design (ratio=1) should require smallest sample size
    n1_balanced = beignet.independent_t_test_sample_size(
        effect_size, torch.tensor(1.0, dtype=torch.float64), power=0.8
    )
    n1_unbalanced = beignet.independent_t_test_sample_size(
        effect_size, torch.tensor(0.5, dtype=torch.float64), power=0.8
    )

    # For same power, balanced design is more efficient
    assert n1_balanced < n1_unbalanced


def test_ttest_ind_sample_size_consistency():
    """Test consistency with power calculation."""
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


def test_ttest_ind_sample_size_different_alphas():
    """Test sample size calculation for different significance levels."""
    effect_size = torch.tensor(0.5, dtype=torch.float64)
    ratio = torch.tensor(1.0, dtype=torch.float64)

    # Stricter alpha should require larger sample size
    n_strict = beignet.independent_t_test_sample_size(
        effect_size, ratio, power=0.8, alpha=0.01
    )
    n_lenient = beignet.independent_t_test_sample_size(
        effect_size, ratio, power=0.8, alpha=0.05
    )

    assert n_strict > n_lenient
