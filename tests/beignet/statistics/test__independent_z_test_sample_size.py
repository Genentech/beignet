"""Test independent z-test sample size (two-sample z-test with known variances)."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from beignet.statistics._independent_z_test_sample_size import (
    independent_z_test_sample_size,
)

try:
    import statsmodels.stats.power as smp

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


@given(
    batch_size=st.integers(min_value=1, max_value=10),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)  # Disable deadline for torch.compile
def test_independent_z_test_sample_size(batch_size: int, dtype: torch.dtype) -> None:
    """Test independent z-test sample size calculation."""

    # Basic functionality tests
    effect_size = torch.rand(batch_size, dtype=dtype) * 0.8 + 0.2  # 0.2 to 1.0
    power = torch.rand(batch_size, dtype=dtype) * 0.3 + 0.5  # 0.5 to 0.8
    ratio = torch.rand(batch_size, dtype=dtype) * 2.0 + 0.5  # 0.5 to 2.5

    # Test basic computation
    sample_size = independent_z_test_sample_size(
        effect_size, ratio, power, alpha=0.05, alternative="two-sided"
    )

    # Check output properties
    assert sample_size.shape == effect_size.shape
    assert sample_size.dtype == dtype
    assert torch.all(sample_size >= 1.0)  # Sample size should be at least 1
    assert torch.all(sample_size <= 10000.0)  # Reasonable upper bound

    # Test different alternatives
    sample_size_larger = independent_z_test_sample_size(
        effect_size, ratio, power, alternative="larger"
    )
    sample_size_smaller = independent_z_test_sample_size(
        effect_size, ratio, power, alternative="smaller"
    )
    sample_size_two_sided = independent_z_test_sample_size(
        effect_size, ratio, power, alternative="two-sided"
    )

    # One-sided tests should generally require smaller samples than two-sided
    assert torch.all(sample_size_larger <= sample_size_two_sided)
    assert torch.all(sample_size_smaller <= sample_size_two_sided)

    # Test monotonicity: larger effect size should require smaller sample
    small_effect = effect_size * 0.5
    large_effect = effect_size * 1.5
    sample_small = independent_z_test_sample_size(
        small_effect, ratio, power, alternative="larger"
    )
    sample_large = independent_z_test_sample_size(
        large_effect, ratio, power, alternative="larger"
    )
    assert torch.all(sample_small >= sample_large)

    # Test monotonicity: higher power should require larger sample
    low_power = power * 0.8
    high_power = torch.clamp(power * 1.2, max=0.95)
    sample_low = independent_z_test_sample_size(
        effect_size, ratio, low_power, alternative="larger"
    )
    sample_high = independent_z_test_sample_size(
        effect_size, ratio, high_power, alternative="larger"
    )
    assert torch.all(sample_high >= sample_low)

    # Test gradients
    effect_size.requires_grad_(True)
    ratio.requires_grad_(True)
    power.requires_grad_(True)

    sample_size = independent_z_test_sample_size(
        effect_size, ratio, power, alternative="larger"
    )
    loss = sample_size.sum()
    loss.backward()

    # Gradients should exist and be finite
    assert effect_size.grad is not None
    assert ratio.grad is not None
    assert power.grad is not None
    assert torch.all(torch.isfinite(effect_size.grad))
    assert torch.all(torch.isfinite(ratio.grad))
    assert torch.all(torch.isfinite(power.grad))

    # Test torch.compile compatibility
    compiled_func = torch.compile(independent_z_test_sample_size, fullgraph=True)
    sample_compiled = compiled_func(
        effect_size.detach(),
        ratio.detach(),
        power.detach(),
        alpha=0.05,
        alternative="larger",
    )
    sample_regular = independent_z_test_sample_size(
        effect_size.detach(),
        ratio.detach(),
        power.detach(),
        alpha=0.05,
        alternative="larger",
    )
    assert torch.allclose(sample_compiled, sample_regular, rtol=1e-5)

    # Test with out parameter
    out = torch.empty_like(sample_size)
    result = independent_z_test_sample_size(
        effect_size.detach(),
        ratio.detach(),
        power.detach(),
        alpha=0.05,
        alternative="larger",
        out=out,
    )
    assert torch.allclose(out, sample_regular, rtol=1e-5)
    assert result is out


def test_independent_z_test_sample_size_known_values() -> None:
    """Test independent normal sample size with known values."""

    # Test case: effect size = 0.5, power = 0.8, equal groups
    effect_size = torch.tensor(0.5)
    ratio = torch.tensor(1.0)  # Equal group sizes
    power = torch.tensor(0.8)

    sample_size_one_sided = independent_z_test_sample_size(
        effect_size, ratio, power, alpha=0.05, alternative="larger"
    )

    # Should be reasonable sample size (between 15 and 150 for moderate effect)
    assert 15 <= float(sample_size_one_sided) <= 150

    # Test invalid alternative
    with pytest.raises(ValueError):
        independent_z_test_sample_size(effect_size, ratio, power, alternative="invalid")

    # Test extreme values
    with pytest.raises(ValueError):
        independent_z_test_sample_size(
            effect_size, ratio, torch.tensor(1.5), alpha=0.05
        )  # Power > 1

    with pytest.raises(ValueError):
        independent_z_test_sample_size(
            effect_size, ratio, torch.tensor(-0.1), alpha=0.05
        )  # Power < 0


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not available")
def test_independent_z_test_sample_size_vs_statsmodels() -> None:
    """Test against statsmodels for verification."""

    effect_size = 0.5
    power = 0.8
    alpha = 0.05
    ratio = 1.0  # Equal groups

    # Test two-sided
    our_result = independent_z_test_sample_size(
        torch.tensor(effect_size),
        torch.tensor(ratio),
        torch.tensor(power),
        alpha=alpha,
        alternative="two-sided",
    )

    # Statsmodels uses normal_power for z-tests with known variance
    # For independent samples we need to solve for the effective sample size
    statsmodels_result = smp.normal_power(
        effect_size=effect_size,
        nobs=None,
        alpha=alpha,
        power=power,
        alternative="two-sided",
    )

    # For equal groups, n1 = n2 = n, so effective_n = n/2, thus n = 2 * effective_n
    expected_n1 = statsmodels_result * 2

    # Should be close (within 15% or 3 units)
    assert abs(float(our_result) - expected_n1) <= max(3, 0.15 * expected_n1)
