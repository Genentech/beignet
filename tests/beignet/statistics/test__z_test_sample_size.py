"""Test z-test sample size (z-test with known variance)."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from beignet.statistics._z_test_sample_size import z_test_sample_size

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
def test_z_test_sample_size(batch_size: int, dtype: torch.dtype) -> None:
    """Test z-test sample size calculation."""

    # Basic functionality tests
    effect_size = torch.rand(batch_size, dtype=dtype) * 0.8 + 0.2  # 0.2 to 1.0
    power = torch.rand(batch_size, dtype=dtype) * 0.3 + 0.5  # 0.5 to 0.8

    # Test basic computation
    sample_size = z_test_sample_size(
        effect_size, power, alpha=0.05, alternative="two-sided"
    )

    # Check output properties
    assert sample_size.shape == effect_size.shape
    assert sample_size.dtype == dtype
    assert torch.all(sample_size >= 1.0)  # Sample size should be at least 1
    assert torch.all(sample_size <= 10000.0)  # Reasonable upper bound

    # Test different alternatives
    sample_size_larger = z_test_sample_size(effect_size, power, alternative="larger")
    sample_size_smaller = z_test_sample_size(effect_size, power, alternative="smaller")
    sample_size_two_sided = z_test_sample_size(
        effect_size, power, alternative="two-sided"
    )

    # One-sided tests should generally require smaller samples than two-sided
    assert torch.all(sample_size_larger <= sample_size_two_sided)
    assert torch.all(sample_size_smaller <= sample_size_two_sided)

    # Test monotonicity: larger effect size should require smaller sample
    small_effect = effect_size * 0.5
    large_effect = effect_size * 1.5
    sample_small = z_test_sample_size(small_effect, power, alternative="larger")
    sample_large = z_test_sample_size(large_effect, power, alternative="larger")
    assert torch.all(sample_small >= sample_large)

    # Test monotonicity: higher power should require larger sample
    low_power = power * 0.8
    high_power = torch.clamp(power * 1.2, max=0.95)
    sample_low = z_test_sample_size(effect_size, low_power, alternative="larger")
    sample_high = z_test_sample_size(effect_size, high_power, alternative="larger")
    assert torch.all(sample_high >= sample_low)

    # Test gradients
    effect_size.requires_grad_(True)
    power.requires_grad_(True)

    sample_size = z_test_sample_size(effect_size, power, alternative="larger")
    loss = sample_size.sum()
    loss.backward()

    # Gradients should exist and be finite
    assert effect_size.grad is not None
    assert power.grad is not None
    assert torch.all(torch.isfinite(effect_size.grad))
    assert torch.all(torch.isfinite(power.grad))

    # Test torch.compile compatibility
    compiled_func = torch.compile(z_test_sample_size, fullgraph=True)
    sample_compiled = compiled_func(
        effect_size.detach(), power.detach(), alpha=0.05, alternative="larger"
    )
    sample_regular = z_test_sample_size(
        effect_size.detach(), power.detach(), alpha=0.05, alternative="larger"
    )
    assert torch.allclose(sample_compiled, sample_regular, rtol=1e-5)

    # Test with out parameter
    out = torch.empty_like(sample_size)
    result = z_test_sample_size(
        effect_size.detach(), power.detach(), alpha=0.05, alternative="larger", out=out
    )
    assert torch.allclose(out, sample_regular, rtol=1e-5)
    assert result is out


def test_z_test_sample_size_known_values() -> None:
    """Test normal sample size with known values."""

    # Test case: effect size = 0.5, power = 0.8, should give reasonable sample size
    effect_size = torch.tensor(0.5)
    power = torch.tensor(0.8)

    sample_size_one_sided = z_test_sample_size(
        effect_size, power, alpha=0.05, alternative="larger"
    )

    # Should be reasonable sample size (between 10 and 100 for moderate effect)
    assert 10 <= float(sample_size_one_sided) <= 100

    # Test invalid alternative
    with pytest.raises(ValueError):
        z_test_sample_size(effect_size, power, alternative="invalid")

    # Test extreme values
    with pytest.raises(ValueError):
        z_test_sample_size(effect_size, torch.tensor(1.5), alpha=0.05)  # Power > 1

    with pytest.raises(ValueError):
        z_test_sample_size(effect_size, torch.tensor(-0.1), alpha=0.05)  # Power < 0


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not available")
def test_z_test_sample_size_vs_statsmodels() -> None:
    """Test against statsmodels for verification."""

    effect_size = 0.5
    power = 0.8
    alpha = 0.05

    # Test two-sided
    our_result = z_test_sample_size(
        torch.tensor(effect_size),
        torch.tensor(power),
        alpha=alpha,
        alternative="two-sided",
    )

    # Statsmodels uses normal_power for z-tests with known variance
    statsmodels_result = smp.normal_power(
        effect_size=effect_size,
        nobs=None,
        alpha=alpha,
        power=power,
        alternative="two-sided",
    )

    # Should be close (within 10% or 2 units)
    assert abs(float(our_result) - statsmodels_result) <= max(
        2, 0.1 * statsmodels_result
    )
