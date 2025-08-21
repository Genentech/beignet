"""Test F-test sample size."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from beignet._f_test_sample_size import f_test_sample_size

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
def test_f_test_sample_size(batch_size: int, dtype: torch.dtype) -> None:
    """Test F-test sample size calculation."""

    # Basic functionality tests
    effect_size = (
        torch.rand(batch_size, dtype=dtype) * 0.3 + 0.05
    )  # 0.05 to 0.35 (Cohen's f²)
    df1 = torch.randint(1, 10, (batch_size,), dtype=dtype)
    power = torch.rand(batch_size, dtype=dtype) * 0.3 + 0.5  # 0.5 to 0.8

    # Test basic computation
    sample_size = f_test_sample_size(effect_size, df1, power, alpha=0.05)

    # Check output properties
    assert sample_size.shape == effect_size.shape
    assert sample_size.dtype == dtype
    assert torch.all(sample_size >= 10.0)  # Sample size should be reasonable
    assert torch.all(sample_size <= 10000.0)  # Reasonable upper bound

    # Test monotonicity: larger effect size should require smaller sample
    small_effect = effect_size * 0.5
    large_effect = effect_size * 1.5
    sample_small = f_test_sample_size(small_effect, df1, power)
    sample_large = f_test_sample_size(large_effect, df1, power)
    assert torch.all(sample_small >= sample_large)

    # Test monotonicity: higher power should require larger sample
    low_power = power * 0.8
    high_power = torch.clamp(power * 1.2, max=0.95)
    sample_low = f_test_sample_size(effect_size, df1, low_power)
    sample_high = f_test_sample_size(effect_size, df1, high_power)
    assert torch.all(sample_high >= sample_low)

    # Test gradients
    effect_size.requires_grad_(True)
    df1.requires_grad_(True)
    power.requires_grad_(True)

    sample_size = f_test_sample_size(effect_size, df1, power)
    loss = sample_size.sum()
    loss.backward()

    # Gradients should exist and be finite
    assert effect_size.grad is not None
    assert df1.grad is not None
    assert power.grad is not None
    assert torch.all(torch.isfinite(effect_size.grad))
    assert torch.all(torch.isfinite(df1.grad))
    assert torch.all(torch.isfinite(power.grad))

    # Test torch.compile compatibility
    compiled_func = torch.compile(f_test_sample_size, fullgraph=True)
    sample_compiled = compiled_func(
        effect_size.detach(), df1.detach(), power.detach(), alpha=0.05
    )
    sample_regular = f_test_sample_size(
        effect_size.detach(), df1.detach(), power.detach(), alpha=0.05
    )
    assert torch.allclose(sample_compiled, sample_regular, rtol=1e-5)

    # Test with out parameter
    out = torch.empty_like(sample_size)
    result = f_test_sample_size(
        effect_size.detach(), df1.detach(), power.detach(), alpha=0.05, out=out
    )
    assert torch.allclose(out, sample_regular, rtol=1e-5)
    assert result is out


def test_f_test_sample_size_known_values() -> None:
    """Test F-test sample size with known values."""

    # Test case: medium effect size, power = 0.8
    effect_size = torch.tensor(0.15)  # Cohen's f² = 0.15 (medium effect)
    df1 = torch.tensor(3)
    power = torch.tensor(0.8)

    sample_size = f_test_sample_size(effect_size, df1, power, alpha=0.05)

    # Should be reasonable sample size (around 100 for this configuration)
    assert 50 <= float(sample_size) <= 200

    # Test extreme values
    with pytest.raises(ValueError):
        f_test_sample_size(effect_size, df1, torch.tensor(1.5), alpha=0.05)  # Power > 1

    with pytest.raises(ValueError):
        f_test_sample_size(
            effect_size, df1, torch.tensor(-0.1), alpha=0.05
        )  # Power < 0


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not available")
def test_f_test_sample_size_vs_statsmodels() -> None:
    """Test against statsmodels for verification."""

    effect_size = 0.15  # Cohen's f²
    df1 = 3
    power = 0.8
    alpha = 0.05

    our_result = f_test_sample_size(
        torch.tensor(effect_size), torch.tensor(df1), torch.tensor(power), alpha=alpha
    )

    # Use statsmodels ftest_power to solve for sample size
    statsmodels_result = smp.ftest_power(
        effect_size=effect_size,
        df_num=df1,
        df_denom=None,  # Will be computed
        alpha=alpha,
        power=power,
        ncc=None,  # Total sample size to be determined
    )

    # Should be close (within 15% or 10 units)
    assert abs(float(our_result) - statsmodels_result) <= max(
        10, 0.15 * statsmodels_result
    )
