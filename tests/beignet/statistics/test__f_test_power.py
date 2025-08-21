"""Test F-test power."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from beignet.statistics._f_test_power import f_test_power

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
def test_f_test_power(batch_size: int, dtype: torch.dtype) -> None:
    """Test F-test power calculation."""

    # Basic functionality tests
    effect_size = (
        torch.rand(batch_size, dtype=dtype) * 0.3 + 0.05
    )  # 0.05 to 0.35 (Cohen's f²)
    df1 = torch.randint(1, 10, (batch_size,), dtype=dtype)
    df2 = torch.randint(20, 100, (batch_size,), dtype=dtype)

    # Test basic computation
    power = f_test_power(effect_size, df1, df2, alpha=0.05)

    # Check output properties
    assert power.shape == effect_size.shape
    assert power.dtype == dtype
    assert torch.all(power >= 0.0)
    assert torch.all(power <= 1.0)

    # Test monotonicity: larger effect size should increase power
    small_effect = effect_size * 0.5
    large_effect = effect_size * 1.5
    power_small = f_test_power(small_effect, df1, df2)
    power_large = f_test_power(large_effect, df1, df2)
    assert torch.all(power_large >= power_small)

    # Test monotonicity: larger df2 should generally increase power (more precision)
    small_df2 = torch.clamp(df2 * 0.7, min=10)
    large_df2 = df2 * 1.5
    power_small = f_test_power(effect_size, df1, small_df2)
    power_large = f_test_power(effect_size, df1, large_df2)
    assert torch.all(power_large >= power_small)

    # Test gradients
    effect_size.requires_grad_(True)
    df1.requires_grad_(True)
    df2.requires_grad_(True)

    power = f_test_power(effect_size, df1, df2)
    loss = power.sum()
    loss.backward()

    # Gradients should exist and be finite
    assert effect_size.grad is not None
    assert df1.grad is not None
    assert df2.grad is not None
    assert torch.all(torch.isfinite(effect_size.grad))
    assert torch.all(torch.isfinite(df1.grad))
    assert torch.all(torch.isfinite(df2.grad))

    # Test torch.compile compatibility
    compiled_func = torch.compile(f_test_power, fullgraph=True)
    power_compiled = compiled_func(
        effect_size.detach(), df1.detach(), df2.detach(), alpha=0.05
    )
    power_regular = f_test_power(
        effect_size.detach(), df1.detach(), df2.detach(), alpha=0.05
    )
    assert torch.allclose(power_compiled, power_regular, rtol=1e-5)

    # Test with out parameter
    out = torch.empty_like(power)
    result = f_test_power(
        effect_size.detach(), df1.detach(), df2.detach(), alpha=0.05, out=out
    )
    assert torch.allclose(out, power_regular, rtol=1e-5)
    assert result is out


def test_f_test_power_known_values() -> None:
    """Test F-test power with known values."""

    # Test case: medium effect size, moderate df
    effect_size = torch.tensor(0.15)  # Cohen's f² = 0.15 (medium effect)
    df1 = torch.tensor(3)
    df2 = torch.tensor(96)

    power = f_test_power(effect_size, df1, df2, alpha=0.05)

    # Should be reasonable power (around 0.9 for this configuration)
    assert 0.6 <= float(power) <= 1.0

    # Test edge case: zero effect size should give power ≈ alpha
    zero_effect = torch.tensor(0.0)
    power_zero = f_test_power(zero_effect, df1, df2, alpha=0.05)
    assert abs(float(power_zero) - 0.05) <= 0.03


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not available")
def test_f_test_power_vs_statsmodels() -> None:
    """Test against statsmodels for verification."""

    effect_size = 0.15  # Cohen's f²
    df1 = 3
    df2 = 96
    alpha = 0.05

    # Calculate total sample size for statsmodels
    total_n = df2 + df1 + 1

    our_result = f_test_power(
        torch.tensor(effect_size), torch.tensor(df1), torch.tensor(df2), alpha=alpha
    )

    # Use statsmodels f_test_power
    statsmodels_result = smp.f_test_power(
        effect_size=effect_size,
        df_num=df1,
        df_denom=df2,
        alpha=alpha,
        ncc=total_n,  # Total sample size
    )

    # Should be close (within 0.05)
    assert abs(float(our_result) - statsmodels_result) <= 0.05
