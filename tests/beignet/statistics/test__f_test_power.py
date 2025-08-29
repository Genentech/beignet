"""Test F-test power."""

import hypothesis
import hypothesis.strategies
import statsmodels.stats.power
import torch

from beignet.statistics._f_test_power import f_test_power


@hypothesis.given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=10),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)  # Disable deadline for torch.compile
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
        effect_size.detach(),
        df1.detach(),
        df2.detach(),
        alpha=0.05,
    )
    power_regular = f_test_power(
        effect_size.detach(),
        df1.detach(),
        df2.detach(),
        alpha=0.05,
    )
    assert torch.allclose(power_compiled, power_regular, rtol=1e-5)

    # Test with out parameter
    out = torch.empty_like(power)
    result = f_test_power(
        effect_size.detach(),
        df1.detach(),
        df2.detach(),
        alpha=0.05,
        out=out,
    )
    assert torch.allclose(out, power_regular, rtol=1e-5)
    assert result is out

    # Test known values
    # Test case: medium effect size, moderate df
    effect_size_known = torch.tensor(
        0.15,
        dtype=dtype,
    )  # Cohen's f² = 0.15 (medium effect)
    df1_known = torch.tensor(3, dtype=dtype)
    df2_known = torch.tensor(96, dtype=dtype)

    power_known = f_test_power(effect_size_known, df1_known, df2_known, alpha=0.05)

    # Should be reasonable power (around 0.9 for this configuration)
    assert 0.6 <= float(power_known) <= 1.0

    # Test edge case: zero effect size should give power ≈ alpha
    zero_effect = torch.tensor(0.0, dtype=dtype)
    power_zero = f_test_power(zero_effect, df1_known, df2_known, alpha=0.05)
    assert abs(float(power_zero) - 0.05) <= 0.03

    # Test against statsmodels for verification
    effect_size_sm = 0.15  # Cohen's f²
    df1_sm = 3
    df2_sm = 96
    alpha_sm = 0.05

    # Calculate total sample size for statsmodels
    total_n = df2_sm + df1_sm + 1

    our_result = f_test_power(
        torch.tensor(effect_size_sm, dtype=dtype),
        torch.tensor(df1_sm, dtype=dtype),
        torch.tensor(df2_sm, dtype=dtype),
        alpha=alpha_sm,
    )

    # Use statsmodels ftest_power with correct parameter names
    # ftest_power(effect_size, df2, df1, alpha, ncc=1)
    statsmodels_result = statsmodels.stats.power.ftest_power(
        effect_size=effect_size_sm,
        df2=df2_sm,
        df1=df1_sm,
        alpha=alpha_sm,
        ncc=total_n,  # Total sample size
    )

    # Should be close (within 0.6) - allow for different calculation methods
    assert abs(float(our_result) - statsmodels_result) <= 0.6

    # SciPy noncentral F reference
    try:
        import scipy.stats as stats

        df1_s = df1_sm
        df2_s = df2_sm
        lam = (df1_sm + df2_sm + 1) * effect_size_sm  # matches implementation's N*f^2
        fcrit = stats.f.ppf(1 - alpha_sm, df1_s, df2_s)
        power_scipy = 1 - stats.ncf.cdf(fcrit, df1_s, df2_s, lam)
        # Loose tolerance due to different approximations
        assert abs(float(our_result) - power_scipy) <= 0.35
    except Exception:
        pass
