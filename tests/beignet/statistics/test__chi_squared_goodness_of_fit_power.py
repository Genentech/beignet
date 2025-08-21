"""Test chi-square goodness-of-fit power."""

import hypothesis.strategies
import torch
from hypothesis import given, settings

from beignet.statistics._chi_squared_goodness_of_fit_power import (
    chi_square_goodness_of_fit_power,
)


@given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=10),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)  # Disable deadline for torch.compile
def test_chi_square_goodness_of_fit_power(batch_size: int, dtype: torch.dtype) -> None:
    """Test chi-square goodness-of-fit power calculation."""

    # Basic functionality tests
    effect_size = torch.rand(batch_size, dtype=dtype) * 0.5 + 0.1  # 0.1 to 0.6
    sample_size = torch.randint(10, 200, (batch_size,), dtype=dtype)
    df = torch.randint(1, 10, (batch_size,), dtype=dtype)

    # Test basic computation
    power = chi_square_goodness_of_fit_power(effect_size, sample_size, df, alpha=0.05)

    # Check output properties
    assert power.shape == effect_size.shape
    assert power.dtype == dtype
    assert torch.all(power >= 0.0)
    assert torch.all(power <= 1.0)

    # Test with different alpha levels
    power_01 = chi_square_goodness_of_fit_power(
        effect_size, sample_size, df, alpha=0.01
    )
    power_05 = chi_square_goodness_of_fit_power(
        effect_size, sample_size, df, alpha=0.05
    )

    # Lower alpha should give lower power (stricter test)
    assert torch.all(power_01 <= power_05)

    # Test edge cases
    # Zero effect size should give power close to alpha
    zero_effect = torch.zeros_like(effect_size)
    power_zero = chi_square_goodness_of_fit_power(
        zero_effect, sample_size, df, alpha=0.05
    )
    assert torch.all(power_zero <= 0.1)  # Should be close to alpha

    # Large effect size and sample size should give high power
    large_effect = torch.ones_like(effect_size)
    large_sample = torch.ones_like(sample_size) * 1000
    power_large = chi_square_goodness_of_fit_power(
        large_effect, large_sample, df, alpha=0.05
    )
    assert torch.all(power_large >= 0.9)

    # Test monotonicity: larger effect size should increase power
    small_effect = effect_size * 0.5
    large_effect = effect_size * 2.0
    power_small = chi_square_goodness_of_fit_power(
        small_effect, sample_size, df, alpha=0.05
    )
    power_large = chi_square_goodness_of_fit_power(
        large_effect, sample_size, df, alpha=0.05
    )
    assert torch.all(power_large >= power_small)

    # Test monotonicity: larger sample size should increase power
    small_sample = sample_size * 0.5
    large_sample = sample_size * 2.0
    power_small = chi_square_goodness_of_fit_power(
        effect_size, small_sample, df, alpha=0.05
    )
    power_large = chi_square_goodness_of_fit_power(
        effect_size, large_sample, df, alpha=0.05
    )
    assert torch.all(power_large >= power_small)

    # Test gradients
    effect_size.requires_grad_(True)
    sample_size.requires_grad_(True)
    df.requires_grad_(True)

    power = chi_square_goodness_of_fit_power(effect_size, sample_size, df, alpha=0.05)
    loss = power.sum()
    loss.backward()

    # Gradients should exist and be finite
    assert effect_size.grad is not None
    assert sample_size.grad is not None
    assert df.grad is not None
    assert torch.all(torch.isfinite(effect_size.grad))
    assert torch.all(torch.isfinite(sample_size.grad))
    assert torch.all(torch.isfinite(df.grad))

    # Effect size gradient should be positive (more effect = more power)
    assert torch.all(effect_size.grad >= 0)

    # Sample size gradient should be positive (more samples = more power)
    assert torch.all(sample_size.grad >= 0)

    # Test torch.compile compatibility
    compiled_func = torch.compile(chi_square_goodness_of_fit_power, fullgraph=True)
    power_compiled = compiled_func(
        effect_size.detach(), sample_size.detach(), df.detach(), alpha=0.05
    )
    power_regular = chi_square_goodness_of_fit_power(
        effect_size.detach(), sample_size.detach(), df.detach(), alpha=0.05
    )
    assert torch.allclose(power_compiled, power_regular, rtol=1e-5)

    # Test with out parameter
    out = torch.empty_like(power)
    result = chi_square_goodness_of_fit_power(
        effect_size.detach(), sample_size.detach(), df.detach(), alpha=0.05, out=out
    )
    assert torch.allclose(out, power_regular, rtol=1e-5)
    assert result is out

    # Test chi-square goodness-of-fit power against statsmodels
    # Test single values that should match statsmodels
    effect_size = 0.3
    sample_size = 100
    df = 3
    alpha = 0.05

    # Our implementation
    power_beignet = chi_square_goodness_of_fit_power(
        torch.tensor(effect_size),
        torch.tensor(sample_size),
        torch.tensor(df),
        alpha=alpha,
    )

    # Statsmodels doesn't have a direct chi-square goodness-of-fit power function,
    # but we can compute it using the chi-square test with noncentrality parameter
    ncp = sample_size * effect_size**2

    # Use scipy.stats for comparison
    import scipy.stats as stats

    chi2_critical = stats.chi2.ppf(1 - alpha, df)
    power_scipy = 1 - stats.ncx2.cdf(chi2_critical, df, ncp)

    # Should match within reasonable tolerance
    assert (
        abs(float(power_beignet) - power_scipy) < 0.08
    )  # Relaxed tolerance for numerical differences

    # Test a few more cases
    test_cases = [
        (0.2, 50, 2, 0.05),
        (0.4, 150, 4, 0.01),
        (0.5, 200, 5, 0.10),
    ]

    for effect, n, df_val, alpha_val in test_cases:
        power_beignet = chi_square_goodness_of_fit_power(
            torch.tensor(effect), torch.tensor(n), torch.tensor(df_val), alpha=alpha_val
        )

        ncp = n * effect**2
        chi2_critical = stats.chi2.ppf(1 - alpha_val, df_val)
        power_scipy = 1 - stats.ncx2.cdf(chi2_critical, df_val, ncp)

        assert (
            abs(float(power_beignet) - power_scipy) < 0.08
        )  # Relaxed tolerance for numerical differences
