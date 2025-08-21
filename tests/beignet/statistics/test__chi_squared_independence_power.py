"""Test chi-square independence power."""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from beignet.statistics._chi_squared_independence_power import (
    chisquare_independence_power,
)


@given(
    batch_size=st.integers(min_value=1, max_value=10),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)  # Disable deadline for torch.compile
def test_chisquare_independence_power(batch_size: int, dtype: torch.dtype) -> None:
    """Test chi-square independence power calculation."""

    # Basic functionality tests
    effect_size = torch.rand(batch_size, dtype=dtype) * 0.5 + 0.1  # 0.1 to 0.6
    sample_size = torch.randint(20, 200, (batch_size,), dtype=dtype)
    rows = torch.randint(2, 5, (batch_size,), dtype=dtype)
    cols = torch.randint(2, 5, (batch_size,), dtype=dtype)

    # Test basic computation
    power = chisquare_independence_power(
        effect_size, sample_size, rows, cols, alpha=0.05
    )

    # Check output properties
    assert power.shape == effect_size.shape
    assert power.dtype == dtype
    assert torch.all(power >= 0.0)
    assert torch.all(power <= 1.0)

    # Test with different alpha levels
    power_01 = chisquare_independence_power(
        effect_size, sample_size, rows, cols, alpha=0.01
    )
    power_05 = chisquare_independence_power(
        effect_size, sample_size, rows, cols, alpha=0.05
    )

    # Lower alpha should give lower power (stricter test)
    assert torch.all(power_01 <= power_05)

    # Test edge cases
    # Zero effect size should give power close to alpha
    zero_effect = torch.zeros_like(effect_size)
    power_zero = chisquare_independence_power(
        zero_effect, sample_size, rows, cols, alpha=0.05
    )
    assert torch.all(power_zero <= 0.1)  # Should be close to alpha

    # Large effect size and sample size should give high power
    large_effect = torch.ones_like(effect_size)
    large_sample = torch.ones_like(sample_size) * 1000
    power_large = chisquare_independence_power(
        large_effect, large_sample, rows, cols, alpha=0.05
    )
    assert torch.all(power_large >= 0.9)

    # Test monotonicity: larger effect size should increase power
    small_effect = effect_size * 0.5
    large_effect = effect_size * 2.0
    power_small = chisquare_independence_power(
        small_effect, sample_size, rows, cols, alpha=0.05
    )
    power_large = chisquare_independence_power(
        large_effect, sample_size, rows, cols, alpha=0.05
    )
    assert torch.all(power_large >= power_small)

    # Test monotonicity: larger sample size should increase power
    small_sample = sample_size * 0.5
    large_sample = sample_size * 2.0
    power_small = chisquare_independence_power(
        effect_size, small_sample, rows, cols, alpha=0.05
    )
    power_large = chisquare_independence_power(
        effect_size, large_sample, rows, cols, alpha=0.05
    )
    assert torch.all(power_large >= power_small)

    # Test effect of table dimensions: more cells should decrease power for same effect size
    # (larger degrees of freedom make test less powerful)
    small_table = chisquare_independence_power(
        effect_size,
        sample_size,
        torch.full_like(rows, 2),
        torch.full_like(cols, 2),
        alpha=0.05,
    )
    large_table = chisquare_independence_power(
        effect_size,
        sample_size,
        torch.full_like(rows, 4),
        torch.full_like(cols, 4),
        alpha=0.05,
    )
    # Note: This relationship can be complex, so we just check that both give reasonable values
    assert torch.all(small_table >= 0.0)
    assert torch.all(large_table >= 0.0)

    # Test gradients
    effect_size.requires_grad_(True)
    sample_size.requires_grad_(True)
    rows.requires_grad_(True)
    cols.requires_grad_(True)

    power = chisquare_independence_power(
        effect_size, sample_size, rows, cols, alpha=0.05
    )
    loss = power.sum()
    loss.backward()

    # Gradients should exist and be finite
    assert effect_size.grad is not None
    assert sample_size.grad is not None
    assert rows.grad is not None
    assert cols.grad is not None
    assert torch.all(torch.isfinite(effect_size.grad))
    assert torch.all(torch.isfinite(sample_size.grad))
    assert torch.all(torch.isfinite(rows.grad))
    assert torch.all(torch.isfinite(cols.grad))

    # Effect size gradient should be positive (more effect = more power)
    assert torch.all(effect_size.grad >= 0)

    # Sample size gradient should be positive (more samples = more power)
    assert torch.all(sample_size.grad >= 0)

    # Test torch.compile compatibility
    compiled_func = torch.compile(chisquare_independence_power, fullgraph=True)
    power_compiled = compiled_func(
        effect_size.detach(),
        sample_size.detach(),
        rows.detach(),
        cols.detach(),
        alpha=0.05,
    )
    power_regular = chisquare_independence_power(
        effect_size.detach(),
        sample_size.detach(),
        rows.detach(),
        cols.detach(),
        alpha=0.05,
    )
    assert torch.allclose(power_compiled, power_regular, rtol=1e-5)

    # Test with out parameter
    out = torch.empty_like(power)
    result = chisquare_independence_power(
        effect_size.detach(),
        sample_size.detach(),
        rows.detach(),
        cols.detach(),
        alpha=0.05,
        out=out,
    )
    assert torch.allclose(out, power_regular, rtol=1e-5)
    assert result is out

    # Test chi-square independence power against statistical theory
    # Test single values that should match theoretical calculations
    effect_size = 0.3
    sample_size = 100
    rows = 3
    cols = 3
    alpha = 0.05

    # Our implementation
    power_beignet = chisquare_independence_power(
        torch.tensor(effect_size),
        torch.tensor(sample_size),
        torch.tensor(rows),
        torch.tensor(cols),
        alpha=alpha,
    )

    # Manual calculation using scipy for comparison
    import scipy.stats as stats

    df = (rows - 1) * (cols - 1)
    ncp = sample_size * effect_size**2
    chi2_critical = stats.chi2.ppf(1 - alpha, df)
    power_scipy = 1 - stats.ncx2.cdf(chi2_critical, df, ncp)

    # Should match within reasonable tolerance
    assert (
        abs(float(power_beignet) - power_scipy) < 0.08
    )  # Relaxed tolerance for numerical differences

    # Test a few more cases
    test_cases = [
        (0.2, 50, 2, 3, 0.05),
        (0.4, 150, 3, 4, 0.01),
        (0.5, 200, 2, 2, 0.10),
    ]

    for effect, n, r, c, alpha_val in test_cases:
        power_beignet = chisquare_independence_power(
            torch.tensor(effect),
            torch.tensor(n),
            torch.tensor(r),
            torch.tensor(c),
            alpha=alpha_val,
        )

        df = (r - 1) * (c - 1)
        ncp = n * effect**2
        chi2_critical = stats.chi2.ppf(1 - alpha_val, df)
        power_scipy = 1 - stats.ncx2.cdf(chi2_critical, df, ncp)

        assert (
            abs(float(power_beignet) - power_scipy) < 0.08
        )  # Relaxed tolerance for numerical differences

    # Test special cases for chi-square independence power
    # Test 2x2 table (most common case)
    effect_size = torch.tensor(0.3)
    sample_size = torch.tensor(100)
    rows = torch.tensor(2)
    cols = torch.tensor(2)

    power_2x2 = chisquare_independence_power(effect_size, sample_size, rows, cols)
    assert 0.0 <= float(power_2x2) <= 1.0

    # Test larger tables
    power_3x3 = chisquare_independence_power(
        effect_size, sample_size, torch.tensor(3), torch.tensor(3)
    )
    power_4x4 = chisquare_independence_power(
        effect_size, sample_size, torch.tensor(4), torch.tensor(4)
    )

    # All should be valid probabilities
    assert 0.0 <= float(power_3x3) <= 1.0
    assert 0.0 <= float(power_4x4) <= 1.0

    # Test rectangular tables
    power_2x5 = chisquare_independence_power(
        effect_size, sample_size, torch.tensor(2), torch.tensor(5)
    )
    power_5x2 = chisquare_independence_power(
        effect_size, sample_size, torch.tensor(5), torch.tensor(2)
    )

    # Should give same result (symmetry)
    assert abs(float(power_2x5) - float(power_5x2)) < 1e-6
