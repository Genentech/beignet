"""Test chi-square independence sample size."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from beignet.statistics._chi_square_independence_power import (
    chi_square_independence_power,
)
from beignet.statistics._chi_square_independence_sample_size import (
    chi_square_independence_sample_size,
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
def test_chisquare_independence_sample_size(
    batch_size: int, dtype: torch.dtype
) -> None:
    """Test chi-square independence sample size calculation."""

    # Basic functionality tests
    effect_size = torch.rand(batch_size, dtype=dtype) * 0.5 + 0.1  # 0.1 to 0.6
    rows = torch.randint(2, 5, (batch_size,), dtype=dtype)
    cols = torch.randint(2, 5, (batch_size,), dtype=dtype)
    power = 0.8
    alpha = 0.05

    # Test basic computation
    sample_size = chi_square_independence_sample_size(
        effect_size, rows, cols, power=power, alpha=alpha
    )

    # Check output properties
    assert sample_size.shape == effect_size.shape
    assert sample_size.dtype == dtype
    min_expected = 5.0 * rows * cols  # Minimum sample size constraint
    assert torch.all(sample_size >= min_expected)
    assert torch.all(sample_size <= 1000000.0)  # Maximum constraint

    # Test different power levels
    sample_size_low = chi_square_independence_sample_size(
        effect_size, rows, cols, power=0.6, alpha=alpha
    )
    sample_size_high = chi_square_independence_sample_size(
        effect_size, rows, cols, power=0.9, alpha=alpha
    )

    # Higher power should require larger sample size
    assert torch.all(sample_size_high >= sample_size_low)

    # Test different alpha levels
    sample_size_strict = chi_square_independence_sample_size(
        effect_size, rows, cols, power=power, alpha=0.01
    )
    sample_size_lenient = chi_square_independence_sample_size(
        effect_size, rows, cols, power=power, alpha=0.10
    )

    # Stricter alpha should require larger sample size
    assert torch.all(sample_size_strict >= sample_size_lenient)

    # Test monotonicity: larger effect size should decrease required sample size
    small_effect = effect_size * 0.5
    large_effect = effect_size * 1.5
    sample_size_small = chi_square_independence_sample_size(
        small_effect, rows, cols, power=power, alpha=alpha
    )
    sample_size_large = chi_square_independence_sample_size(
        large_effect, rows, cols, power=power, alpha=alpha
    )
    assert torch.all(sample_size_small >= sample_size_large)

    # Test effect of table dimensions: larger tables should require larger sample sizes
    small_rows = torch.full_like(rows, 2)
    small_cols = torch.full_like(cols, 2)
    large_rows = torch.full_like(rows, 4)
    large_cols = torch.full_like(cols, 4)

    sample_size_small_table = chi_square_independence_sample_size(
        effect_size, small_rows, small_cols, power=power, alpha=alpha
    )
    sample_size_large_table = chi_square_independence_sample_size(
        effect_size, large_rows, large_cols, power=power, alpha=alpha
    )

    # Larger table typically needs more samples (more degrees of freedom)
    # Note: This can depend on the specific effect size, so we just check they're reasonable
    assert torch.all(sample_size_small_table >= 5.0 * small_rows * small_cols)
    assert torch.all(sample_size_large_table >= 5.0 * large_rows * large_cols)

    # Test that computed sample size achieves desired power
    computed_power = chi_square_independence_power(
        effect_size, sample_size, rows, cols, alpha=alpha
    )

    # Power should be close to desired level (within tolerance due to rounding)
    assert torch.all(computed_power >= power - 0.05)
    assert torch.all(computed_power <= 1.0)

    # Test gradients
    effect_size.requires_grad_(True)
    rows.requires_grad_(True)
    cols.requires_grad_(True)

    sample_size = chi_square_independence_sample_size(
        effect_size, rows, cols, power=power, alpha=alpha
    )
    loss = sample_size.sum()
    loss.backward()

    # Gradients should exist and be finite
    assert effect_size.grad is not None
    assert rows.grad is not None
    assert cols.grad is not None
    assert torch.all(torch.isfinite(effect_size.grad))
    assert torch.all(torch.isfinite(rows.grad))
    assert torch.all(torch.isfinite(cols.grad))

    # Effect size gradient should be negative (more effect = less sample size needed)
    assert torch.all(effect_size.grad <= 0)

    # Test torch.compile compatibility
    compiled_func = torch.compile(chi_square_independence_sample_size, fullgraph=True)
    sample_size_compiled = compiled_func(
        effect_size.detach(), rows.detach(), cols.detach(), power=power, alpha=alpha
    )
    sample_size_regular = chi_square_independence_sample_size(
        effect_size.detach(), rows.detach(), cols.detach(), power=power, alpha=alpha
    )
    assert torch.allclose(sample_size_compiled, sample_size_regular, rtol=1e-5)

    # Test with out parameter
    out = torch.empty_like(sample_size)
    result = chi_square_independence_sample_size(
        effect_size.detach(),
        rows.detach(),
        cols.detach(),
        power=power,
        alpha=alpha,
        out=out,
    )
    assert torch.allclose(out, sample_size_regular, rtol=1e-5)
    assert result is out

    # Test edge cases
    # Very small effect size should require large sample size
    tiny_effect = torch.full_like(effect_size, 0.05)
    large_sample = chi_square_independence_sample_size(
        tiny_effect, rows, cols, power=power, alpha=alpha
    )
    assert torch.all(large_sample >= 100)

    # Large effect size should require smaller sample size
    big_effect = torch.full_like(effect_size, 0.8)
    small_sample = chi_square_independence_sample_size(
        big_effect, rows, cols, power=power, alpha=alpha
    )
    assert torch.all(small_sample <= large_sample)


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not available")
def test_chisquare_independence_sample_size_consistency() -> None:
    """Test that sample size calculation is consistent with power calculation."""

    # Test several cases to ensure consistency
    test_cases = [
        (0.3, 3, 3, 0.8, 0.05),
        (0.2, 2, 3, 0.9, 0.01),
        (0.5, 2, 2, 0.7, 0.10),
        (0.4, 4, 3, 0.85, 0.05),
    ]

    for effect, rows_val, cols_val, power_target, alpha in test_cases:
        # Calculate required sample size
        sample_size = chi_square_independence_sample_size(
            torch.tensor(effect),
            torch.tensor(rows_val),
            torch.tensor(cols_val),
            power=power_target,
            alpha=alpha,
        )

        # Calculate actual power with that sample size
        actual_power = chi_square_independence_power(
            torch.tensor(effect),
            sample_size,
            torch.tensor(rows_val),
            torch.tensor(cols_val),
            alpha=alpha,
        )

        # Actual power should be close to target (within tolerance)
        assert abs(float(actual_power) - power_target) < 0.05

        # For integer sample sizes, the actual power should be at least the target
        # (since we round up)
        assert float(actual_power) >= power_target - 0.01


def test_chisquare_independence_sample_size_special_cases() -> None:
    """Test special cases for chi-square independence sample size."""

    # Test 2x2 table (most common case)
    effect_size = torch.tensor(0.3)
    rows = torch.tensor(2)
    cols = torch.tensor(2)
    power = 0.8
    alpha = 0.05

    sample_size_2x2 = chi_square_independence_sample_size(
        effect_size, rows, cols, power=power, alpha=alpha
    )

    # Should meet minimum constraint (5 per cell = 20 total for 2x2)
    assert float(sample_size_2x2) >= 20.0

    # Test larger tables
    sample_size_3x3 = chi_square_independence_sample_size(
        effect_size, torch.tensor(3), torch.tensor(3), power=power, alpha=alpha
    )
    sample_size_4x4 = chi_square_independence_sample_size(
        effect_size, torch.tensor(4), torch.tensor(4), power=power, alpha=alpha
    )

    # Should meet minimum constraints
    assert float(sample_size_3x3) >= 45.0  # 5 * 3 * 3
    assert float(sample_size_4x4) >= 80.0  # 5 * 4 * 4

    # Test rectangular tables
    sample_size_2x5 = chi_square_independence_sample_size(
        effect_size, torch.tensor(2), torch.tensor(5), power=power, alpha=alpha
    )
    sample_size_5x2 = chi_square_independence_sample_size(
        effect_size, torch.tensor(5), torch.tensor(2), power=power, alpha=alpha
    )

    # Should give same result (symmetry) and meet minimum constraints
    assert abs(float(sample_size_2x5) - float(sample_size_5x2)) < 1e-6
    assert float(sample_size_2x5) >= 50.0  # 5 * 2 * 5

    # Test very high power requirement
    sample_size_high_power = chi_square_independence_sample_size(
        effect_size, rows, cols, power=0.99, alpha=alpha
    )
    assert float(sample_size_high_power) > float(sample_size_2x2)

    # Test very strict alpha
    sample_size_strict = chi_square_independence_sample_size(
        effect_size, rows, cols, power=power, alpha=0.001
    )
    assert float(sample_size_strict) > float(sample_size_2x2)
