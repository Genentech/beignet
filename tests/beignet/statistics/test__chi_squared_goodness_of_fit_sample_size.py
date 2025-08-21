"""Test chi-square goodness-of-fit sample size."""

import hypothesis.strategies
import torch
from hypothesis import given, settings

from beignet.statistics._chi_squared_goodness_of_fit_power import (
    chi_square_goodness_of_fit_power,
)
from beignet.statistics._chi_squared_goodness_of_fit_sample_size import (
    chi_square_goodness_of_fit_power_sample_size,
)


@given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=10),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)  # Disable deadline for torch.compile
def test_chi_square_goodness_of_fit_power_sample_size(
    batch_size: int, dtype: torch.dtype
) -> None:
    """Test chi-square goodness-of-fit sample size calculation."""

    # Basic functionality tests
    effect_size = torch.rand(batch_size, dtype=dtype) * 0.5 + 0.1  # 0.1 to 0.6
    df = torch.randint(1, 10, (batch_size,), dtype=dtype)
    power = 0.8
    alpha = 0.05

    # Test basic computation
    sample_size = chi_square_goodness_of_fit_power_sample_size(
        effect_size, df, power=power, alpha=alpha
    )

    # Check output properties
    assert sample_size.shape == effect_size.shape
    assert sample_size.dtype == dtype
    assert torch.all(sample_size >= 5.0)  # Minimum for chi-square validity
    assert torch.all(sample_size <= 1000000.0)  # Maximum constraint

    # Test different power levels
    sample_size_low = chi_square_goodness_of_fit_power_sample_size(
        effect_size, df, power=0.6, alpha=alpha
    )
    sample_size_high = chi_square_goodness_of_fit_power_sample_size(
        effect_size, df, power=0.9, alpha=alpha
    )

    # Higher power should require larger sample size
    assert torch.all(sample_size_high >= sample_size_low)

    # Test different alpha levels
    sample_size_strict = chi_square_goodness_of_fit_power_sample_size(
        effect_size, df, power=power, alpha=0.01
    )
    sample_size_lenient = chi_square_goodness_of_fit_power_sample_size(
        effect_size, df, power=power, alpha=0.10
    )

    # Stricter alpha should require larger sample size
    assert torch.all(sample_size_strict >= sample_size_lenient)

    # Test monotonicity: larger effect size should decrease required sample size
    small_effect = effect_size * 0.5
    large_effect = effect_size * 1.5
    sample_size_small = chi_square_goodness_of_fit_power_sample_size(
        small_effect, df, power=power, alpha=alpha
    )
    sample_size_large = chi_square_goodness_of_fit_power_sample_size(
        large_effect, df, power=power, alpha=alpha
    )
    assert torch.all(sample_size_small >= sample_size_large)

    # Test that computed sample size achieves desired power
    computed_power = chi_square_goodness_of_fit_power(
        effect_size, sample_size, df, alpha=alpha
    )

    # Power should be close to desired level (within tolerance due to rounding)
    assert torch.all(computed_power >= power - 0.05)
    assert torch.all(computed_power <= 1.0)

    # Test gradients
    effect_size.requires_grad_(True)
    df.requires_grad_(True)

    sample_size = chi_square_goodness_of_fit_power_sample_size(
        effect_size, df, power=power, alpha=alpha
    )
    loss = sample_size.sum()
    loss.backward()

    # Gradients should exist and be finite
    assert effect_size.grad is not None
    assert df.grad is not None
    assert torch.all(torch.isfinite(effect_size.grad))
    assert torch.all(torch.isfinite(df.grad))

    # Effect size gradient should be negative (more effect = less sample size needed)
    assert torch.all(effect_size.grad <= 0)

    # Test torch.compile compatibility
    compiled_func = torch.compile(
        chi_square_goodness_of_fit_power_sample_size, fullgraph=True
    )
    sample_size_compiled = compiled_func(
        effect_size.detach(), df.detach(), power=power, alpha=alpha
    )
    sample_size_regular = chi_square_goodness_of_fit_power_sample_size(
        effect_size.detach(), df.detach(), power=power, alpha=alpha
    )
    assert torch.allclose(sample_size_compiled, sample_size_regular, rtol=1e-5)

    # Test with out parameter
    out = torch.empty_like(sample_size)
    result = chi_square_goodness_of_fit_power_sample_size(
        effect_size.detach(), df.detach(), power=power, alpha=alpha, out=out
    )
    assert torch.allclose(out, sample_size_regular, rtol=1e-5)
    assert result is out

    # Test edge cases
    # Very small effect size should require large sample size
    tiny_effect = torch.full_like(effect_size, 0.05)
    large_sample = chi_square_goodness_of_fit_power_sample_size(
        tiny_effect, df, power=power, alpha=alpha
    )
    assert torch.all(large_sample >= 100)

    # Large effect size should require smaller sample size
    big_effect = torch.full_like(effect_size, 0.8)
    small_sample = chi_square_goodness_of_fit_power_sample_size(
        big_effect, df, power=power, alpha=alpha
    )
    assert torch.all(small_sample <= large_sample)

    # Test that sample size calculation is consistent with power calculation
    # Test several cases to ensure consistency
    test_cases = [
        (0.3, 3, 0.8, 0.05),
        (0.2, 2, 0.9, 0.01),
        (0.5, 5, 0.7, 0.10),
        (0.4, 4, 0.85, 0.05),
    ]

    for effect, df_val, power_target, alpha in test_cases:
        # Calculate required sample size
        sample_size = chi_square_goodness_of_fit_power_sample_size(
            torch.tensor(effect), torch.tensor(df_val), power=power_target, alpha=alpha
        )

        # Calculate actual power with that sample size
        actual_power = chi_square_goodness_of_fit_power(
            torch.tensor(effect), sample_size, torch.tensor(df_val), alpha=alpha
        )

        # Actual power should be close to target (within tolerance)
        assert abs(float(actual_power) - power_target) < 0.05

        # For integer sample sizes, the actual power should be at least the target
        # (since we round up)
        assert float(actual_power) >= power_target - 0.01
