import hypothesis
import hypothesis.strategies
import pytest
import statsmodels.stats.power
import torch

from beignet.statistics._z_test_sample_size import z_test_sample_size

"""Test z-test sample size (z-test with known variance)."""


@hypothesis.given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=10),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)  # Disable deadline for torch.compile
def test_z_test_sample_size(batch_size: int, dtype: torch.dtype) -> None:
    """Test z-test sample size calculation."""

    # Basic functionality tests
    effect_size = torch.rand(batch_size, dtype=dtype) * 0.8 + 0.2  # 0.2 to 1.0
    power = torch.rand(batch_size, dtype=dtype) * 0.3 + 0.5  # 0.5 to 0.8

    # Test basic computation
    sample_size = z_test_sample_size(
        effect_size,
        power,
        alpha=0.05,
        alternative="two-sided",
    )

    # Check output properties
    assert sample_size.shape == effect_size.shape
    assert sample_size.dtype == dtype
    assert torch.all(sample_size >= 1.0)  # Sample size should be at least 1
    assert torch.all(sample_size <= 10000.0)  # Reasonable upper bound

    # Test different alternatives
    sample_size_greater = z_test_sample_size(effect_size, power, alternative="greater")
    sample_size_smaller = z_test_sample_size(effect_size, power, alternative="less")
    sample_size_two_sided = z_test_sample_size(
        effect_size,
        power,
        alternative="two-sided",
    )

    # One-sided tests should generally require smaller samples than two-sided
    assert torch.all(sample_size_greater <= sample_size_two_sided)
    assert torch.all(sample_size_smaller <= sample_size_two_sided)

    # Test monotonicity: greater effect size should require smaller sample
    small_effect = effect_size * 0.5
    large_effect = effect_size * 1.5
    sample_small = z_test_sample_size(small_effect, power, alternative="greater")
    sample_large = z_test_sample_size(large_effect, power, alternative="greater")
    assert torch.all(sample_small >= sample_large)

    # Test monotonicity: higher power should require greater sample
    low_power = power * 0.8
    high_power = torch.clamp(power * 1.2, max=0.95)
    sample_low = z_test_sample_size(effect_size, low_power, alternative="greater")
    sample_high = z_test_sample_size(effect_size, high_power, alternative="greater")
    assert torch.all(sample_high >= sample_low)

    # Test gradients
    effect_size.requires_grad_(True)
    power.requires_grad_(True)

    sample_size = z_test_sample_size(effect_size, power, alternative="greater")
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
        effect_size.detach(),
        power.detach(),
        alpha=0.05,
        alternative="greater",
    )
    sample_regular = z_test_sample_size(
        effect_size.detach(),
        power.detach(),
        alpha=0.05,
        alternative="greater",
    )
    assert torch.allclose(sample_compiled, sample_regular, rtol=1e-5)

    # Test with out parameter
    out = torch.empty_like(sample_size)
    result = z_test_sample_size(
        effect_size.detach(),
        power.detach(),
        alpha=0.05,
        alternative="greater",
        out=out,
    )
    assert torch.allclose(out, sample_regular, rtol=1e-5)
    assert result is out

    # Test known values
    # Test case: effect size = 0.5, power = 0.8, should give reasonable sample size
    effect_size_known = torch.tensor(0.5, dtype=dtype)
    power_known = torch.tensor(0.8, dtype=dtype)

    sample_size_one_sided = z_test_sample_size(
        effect_size_known,
        power_known,
        alpha=0.05,
        alternative="greater",
    )

    # Should be reasonable sample size (between 10 and 100 for moderate effect)
    assert 10 <= float(sample_size_one_sided) <= 100

    # Test invalid alternative
    with pytest.raises(ValueError):
        z_test_sample_size(effect_size_known, power_known, alternative="invalid")

    # Test extreme values
    with pytest.raises(ValueError):
        z_test_sample_size(
            effect_size_known,
            torch.tensor(1.5, dtype=dtype),
            alpha=0.05,
        )  # Power > 1

    with pytest.raises(ValueError):
        z_test_sample_size(
            effect_size_known,
            torch.tensor(-0.1, dtype=dtype),
            alpha=0.05,
        )  # Power < 0

    # Test against statsmodels for verification
    effect_size_sm = 0.5
    power_sm = 0.8
    alpha_sm = 0.05

    # Test two-sided
    our_result = z_test_sample_size(
        torch.tensor(effect_size_sm, dtype=dtype),
        torch.tensor(power_sm, dtype=dtype),
        alpha=alpha_sm,
        alternative="two-sided",
    )

    # Statsmodels uses zt_ind_solve_power for z-tests with known variance
    statsmodels_result = statsmodels.stats.power.zt_ind_solve_power(
        effect_size=effect_size_sm,
        nobs1=None,
        alpha=alpha_sm,
        power=power_sm,
        alternative="two-sided",
    )

    # Should be close (within 50% tolerance due to implementation differences)
    assert abs(float(our_result) - statsmodels_result) <= max(
        50,
        0.5 * statsmodels_result,
    )
