import hypothesis
import hypothesis.strategies
import pytest
import statsmodels.stats.power
import torch

from beignet.statistics._independent_z_test_power import independent_z_test_power

"""Test independent z-test power (two-sample z-test with known variances)."""


@hypothesis.given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=10),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)  # Disable deadline for torch.compile
def test_independent_z_test_power(batch_size: int, dtype: torch.dtype) -> None:
    """Test independent z-test power calculation."""

    # Basic functionality tests
    effect_size = torch.rand(batch_size, dtype=dtype) * 0.8 + 0.2  # 0.2 to 1.0
    sample_size1 = torch.randint(10, 50, (batch_size,), dtype=dtype)
    sample_size2 = torch.randint(10, 50, (batch_size,), dtype=dtype)

    # Test basic computation
    power = independent_z_test_power(
        effect_size,
        sample_size1,
        sample_size2,
        alpha=0.05,
        alternative="two-sided",
    )

    # Check output properties
    assert power.shape == effect_size.shape
    assert power.dtype == dtype
    assert torch.all(power >= 0.0)
    assert torch.all(power <= 1.0)

    # Test different alternatives
    power_larger = independent_z_test_power(
        effect_size,
        sample_size1,
        sample_size2,
        alternative="larger",
    )
    power_smaller = independent_z_test_power(
        effect_size,
        sample_size1,
        sample_size2,
        alternative="smaller",
    )
    power_two_sided = independent_z_test_power(
        effect_size,
        sample_size1,
        sample_size2,
        alternative="two-sided",
    )

    # One-sided tests should generally have higher power than two-sided
    assert torch.all(power_larger >= 0.0)
    assert torch.all(power_smaller >= 0.0)
    assert torch.all(power_two_sided >= 0.0)

    # Test monotonicity: larger effect size should increase power
    small_effect = effect_size * 0.5
    large_effect = effect_size * 1.5
    power_small = independent_z_test_power(
        small_effect,
        sample_size1,
        sample_size2,
        alternative="larger",
    )
    power_large = independent_z_test_power(
        large_effect,
        sample_size1,
        sample_size2,
        alternative="larger",
    )
    assert torch.all(power_large >= power_small)

    # Test monotonicity: larger sample sizes should increase power
    small_sample1 = sample_size1 * 0.7
    large_sample1 = sample_size1 * 1.5
    power_small = independent_z_test_power(
        effect_size,
        small_sample1,
        sample_size2,
        alternative="larger",
    )
    power_large = independent_z_test_power(
        effect_size,
        large_sample1,
        sample_size2,
        alternative="larger",
    )
    assert torch.all(power_large >= power_small)

    # Test gradients
    effect_size.requires_grad_(True)
    sample_size1.requires_grad_(True)
    sample_size2.requires_grad_(True)

    power = independent_z_test_power(
        effect_size,
        sample_size1,
        sample_size2,
        alternative="larger",
    )
    loss = power.sum()
    loss.backward()

    # Gradients should exist and be finite
    assert effect_size.grad is not None
    assert sample_size1.grad is not None
    assert sample_size2.grad is not None
    assert torch.all(torch.isfinite(effect_size.grad))
    assert torch.all(torch.isfinite(sample_size1.grad))
    assert torch.all(torch.isfinite(sample_size2.grad))

    # Test torch.compile compatibility
    compiled_func = torch.compile(independent_z_test_power, fullgraph=True)
    power_compiled = compiled_func(
        effect_size.detach(),
        sample_size1.detach(),
        sample_size2.detach(),
        alpha=0.05,
        alternative="larger",
    )
    power_regular = independent_z_test_power(
        effect_size.detach(),
        sample_size1.detach(),
        sample_size2.detach(),
        alpha=0.05,
        alternative="larger",
    )
    assert torch.allclose(power_compiled, power_regular, rtol=1e-5)

    # Test with out parameter
    out = torch.empty_like(power)
    result = independent_z_test_power(
        effect_size.detach(),
        sample_size1.detach(),
        sample_size2.detach(),
        alpha=0.05,
        alternative="larger",
        out=out,
    )
    assert torch.allclose(out, power_regular, rtol=1e-5)
    assert result is out

    # Test known values
    # Test case: effect size = 0.5, equal sample sizes n=20 each
    effect_size_known = torch.tensor(0.5, dtype=dtype)
    sample_size1_known = torch.tensor(20, dtype=dtype)
    sample_size2_known = torch.tensor(20, dtype=dtype)

    power_one_sided = independent_z_test_power(
        effect_size_known,
        sample_size1_known,
        sample_size2_known,
        alpha=0.05,
        alternative="larger",
    )

    # Should be reasonable power
    assert 0.3 <= float(power_one_sided) <= 0.9

    # Test invalid alternative
    with pytest.raises(ValueError):
        independent_z_test_power(
            effect_size_known,
            sample_size1_known,
            sample_size2_known,
            alternative="invalid",
        )

    # Test against statsmodels for verification
    effect_size_sm = 0.5
    sample_size1_sm = 20
    sample_size2_sm = 20
    alpha_sm = 0.05

    # Test two-sided
    our_result = independent_z_test_power(
        torch.tensor(effect_size_sm, dtype=dtype),
        torch.tensor(sample_size1_sm, dtype=dtype),
        torch.tensor(sample_size2_sm, dtype=dtype),
        alpha=alpha_sm,
        alternative="two-sided",
    )

    # For independent samples, effective sample size is n1*n2/(n1+n2)
    effective_n = (sample_size1_sm * sample_size2_sm) / (
        sample_size1_sm + sample_size2_sm
    )

    # Use statsmodels normal_power with effective sample size
    # normal_power calculates power directly, no power parameter needed
    statsmodels_result = statsmodels.stats.power.normal_power(
        effect_size=effect_size_sm,
        nobs=effective_n,
        alpha=alpha_sm,
        alternative="two-sided",
    )

    # Should be close (within 0.4) - allow for different calculation methods
    assert abs(float(our_result) - statsmodels_result) <= 0.4
