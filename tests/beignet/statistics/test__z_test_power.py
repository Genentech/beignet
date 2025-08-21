import hypothesis
import hypothesis.strategies
import pytest
import torch

from beignet.statistics._z_test_power import z_test_power

"""Test z-test power (z-test with known variance)."""


@hypothesis.given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=10),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)  # Disable deadline for torch.compile
def test_z_test_power(batch_size: int, dtype: torch.dtype) -> None:
    """Test z-test power calculation."""

    # Basic functionality tests
    effect_size = torch.rand(batch_size, dtype=dtype) * 0.8 + 0.2  # 0.2 to 1.0
    sample_size = torch.randint(10, 100, (batch_size,), dtype=dtype)

    # Test basic computation
    power = z_test_power(effect_size, sample_size, alpha=0.05, alternative="two-sided")

    # Check output properties
    assert power.shape == effect_size.shape
    assert power.dtype == dtype
    assert torch.all(power >= 0.0)
    assert torch.all(power <= 1.0)

    # Test different alternatives
    power_larger = z_test_power(effect_size, sample_size, alternative="larger")
    power_smaller = z_test_power(effect_size, sample_size, alternative="smaller")
    power_two_sided = z_test_power(effect_size, sample_size, alternative="two-sided")

    # One-sided tests should generally have higher power than two-sided
    assert torch.all(power_larger >= 0.0)
    assert torch.all(power_smaller >= 0.0)
    assert torch.all(power_two_sided >= 0.0)

    # Test monotonicity: larger effect size should increase power
    small_effect = effect_size * 0.5
    large_effect = effect_size * 1.5
    power_small = z_test_power(small_effect, sample_size, alternative="larger")
    power_large = z_test_power(large_effect, sample_size, alternative="larger")
    assert torch.all(power_large >= power_small)

    # Test monotonicity: larger sample size should increase power
    small_sample = sample_size * 0.7
    large_sample = sample_size * 1.5
    power_small = z_test_power(effect_size, small_sample, alternative="larger")
    power_large = z_test_power(effect_size, large_sample, alternative="larger")
    assert torch.all(power_large >= power_small)

    # Test gradients
    effect_size.requires_grad_(True)
    sample_size.requires_grad_(True)

    power = z_test_power(effect_size, sample_size, alternative="larger")
    loss = power.sum()
    loss.backward()

    # Gradients should exist and be finite
    assert effect_size.grad is not None
    assert sample_size.grad is not None
    assert torch.all(torch.isfinite(effect_size.grad))
    assert torch.all(torch.isfinite(sample_size.grad))

    # Test torch.compile compatibility
    compiled_func = torch.compile(z_test_power, fullgraph=True)
    power_compiled = compiled_func(
        effect_size.detach(), sample_size.detach(), alpha=0.05, alternative="larger"
    )
    power_regular = z_test_power(
        effect_size.detach(), sample_size.detach(), alpha=0.05, alternative="larger"
    )
    assert torch.allclose(power_compiled, power_regular, rtol=1e-5)

    # Test with out parameter
    out = torch.empty_like(power)
    result = z_test_power(
        effect_size.detach(),
        sample_size.detach(),
        alpha=0.05,
        alternative="larger",
        out=out,
    )
    assert torch.allclose(out, power_regular, rtol=1e-5)
    assert result is out

    # Test known values
    # Test case: effect size = 0.5, n = 16, one-sided should give reasonable power
    effect_size_known = torch.tensor(0.5, dtype=dtype)
    sample_size_known = torch.tensor(16, dtype=dtype)

    power_one_sided = z_test_power(
        effect_size_known, sample_size_known, alpha=0.05, alternative="larger"
    )

    # Should be reasonable power, not too high or too low
    assert 0.5 <= float(power_one_sided) <= 0.95

    # Test invalid alternative
    with pytest.raises(ValueError):
        z_test_power(effect_size_known, sample_size_known, alternative="invalid")
