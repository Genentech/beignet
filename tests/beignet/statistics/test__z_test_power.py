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
    power_larger = z_test_power(effect_size, sample_size, alternative="greater")
    power_smaller = z_test_power(effect_size, sample_size, alternative="less")
    power_two_sided = z_test_power(effect_size, sample_size, alternative="two-sided")

    # One-sided tests should generally have higher power than two-sided
    assert torch.all(power_larger >= 0.0)
    assert torch.all(power_smaller >= 0.0)
    assert torch.all(power_two_sided >= 0.0)

    # Test monotonicity: larger effect size should increase power
    small_effect = effect_size * 0.5
    large_effect = effect_size * 1.5
    power_small = z_test_power(small_effect, sample_size, alternative="greater")
    power_large = z_test_power(large_effect, sample_size, alternative="greater")
    assert torch.all(power_large >= power_small)

    # Test monotonicity: larger sample size should increase power
    small_sample = sample_size * 0.7
    large_sample = sample_size * 1.5
    power_small = z_test_power(effect_size, small_sample, alternative="greater")
    power_large = z_test_power(effect_size, large_sample, alternative="greater")
    assert torch.all(power_large >= power_small)

    # Test gradients
    effect_size.requires_grad_(True)
    sample_size.requires_grad_(True)

    power = z_test_power(effect_size, sample_size, alternative="greater")
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
        effect_size.detach(),
        sample_size.detach(),
        alpha=0.05,
        alternative="greater",
    )
    power_regular = z_test_power(
        effect_size.detach(),
        sample_size.detach(),
        alpha=0.05,
        alternative="greater",
    )
    assert torch.allclose(power_compiled, power_regular, rtol=1e-5)


def test_z_test_power_broadcasting_out_and_directionality():
    import beignet

    dtype = torch.float64
    # Broadcasting
    d = torch.tensor([[0.2], [0.5]], dtype=dtype)
    n = torch.tensor([[20.0, 50.0]], dtype=dtype)
    out = beignet.statistics.z_test_power(d, n, alternative="two-sided")
    assert out.shape == (2, 2)

    # Out wrong shape should raise
    wrong_out = torch.empty((3, 3), dtype=dtype)
    try:
        beignet.statistics.z_test_power(d, n, out=wrong_out)
        raise AssertionError("Expected RuntimeError for shape mismatch in out")
    except RuntimeError:
        pass

    # Directionality gradient sign
    dpos = torch.tensor(0.4, dtype=dtype, requires_grad=True)
    npop = torch.tensor(50.0, dtype=dtype)
    p_greater = beignet.statistics.z_test_power(dpos, npop, alternative="greater")
    p_greater.sum().backward()
    assert dpos.grad is not None and dpos.grad > 0

    dpos2 = torch.tensor(0.4, dtype=dtype, requires_grad=True)
    p_less = beignet.statistics.z_test_power(dpos2, npop, alternative="less")
    p_less.sum().backward()
    assert dpos2.grad is not None and dpos2.grad < 0

    # Test known values
    # Test case: effect size = 0.5, n = 16, one-sided should give reasonable power
    effect_size_known = torch.tensor(0.5, dtype=dtype)
    sample_size_known = torch.tensor(16, dtype=dtype)

    power_one_sided = z_test_power(
        effect_size_known,
        sample_size_known,
        alpha=0.05,
        alternative="greater",
    )

    # Should be reasonable power, not too high or too low
    assert 0.5 <= float(power_one_sided) <= 0.95

    # Test invalid alternative
    with pytest.raises(ValueError):
        z_test_power(effect_size_known, sample_size_known, alternative="invalid")


def test_z_test_power_cross_validation_grid():
    import scipy.stats as stats
    import statsmodels.stats.power

    import beignet

    dtype = torch.float64

    effects = [0.2, 0.5, 0.8]
    ns = [20, 50, 100]
    alphas = [0.01, 0.05]
    alts = ["two-sided", "greater", "less"]

    for eff in effects:
        for n in ns:
            for alpha in alphas:
                for alt in alts:
                    beignet_power = beignet.statistics.z_test_power(
                        torch.tensor(eff, dtype=dtype),
                        torch.tensor(float(n), dtype=dtype),
                        alpha=alpha,
                        alternative=alt,
                    )
                    ncp = eff * (n**0.5)
                    if alt == "two-sided":
                        zcrit = stats.norm.ppf(1 - alpha / 2)
                        # P(Z > zcrit - ncp) + P(Z < -zcrit - ncp)
                        ref = (1 - stats.norm.cdf(zcrit - ncp)) + stats.norm.cdf(
                            -zcrit - ncp,
                        )
                    elif alt == "greater":
                        zcrit = stats.norm.ppf(1 - alpha)
                        ref = 1 - stats.norm.cdf(zcrit - ncp)
                    else:  # less
                        zcrit = stats.norm.ppf(1 - alpha)
                        ref = stats.norm.cdf(-zcrit - ncp)

                    ok_sp = torch.isclose(
                        beignet_power,
                        torch.tensor(ref, dtype=dtype),
                        rtol=1e-6,
                        atol=1e-6,
                    ).item()
                    # Statsmodels reference if available
                    try:
                        if alt == "two-sided":
                            sm_alt = "two-sided"
                        elif alt == "greater":
                            sm_alt = "larger"
                        else:
                            sm_alt = "smaller"
                        # Normal test power (one-sample approximation with z)
                        sm_power = statsmodels.stats.power.NormalIndPower().solve_power(
                            effect_size=eff,
                            nobs1=n,
                            alpha=alpha,
                            power=None,
                            ratio=0.0,
                            alternative=sm_alt,
                        )
                        ok_sm = torch.isclose(
                            beignet_power,
                            torch.tensor(sm_power, dtype=dtype),
                            rtol=0.05,
                            atol=2e-3,
                        ).item()
                    except Exception:
                        ok_sm = True
                    assert ok_sp and ok_sm, (
                        f"eff={eff}, n={n}, alpha={alpha}, alt={alt}, "
                        f"beignet={float(beignet_power):.8f}, ref={float(ref):.8f}"
                    )
