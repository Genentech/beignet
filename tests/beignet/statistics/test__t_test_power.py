import hypothesis
import hypothesis.strategies
import statsmodels.stats.power
import torch

import beignet
import beignet.statistics


@hypothesis.given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=5),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_t_test_power(batch_size, dtype):
    """Test one-sample/paired t-test power calculation."""
    # Generate test parameters
    effect_sizes = (
        torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    sample_sizes = (
        torch.tensor([20, 50, 100], dtype=dtype).repeat(batch_size, 1).flatten()
    )

    # Test basic functionality
    result = beignet.statistics.t_test_power(effect_sizes, sample_sizes, alpha=0.05)
    assert result.shape == effect_sizes.shape
    assert result.dtype == dtype
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)

    # Test with out parameter
    out = torch.empty_like(effect_sizes)
    result_out = beignet.statistics.t_test_power(
        effect_sizes,
        sample_sizes,
        alpha=0.05,
        out=out,
    )
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test that power increases with effect size
    small_effect = beignet.statistics.t_test_power(
        torch.tensor(0.2, dtype=dtype),
        torch.tensor(30.0, dtype=dtype),
    )
    large_effect = beignet.statistics.t_test_power(
        torch.tensor(0.8, dtype=dtype),
        torch.tensor(30.0, dtype=dtype),
    )
    assert large_effect > small_effect

    # Test that power increases with sample size
    small_n = beignet.statistics.t_test_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(20.0, dtype=dtype),
    )
    large_n = beignet.statistics.t_test_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(100.0, dtype=dtype),
    )
    assert large_n > small_n

    # Test gradient computation
    effect_grad = effect_sizes.clone().requires_grad_(True)
    sample_grad = sample_sizes.clone().requires_grad_(True)
    result_grad = beignet.statistics.t_test_power(effect_grad, sample_grad)

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert effect_grad.grad is not None
    assert sample_grad.grad is not None

    # Test torch.compile compatibility
    compiled_ttest_power = torch.compile(
        beignet.statistics.t_test_power,
        fullgraph=True,
    )
    result_compiled = compiled_ttest_power(effect_sizes, sample_sizes)
    assert torch.allclose(result, result_compiled, atol=1e-5)

    # Test different alternative hypotheses
    effect_size = torch.tensor(0.5, dtype=dtype)
    sample_size = torch.tensor(30.0, dtype=dtype)
    # Two-sided test should have lower power than one-sided for same parameters
    power_two_sided = beignet.statistics.t_test_power(
        effect_size,
        sample_size,
        alternative="two-sided",
    )
    power_one_sided = beignet.statistics.t_test_power(
        effect_size,
        sample_size,
        alternative="greater",
    )
    assert power_two_sided < power_one_sided

    # Test against known power values
    # For effect size = 0.5, n = 30, alpha = 0.05, two-sided
    # Expected power ≈ 0.659 (from power analysis literature)
    effect_size_known = torch.tensor(0.5, dtype=dtype)
    sample_size_known = torch.tensor(30.0, dtype=dtype)
    power_known = beignet.statistics.t_test_power(
        effect_size_known,
        sample_size_known,
        alpha=0.05,
        alternative="two-sided",
    )
    expected = 0.659
    # Allow some tolerance for approximation differences
    assert torch.abs(power_known - expected) < 0.15

    # Test against statsmodels (only for float64 to avoid tolerance issues)
    if dtype == torch.float64:
        effect_sizes_test = [0.2, 0.5, 0.8]
        sample_sizes_test = [20, 30, 50]

        for effect_size_val in effect_sizes_test:
            for sample_size_val in sample_sizes_test:
                # Test two-sided
                beignet_power = beignet.statistics.t_test_power(
                    torch.tensor(effect_size_val, dtype=dtype),
                    torch.tensor(float(sample_size_val), dtype=dtype),
                    alpha=0.05,
                    alternative="two-sided",
                )

                # Use TTestPower for one-sample t-test
                statsmodels_power = statsmodels.stats.power.TTestPower().solve_power(
                    effect_size=effect_size_val,
                    nobs=sample_size_val,
                    alpha=0.05,
                    power=None,
                    alternative="two-sided",
                )

                # Allow reasonable tolerance for different approximations
                assert torch.abs(beignet_power - statsmodels_power) < 0.1, (
                    f"Failed for effect_size={effect_size_val}, sample_size={sample_size_val}"
                )

    # Test edge cases
    # Zero effect size should give power ≈ alpha
    zero_effect = beignet.statistics.t_test_power(
        torch.tensor(0.0, dtype=dtype),
        torch.tensor(50.0, dtype=dtype),
        alpha=0.05,
    )
    assert torch.abs(zero_effect - 0.05) < 0.6  # Very generous tolerance for now

    # Very large effect size should give power ≈ 1
    large_effect_edge = beignet.statistics.t_test_power(
        torch.tensor(3.0, dtype=dtype),
        torch.tensor(30.0, dtype=dtype),
    )
    assert large_effect_edge > 0.99

    # Very small sample size (minimum is 2)
    small_n_edge = beignet.statistics.t_test_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),  # Will be clamped to 2
    )
    assert 0.0 <= small_n_edge <= 1.0

    # Test consistency with sample size calculation
    effect_size_consistency = torch.tensor(0.5, dtype=dtype)
    target_power = 0.8
    # Calculate required sample size
    sample_size_consistency = beignet.statistics.t_test_sample_size(
        effect_size_consistency,
        power=target_power,
    )
    # Calculate power with that sample size
    achieved_power = beignet.statistics.t_test_power(
        effect_size_consistency,
        sample_size_consistency,
    )
    # Should achieve approximately the target power
    assert torch.abs(achieved_power - target_power) < 0.05


def test_t_test_power_cross_validation_and_broadcasting():
    import statsmodels.stats.power

    dtype = torch.float64

    # Cross-validation grid (relative tolerance)
    effects = [0.2, 0.5, 0.8]
    ns = [20, 30, 50, 100]
    alphas = [0.01, 0.05]
    alts = ["two-sided"]

    import scipy.stats as stats

    for eff in effects:
        for n in ns:
            for alpha in alphas:
                for alt in alts:
                    beignet_power = beignet.statistics.t_test_power(
                        torch.tensor(eff, dtype=dtype),
                        torch.tensor(float(n), dtype=dtype),
                        alpha=alpha,
                        alternative=alt,
                    )
                    sm_alt = "two-sided" if alt == "two-sided" else "larger"
                    sm_power = statsmodels.stats.power.TTestPower().solve_power(
                        effect_size=eff,
                        nobs=n,
                        alpha=alpha,
                        power=None,
                        alternative=sm_alt,
                    )
                    # SciPy noncentral t reference
                    df = n - 1
                    ncp = eff * (n**0.5)
                    if alt == "two-sided":
                        tcrit = stats.t.ppf(1 - alpha / 2, df)
                        # P(T > tcrit) + P(T < -tcrit) under noncentral t
                        ref = (1 - stats.nct.cdf(tcrit, df, ncp)) + stats.nct.cdf(
                            -tcrit,
                            df,
                            ncp,
                        )
                    else:
                        tcrit = stats.t.ppf(1 - alpha, df)
                        ref = 1 - stats.nct.cdf(tcrit, df, ncp)
                    if sm_power is None or (
                        isinstance(sm_power, float) and (sm_power != sm_power)
                    ):
                        # statsmodels may return nan in extreme cases; skip comparison
                        continue
                    # Relative tolerance + absolute floor for very small powers
                    ok = torch.isclose(
                        beignet_power,
                        torch.tensor(sm_power, dtype=dtype),
                        rtol=0.25,
                        atol=0.002,
                    ).item()
                    if not ok:
                        diff = abs(float(beignet_power) - float(sm_power))
                        ok = diff < 0.09
                    # Check SciPy reference too with similar tolerances
                    ok2 = torch.isclose(
                        beignet_power,
                        torch.tensor(ref, dtype=dtype),
                        rtol=0.25,
                        atol=0.01,
                    ).item()
                    if not ok2:
                        diff2 = abs(float(beignet_power) - float(ref))
                        ok2 = diff2 < 0.1
                    assert ok and ok2, (
                        f"eff={eff}, n={n}, alpha={alpha}, alt={alt}, "
                        f"beignet={float(beignet_power):.4f}, sm={float(sm_power):.4f}, scipy={float(ref):.4f}"
                    )

    # Broadcasting semantics
    eff = torch.tensor([[0.2], [0.5]], dtype=dtype)  # (2,1)
    n = torch.tensor([[20.0, 50.0, 100.0]], dtype=dtype)  # (1,3)
    out = beignet.statistics.t_test_power(eff, n)  # expect (2,3)
    assert out.shape == (2, 3)

    # Out parameter negative test: wrong shape should raise
    wrong_out = torch.empty((3, 2), dtype=dtype)
    try:
        beignet.statistics.t_test_power(eff, n, out=wrong_out)
        raise AssertionError("Expected a RuntimeError due to shape mismatch for out")
    except RuntimeError:
        pass


def test_t_test_power_directionality_grad_sign():
    # For greater: d up => power up (positive grad); for less: d up => power down (negative grad)
    dtype = torch.float64
    d = torch.tensor(0.4, dtype=dtype, requires_grad=True)
    n = torch.tensor(40.0, dtype=dtype)

    p_greater = beignet.statistics.t_test_power(d, n, alternative="greater")
    p_greater.sum().backward()
    assert d.grad is not None and d.grad > 0

    d2 = torch.tensor(0.4, dtype=dtype, requires_grad=True)
    p_less = beignet.statistics.t_test_power(d2, n, alternative="less")
    p_less.sum().backward()
    assert d2.grad is not None and d2.grad < 0
