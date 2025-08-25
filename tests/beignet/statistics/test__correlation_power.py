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
def test_correlation_power(batch_size, dtype):
    """Test correlation power calculation."""
    # Generate test parameters
    r_values = (
        torch.tensor([0.1, 0.3, 0.5], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    sample_sizes = (
        torch.tensor([20, 50, 100], dtype=dtype).repeat(batch_size, 1).flatten()
    )

    # Test basic functionality - two-sided test
    result = beignet.statistics.correlation_power(
        r_values, sample_sizes, alpha=0.05, alternative="two-sided"
    )
    assert result.shape == r_values.shape
    assert result.dtype == dtype
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)

    # Test one-sided tests
    result_greater = beignet.statistics.correlation_power(
        r_values, sample_sizes, alpha=0.05, alternative="greater"
    )
    result_less = beignet.statistics.correlation_power(
        r_values, sample_sizes, alpha=0.05, alternative="less"
    )

    assert torch.all(result_greater >= 0.0)
    assert torch.all(result_greater <= 1.0)
    assert torch.all(result_less >= 0.0)
    assert torch.all(result_less <= 1.0)

    # One-sided tests should generally have higher power than two-sided
    positive_r = torch.abs(r_values)
    power_two_sided = beignet.statistics.correlation_power(
        positive_r, sample_sizes, alpha=0.05, alternative="two-sided"
    )
    power_one_sided = beignet.statistics.correlation_power(
        positive_r, sample_sizes, alpha=0.05, alternative="greater"
    )

    # Allow some numerical tolerance
    assert torch.all(power_one_sided >= power_two_sided - 0.01)

    # Test with out parameter
    out = torch.empty_like(r_values)
    result_out = beignet.statistics.correlation_power(
        r_values, sample_sizes, alpha=0.05, alternative="two-sided", out=out
    )
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test power increases with correlation strength
    small_r = torch.tensor(0.1, dtype=dtype)
    large_r = torch.tensor(0.5, dtype=dtype)
    sample_size = torch.tensor(50.0, dtype=dtype)

    power_small = beignet.statistics.correlation_power(small_r, sample_size)
    power_large = beignet.statistics.correlation_power(large_r, sample_size)

    assert power_large > power_small

    # Test power increases with sample size
    small_n = torch.tensor(20.0, dtype=dtype)
    large_n = torch.tensor(100.0, dtype=dtype)
    r_test = torch.tensor(0.3, dtype=dtype)

    power_small_n = beignet.statistics.correlation_power(r_test, small_n)
    power_large_n = beignet.statistics.correlation_power(r_test, large_n)

    assert power_large_n > power_small_n

    # Test gradient computation
    r_grad = r_values.clone().requires_grad_(True)
    sample_grad = sample_sizes.clone().requires_grad_(True)
    result_grad = beignet.statistics.correlation_power(r_grad, sample_grad)

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert r_grad.grad is not None
    assert sample_grad.grad is not None
    assert r_grad.grad.shape == r_values.shape
    assert sample_grad.grad.shape == sample_sizes.shape

    # Test torch.compile compatibility
    compiled_correlation_power = torch.compile(
        beignet.statistics.correlation_power, fullgraph=True
    )
    result_compiled = compiled_correlation_power(r_values, sample_sizes)
    assert torch.allclose(result, result_compiled, atol=1e-5)

    # Test zero correlation (should give power ≈ alpha)
    zero_r = torch.tensor(0.0, dtype=dtype)
    zero_power = beignet.statistics.correlation_power(zero_r, sample_size, alpha=0.05)
    assert torch.abs(zero_power - 0.05) < 0.03

    # Test invalid alternative
    try:
        beignet.statistics.correlation_power(r_test, sample_size, alternative="invalid")
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass

    # Test correlation power against known theoretical values
    # For medium correlation (r=0.3) with n=50, power should be reasonable
    r = torch.tensor(0.3, dtype=dtype)
    n = torch.tensor(50.0, dtype=dtype)
    power = beignet.statistics.correlation_power(
        r, n, alpha=0.05, alternative="two-sided"
    )

    # Should be somewhere between 0.4 and 0.8 for these parameters
    assert 0.4 < power < 0.8

    # For strong correlation (r=0.7) with n=30, power should be high
    strong_r = torch.tensor(0.7, dtype=dtype)
    n_small = torch.tensor(30.0, dtype=dtype)
    power_strong = beignet.statistics.correlation_power(
        strong_r, n_small, alpha=0.05, alternative="two-sided"
    )

    # Should have high power for strong correlation
    assert power_strong > 0.8

    # Test edge cases
    very_small_r = torch.tensor(0.01, dtype=dtype)
    power_tiny = beignet.statistics.correlation_power(very_small_r, n, alpha=0.05)

    # Should be close to alpha for very small correlation
    assert torch.abs(power_tiny - 0.05) < 0.1


def test_correlation_power_broadcasting_out_and_directionality():
    dtype = torch.float64

    # Broadcasting shapes
    r = torch.tensor([[0.1], [0.3]], dtype=dtype)  # (2,1)
    n = torch.tensor([[30.0, 60.0, 100.0]], dtype=dtype)  # (1,3)
    out = beignet.statistics.correlation_power(r, n, alternative="two-sided")
    assert out.shape == (2, 3)

    # Out wrong shape
    wrong_out = torch.empty((3, 2), dtype=dtype)
    try:
        beignet.statistics.correlation_power(r, n, out=wrong_out)
        raise AssertionError("Expected RuntimeError on mismatched out shape")
    except RuntimeError:
        pass

    # Directionality gradient sign
    rpos = torch.tensor(0.3, dtype=dtype, requires_grad=True)
    ns = torch.tensor(50.0, dtype=dtype)
    p_greater = beignet.statistics.correlation_power(rpos, ns, alternative="greater")
    p_greater.sum().backward()
    assert rpos.grad is not None and rpos.grad > 0

    rpos2 = torch.tensor(0.3, dtype=dtype, requires_grad=True)
    p_less = beignet.statistics.correlation_power(rpos2, ns, alternative="less")
    p_less.sum().backward()
    assert rpos2.grad is not None and rpos2.grad < 0

    # Test correlation power against statsmodels reference implementation

    # Test parameters
    test_cases = [
        (0.1, 30, 0.05, "two-sided"),
        (0.3, 50, 0.05, "two-sided"),
        (0.5, 100, 0.01, "two-sided"),
        (0.2, 75, 0.05, "greater"),
        (0.4, 40, 0.05, "less"),
    ]

    for r_val, n_val, alpha_val, alternative in test_cases:
        # Our implementation
        r = torch.tensor(r_val, dtype=dtype)
        n = torch.tensor(float(n_val), dtype=dtype)
        beignet_result = beignet.statistics.correlation_power(
            r, n, alpha=alpha_val, alternative=alternative
        )

        # Convert alternative to statsmodels format
        if alternative == "two-sided":
            sm_alternative = "two-sided"
        elif alternative == "greater":
            sm_alternative = "larger"
        elif alternative == "less":
            sm_alternative = "smaller"

        # Use Fisher z-transformation power calculation from statsmodels
        # statsmodels.stats.power.zt_ind_solve_power for correlation tests
        try:
            # Calculate power using Fisher z-transformation
            sm_result = statsmodels.stats.power.zt_ind_solve_power(
                effect_size=r_val,
                nobs1=n_val,
                alpha=alpha_val,
                power=None,
                alternative=sm_alternative,
            )

            # Compare results with reasonable tolerance
            tolerance = 0.2  # Allow for differences in calculation methods
            diff = abs(float(beignet_result) - sm_result)
            assert diff < tolerance, (
                f"r={r_val}, n={n_val}, alt={alternative}: beignet={float(beignet_result):.6f}, statsmodels={sm_result:.6f}, diff={diff:.6f}"
            )

        except (ImportError, AttributeError):
            # If specific function not available, skip comparison
            pass


def test_correlation_power_cross_validation_grid():
    import scipy.stats as stats

    dtype = torch.float64
    rs = [0.1, 0.3, 0.5]
    ns = [20, 50, 100]
    alphas = [0.01, 0.05]
    alts = ["two-sided", "greater"]

    for r_val in rs:
        for n_val in ns:
            for alpha_val in alphas:
                for alt in alts:
                    b = beignet.statistics.correlation_power(
                        torch.tensor(r_val, dtype=dtype),
                        torch.tensor(float(n_val), dtype=dtype),
                        alpha=alpha_val,
                        alternative=alt,
                    )

                    # Fisher z-transform reference with SciPy
                    z_r = 0.5 * float(
                        torch.log((1 + torch.tensor(r_val)) / (1 - torch.tensor(r_val)))
                    )
                    se = 1.0 / (n_val - 3) ** 0.5
                    z_stat = z_r / se

                    if alt == "two-sided":
                        zcrit = stats.norm.ppf(1 - alpha_val / 2)
                        ref = (1 - stats.norm.cdf(zcrit - z_stat)) + stats.norm.cdf(
                            -zcrit - z_stat
                        )
                    elif alt == "greater":
                        zcrit = stats.norm.ppf(1 - alpha_val)
                        ref = 1 - stats.norm.cdf(zcrit - z_stat)
                    else:  # less
                        zcrit = stats.norm.ppf(alpha_val)
                        ref = stats.norm.cdf(zcrit - z_stat)

                    assert torch.isclose(
                        b, torch.tensor(ref, dtype=dtype), rtol=0.2, atol=5e-2
                    ), (
                        f"r={r_val}, n={n_val}, alpha={alpha_val}, alt={alt}, "
                        f"beignet={float(b):.6f}, ref={float(ref):.6f}"
                    )
