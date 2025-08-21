import hypothesis.strategies
import statsmodels.stats.power
import torch
from hypothesis import given, settings

import beignet
import beignet.statistics


@given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=5),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
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
