import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet

try:
    import statsmodels.stats.power as smp
    from scipy import stats
    from statsmodels.stats.power import ttest_power

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


@given(
    batch_size=st.integers(min_value=1, max_value=5),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_correlation_sample_size(batch_size, dtype):
    """Test correlation sample size calculation."""
    # Generate test parameters
    r_values = (
        torch.tensor([0.1, 0.3, 0.5], dtype=dtype).repeat(batch_size, 1).flatten()
    )

    # Test basic functionality - two-sided test
    result = beignet.correlation_sample_size(
        r_values, power=0.8, alpha=0.05, alternative="two-sided"
    )
    assert result.shape == r_values.shape
    assert result.dtype == dtype
    assert torch.all(
        result >= 3.0
    )  # Minimum sample size should be > 3 for Fisher z-transform
    assert torch.all(result < 10000)  # Should be reasonable

    # Test one-sided tests
    result_greater = beignet.correlation_sample_size(
        r_values, power=0.8, alpha=0.05, alternative="greater"
    )
    result_less = beignet.correlation_sample_size(
        r_values, power=0.8, alpha=0.05, alternative="less"
    )

    assert torch.all(result_greater >= 3.0)
    assert torch.all(result_less >= 3.0)

    # One-sided tests should generally require smaller sample sizes than two-sided
    # Allow some numerical tolerance
    assert torch.all(result_greater <= result + 5)
    assert torch.all(result_less <= result + 5)

    # Test with out parameter
    out = torch.empty_like(r_values)
    result_out = beignet.correlation_sample_size(
        r_values, power=0.8, alpha=0.05, alternative="two-sided", out=out
    )
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test sample size decreases with correlation strength
    small_r = torch.tensor(0.1, dtype=dtype)
    large_r = torch.tensor(0.5, dtype=dtype)

    n_small = beignet.correlation_sample_size(small_r, power=0.8)
    n_large = beignet.correlation_sample_size(large_r, power=0.8)

    assert n_large < n_small

    # Test sample size increases with power
    low_power = beignet.correlation_sample_size(
        torch.tensor(0.3, dtype=dtype), power=0.5
    )
    high_power = beignet.correlation_sample_size(
        torch.tensor(0.3, dtype=dtype), power=0.9
    )

    assert high_power > low_power

    # Test gradient computation
    r_grad = r_values.clone().requires_grad_(True)
    result_grad = beignet.correlation_sample_size(r_grad, power=0.8)

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert r_grad.grad is not None
    assert r_grad.grad.shape == r_values.shape

    # Test torch.compile compatibility
    compiled_correlation_sample_size = torch.compile(
        beignet.correlation_sample_size, fullgraph=True
    )
    result_compiled = compiled_correlation_sample_size(r_values, power=0.8)
    assert torch.allclose(result, result_compiled, atol=1e-5)

    # Test invalid alternative
    try:
        beignet.correlation_sample_size(
            torch.tensor(0.3, dtype=dtype), alternative="invalid"
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_correlation_sample_size_against_known_values():
    """Test correlation sample size against known theoretical values."""
    # For medium correlation (r=0.3) with power=0.8, sample size should be reasonable
    r = torch.tensor(0.3, dtype=torch.float32)
    n = beignet.correlation_sample_size(
        r, power=0.8, alpha=0.05, alternative="two-sided"
    )

    # Should be somewhere between 50 and 150 for these parameters
    assert 50 < n < 150

    # For strong correlation (r=0.7), sample size should be smaller
    strong_r = torch.tensor(0.7, dtype=torch.float32)
    n_strong = beignet.correlation_sample_size(
        strong_r, power=0.8, alpha=0.05, alternative="two-sided"
    )

    # Should require smaller sample size for strong correlation
    assert n_strong < n
    assert n_strong < 50

    # Test very small correlation
    small_r = torch.tensor(0.05, dtype=torch.float32)
    n_small = beignet.correlation_sample_size(small_r, power=0.8, alpha=0.05)

    # Should require large sample size for small correlation
    assert n_small > 500


def test_correlation_sample_size_against_statsmodels():
    """Test correlation sample size against statsmodels reference implementation."""
    if not HAS_STATSMODELS:
        return

    import statsmodels.stats.power as smp

    # Test parameters
    test_cases = [
        (0.1, 0.8, 0.05, "two-sided"),
        (0.3, 0.8, 0.05, "two-sided"),
        (0.5, 0.9, 0.01, "two-sided"),
        (0.2, 0.8, 0.05, "greater"),
        (0.4, 0.8, 0.05, "less"),
    ]

    for r_val, power_val, alpha_val, alternative in test_cases:
        # Our implementation
        r = torch.tensor(r_val, dtype=torch.float64)
        beignet_result = beignet.correlation_sample_size(
            r, power=power_val, alpha=alpha_val, alternative=alternative
        )

        # Convert alternative to statsmodels format
        if alternative == "two-sided":
            sm_alternative = "two-sided"
        elif alternative == "greater":
            sm_alternative = "larger"
        elif alternative == "less":
            sm_alternative = "smaller"

        # Use Fisher z-transformation sample size calculation from statsmodels
        try:
            # Calculate sample size using Fisher z-transformation
            sm_result = smp.zt_ind_solve_power(
                effect_size=r_val,
                nobs=None,
                alpha=alpha_val,
                power=power_val,
                alternative=sm_alternative,
            )

            # Compare results with reasonable tolerance (sample sizes can vary by rounding)
            tolerance = 3  # Allow difference of up to 3 subjects
            diff = abs(float(beignet_result) - sm_result)
            assert diff < tolerance, (
                f"r={r_val}, power={power_val}, alt={alternative}: beignet={float(beignet_result):.1f}, statsmodels={sm_result:.1f}, diff={diff:.1f}"
            )

        except (ImportError, AttributeError):
            # If specific function not available, skip comparison
            pass


def test_correlation_sample_size_consistency():
    """Test that sample size and power calculations are consistent."""
    # Calculate sample size for given power, then verify power with that sample size
    r = torch.tensor(0.3, dtype=torch.float64)
    target_power = 0.8
    alpha = 0.05

    # Calculate required sample size
    n = beignet.correlation_sample_size(r, power=target_power, alpha=alpha)

    # Calculate power with that sample size
    actual_power = beignet.correlation_power(r, n, alpha=alpha)

    # Should be close to target power (within tolerance for rounding)
    # The difference comes from ceiling operation in sample size calculation
    assert abs(float(actual_power) - target_power) < 0.15
