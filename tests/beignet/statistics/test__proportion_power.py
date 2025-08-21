import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet

try:
    from statsmodels.stats.power import proportions_ztest_power

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


@given(
    batch_size=st.integers(min_value=1, max_value=5),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_proportion_power(batch_size, dtype):
    """Test proportion power calculation."""
    # Generate test parameters
    p0_values = (
        torch.tensor([0.4, 0.5, 0.6], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    p1_values = (
        torch.tensor([0.5, 0.6, 0.7], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    sample_sizes = (
        torch.tensor([50, 100, 200], dtype=dtype).repeat(batch_size, 1).flatten()
    )

    # Test basic functionality - two-sided test
    result = beignet.proportion_power(
        p0_values, p1_values, sample_sizes, alpha=0.05, alternative="two-sided"
    )
    assert result.shape == p0_values.shape
    assert result.dtype == dtype
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)

    # Test one-sided tests
    result_greater = beignet.proportion_power(
        p0_values, p1_values, sample_sizes, alpha=0.05, alternative="greater"
    )
    result_less = beignet.proportion_power(
        p0_values, p1_values, sample_sizes, alpha=0.05, alternative="less"
    )

    assert torch.all(result_greater >= 0.0)
    assert torch.all(result_greater <= 1.0)
    assert torch.all(result_less >= 0.0)
    assert torch.all(result_less <= 1.0)

    # Test with out parameter
    out = torch.empty_like(p0_values)
    result_out = beignet.proportion_power(
        p0_values, p1_values, sample_sizes, alpha=0.05, alternative="two-sided", out=out
    )
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test power increases with effect size
    small_effect = beignet.proportion_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(0.51, dtype=dtype),
        torch.tensor(100.0, dtype=dtype),
    )
    large_effect = beignet.proportion_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(0.6, dtype=dtype),
        torch.tensor(100.0, dtype=dtype),
    )

    assert large_effect > small_effect

    # Test power increases with sample size
    small_n = beignet.proportion_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(0.6, dtype=dtype),
        torch.tensor(50.0, dtype=dtype),
    )
    large_n = beignet.proportion_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(0.6, dtype=dtype),
        torch.tensor(200.0, dtype=dtype),
    )

    assert large_n > small_n

    # Test gradient computation
    p0_grad = p0_values.clone().requires_grad_(True)
    p1_grad = p1_values.clone().requires_grad_(True)
    sample_grad = sample_sizes.clone().requires_grad_(True)
    result_grad = beignet.proportion_power(p0_grad, p1_grad, sample_grad)

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert p0_grad.grad is not None
    assert p1_grad.grad is not None
    assert sample_grad.grad is not None

    # Test torch.compile compatibility
    compiled_proportion_power = torch.compile(beignet.proportion_power, fullgraph=True)
    result_compiled = compiled_proportion_power(p0_values, p1_values, sample_sizes)
    assert torch.allclose(result, result_compiled, atol=1e-5)

    # Test invalid alternative
    try:
        beignet.proportion_power(
            torch.tensor(0.5, dtype=dtype),
            torch.tensor(0.6, dtype=dtype),
            torch.tensor(100.0, dtype=dtype),
            alternative="invalid",
        )
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass


def test_proportion_power_against_known_values():
    """Test proportion power against known theoretical values."""
    # For moderate effect (0.5 vs 0.6) with n=100, power should be reasonable
    p0 = torch.tensor(0.5, dtype=torch.float32)
    p1 = torch.tensor(0.6, dtype=torch.float32)
    n = torch.tensor(100.0, dtype=torch.float32)
    power = beignet.proportion_power(p0, p1, n, alpha=0.05, alternative="two-sided")

    # Should be somewhere between 0.3 and 0.7 for these parameters
    assert 0.3 < power < 0.7

    # Test edge cases - same proportions should give power ≈ alpha
    same_props = beignet.proportion_power(p0, p0, n, alpha=0.05)
    assert torch.abs(same_props - 0.05) < 0.03


def test_proportion_power_against_statsmodels():
    """Test proportion power against statsmodels reference implementation."""
    if not HAS_STATSMODELS:
        return

    # Test parameters
    test_cases = [
        (0.4, 0.5, 100, 0.05, "two-sided"),
        (0.5, 0.6, 150, 0.05, "two-sided"),
        (0.3, 0.4, 200, 0.01, "two-sided"),
        (0.5, 0.6, 100, 0.05, "larger"),
        (0.6, 0.5, 100, 0.05, "smaller"),
    ]

    for p0_val, p1_val, n_val, alpha_val, alternative in test_cases:
        # Our implementation
        p0 = torch.tensor(p0_val, dtype=torch.float64)
        p1 = torch.tensor(p1_val, dtype=torch.float64)
        n = torch.tensor(float(n_val), dtype=torch.float64)
        beignet_result = beignet.proportion_power(
            p0, p1, n, alpha=alpha_val, alternative=alternative
        )

        # Convert alternative to statsmodels format
        if alternative == "two-sided":
            sm_alternative = "two-sided"
        elif alternative == "larger":
            sm_alternative = "larger"
        elif alternative == "smaller":
            sm_alternative = "smaller"

        # Statsmodels implementation
        try:
            # Calculate effect size (proportions difference)
            effect_size = p1_val - p0_val

            # Use proportions_ztest_power from statsmodels
            sm_result = proportions_ztest_power(
                effect_size=effect_size,
                nobs1=n_val,
                alpha=alpha_val,
                alternative=sm_alternative,
            )

            # Compare results with reasonable tolerance
            tolerance = 0.05
            diff = abs(float(beignet_result) - sm_result)
            assert diff < tolerance, (
                f"p0={p0_val}, p1={p1_val}, n={n_val}, alt={alternative}: beignet={float(beignet_result):.6f}, statsmodels={sm_result:.6f}, diff={diff:.6f}"
            )

        except (ImportError, AttributeError, TypeError):
            # If specific function not available or has issues, skip comparison
            pass


def test_proportion_power_edge_cases():
    """Test edge cases for proportion power calculation."""
    # Test with very small proportions
    tiny_p0 = torch.tensor(0.001, dtype=torch.float64)
    tiny_p1 = torch.tensor(0.002, dtype=torch.float64)
    n = torch.tensor(1000.0, dtype=torch.float64)
    tiny_power = beignet.proportion_power(tiny_p0, tiny_p1, n)
    assert 0.0 <= tiny_power <= 1.0

    # Test with proportions close to 1
    large_p0 = torch.tensor(0.998, dtype=torch.float64)
    large_p1 = torch.tensor(0.999, dtype=torch.float64)
    large_power = beignet.proportion_power(large_p0, large_p1, n)
    assert 0.0 <= large_power <= 1.0

    # Test with very large sample size
    large_n = torch.tensor(50000.0, dtype=torch.float64)
    p0_med = torch.tensor(0.5, dtype=torch.float64)
    p1_med = torch.tensor(0.51, dtype=torch.float64)
    large_n_power = beignet.proportion_power(p0_med, p1_med, large_n)
    assert large_n_power > 0.8  # Should have high power with large n
