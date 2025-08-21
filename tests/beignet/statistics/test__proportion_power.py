import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet
import beignet.statistics

# from statsmodels.stats.power import proportions_ztest_power  # Function not available in current statsmodels version


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
    result = beignet.statistics.proportion_power(
        p0_values, p1_values, sample_sizes, alpha=0.05, alternative="two-sided"
    )
    assert result.shape == p0_values.shape
    assert result.dtype == dtype
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)

    # Test one-sided tests
    result_greater = beignet.statistics.proportion_power(
        p0_values, p1_values, sample_sizes, alpha=0.05, alternative="greater"
    )
    result_less = beignet.statistics.proportion_power(
        p0_values, p1_values, sample_sizes, alpha=0.05, alternative="less"
    )

    assert torch.all(result_greater >= 0.0)
    assert torch.all(result_greater <= 1.0)
    assert torch.all(result_less >= 0.0)
    assert torch.all(result_less <= 1.0)

    # Test with out parameter
    out = torch.empty_like(p0_values)
    result_out = beignet.statistics.proportion_power(
        p0_values, p1_values, sample_sizes, alpha=0.05, alternative="two-sided", out=out
    )
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test power increases with effect size
    small_effect = beignet.statistics.proportion_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(0.51, dtype=dtype),
        torch.tensor(100.0, dtype=dtype),
    )
    large_effect = beignet.statistics.proportion_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(0.6, dtype=dtype),
        torch.tensor(100.0, dtype=dtype),
    )

    assert large_effect > small_effect

    # Test power increases with sample size
    small_n = beignet.statistics.proportion_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(0.6, dtype=dtype),
        torch.tensor(50.0, dtype=dtype),
    )
    large_n = beignet.statistics.proportion_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(0.6, dtype=dtype),
        torch.tensor(200.0, dtype=dtype),
    )

    assert large_n > small_n

    # Test gradient computation
    p0_grad = p0_values.clone().requires_grad_(True)
    p1_grad = p1_values.clone().requires_grad_(True)
    sample_grad = sample_sizes.clone().requires_grad_(True)
    result_grad = beignet.statistics.proportion_power(p0_grad, p1_grad, sample_grad)

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert p0_grad.grad is not None
    assert p1_grad.grad is not None
    assert sample_grad.grad is not None

    # Test torch.compile compatibility
    compiled_proportion_power = torch.compile(
        beignet.statistics.proportion_power, fullgraph=True
    )
    result_compiled = compiled_proportion_power(p0_values, p1_values, sample_sizes)
    assert torch.allclose(result, result_compiled, atol=1e-5)

    # Test invalid alternative
    try:
        beignet.statistics.proportion_power(
            torch.tensor(0.5, dtype=dtype),
            torch.tensor(0.6, dtype=dtype),
            torch.tensor(100.0, dtype=dtype),
            alternative="invalid",
        )
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass

    # Test proportion power against known theoretical values
    # For moderate effect (0.5 vs 0.6) with n=100, power should be reasonable
    p0 = torch.tensor(0.5, dtype=dtype)
    p1 = torch.tensor(0.6, dtype=dtype)
    n = torch.tensor(100.0, dtype=dtype)
    power = beignet.statistics.proportion_power(
        p0, p1, n, alpha=0.05, alternative="two-sided"
    )

    # Should be somewhere between 0.3 and 0.7 for these parameters
    assert 0.3 < power < 0.7

    # Test edge cases - same proportions should give power ≈ alpha
    same_props = beignet.statistics.proportion_power(p0, p0, n, alpha=0.05)
    assert torch.abs(same_props - 0.05) < 0.03

    # Test edge cases for proportion power calculation
    # Test with very small proportions
    tiny_p0 = torch.tensor(0.001, dtype=dtype)
    tiny_p1 = torch.tensor(0.002, dtype=dtype)
    n = torch.tensor(1000.0, dtype=dtype)
    tiny_power = beignet.statistics.proportion_power(tiny_p0, tiny_p1, n)
    assert 0.0 <= tiny_power <= 1.0

    # Test with proportions close to 1
    large_p0 = torch.tensor(0.998, dtype=dtype)
    large_p1 = torch.tensor(0.999, dtype=dtype)
    large_power = beignet.statistics.proportion_power(large_p0, large_p1, n)
    assert 0.0 <= large_power <= 1.0

    # Test with very large sample size
    large_n = torch.tensor(50000.0, dtype=dtype)
    p0_med = torch.tensor(0.5, dtype=dtype)
    p1_med = torch.tensor(0.51, dtype=dtype)
    large_n_power = beignet.statistics.proportion_power(p0_med, p1_med, large_n)
    assert large_n_power > 0.8  # Should have high power with large n
