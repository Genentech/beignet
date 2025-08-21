import hypothesis
import hypothesis.strategies
import torch

import beignet.statistics

# from statsmodels.stats.power import proportions_ztest_power  # Function not available in current statsmodels version


@hypothesis.given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=5),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_proportion_two_sample_power(batch_size, dtype):
    """Test two-sample proportion power calculation."""
    # Generate test parameters
    p1_values = (
        torch.tensor([0.4, 0.5, 0.6], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    p2_values = (
        torch.tensor([0.5, 0.6, 0.7], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    n1_values = (
        torch.tensor([50, 100, 150], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    n2_values = (
        torch.tensor([50, 100, 150], dtype=dtype).repeat(batch_size, 1).flatten()
    )

    # Test basic functionality - two-sided test
    result = beignet.statistics.proportion_two_sample_power(
        p1_values, p2_values, n1_values, n2_values, alpha=0.05, alternative="two-sided"
    )
    assert result.shape == p1_values.shape
    assert result.dtype == dtype
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)

    # Test with equal sample sizes (n2=None)
    result_equal = beignet.statistics.proportion_two_sample_power(
        p1_values, p2_values, n1_values, None, alpha=0.05, alternative="two-sided"
    )
    result_explicit = beignet.statistics.proportion_two_sample_power(
        p1_values, p2_values, n1_values, n1_values, alpha=0.05, alternative="two-sided"
    )
    assert torch.allclose(result_equal, result_explicit)

    # Test one-sided tests
    result_greater = beignet.statistics.proportion_two_sample_power(
        p1_values, p2_values, n1_values, n2_values, alpha=0.05, alternative="greater"
    )
    result_less = beignet.statistics.proportion_two_sample_power(
        p1_values, p2_values, n1_values, n2_values, alpha=0.05, alternative="less"
    )

    assert torch.all(result_greater >= 0.0)
    assert torch.all(result_greater <= 1.0)
    assert torch.all(result_less >= 0.0)
    assert torch.all(result_less <= 1.0)

    # Test with out parameter
    out = torch.empty_like(p1_values)
    result_out = beignet.statistics.proportion_two_sample_power(
        p1_values,
        p2_values,
        n1_values,
        n2_values,
        alpha=0.05,
        alternative="two-sided",
        out=out,
    )
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test power increases with effect size
    small_effect = beignet.statistics.proportion_two_sample_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(0.51, dtype=dtype),
        torch.tensor(100.0, dtype=dtype),
        torch.tensor(100.0, dtype=dtype),
    )
    large_effect = beignet.statistics.proportion_two_sample_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(0.6, dtype=dtype),
        torch.tensor(100.0, dtype=dtype),
        torch.tensor(100.0, dtype=dtype),
    )

    assert large_effect > small_effect

    # Test power increases with sample size
    small_n = beignet.statistics.proportion_two_sample_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(0.6, dtype=dtype),
        torch.tensor(50.0, dtype=dtype),
        torch.tensor(50.0, dtype=dtype),
    )
    large_n = beignet.statistics.proportion_two_sample_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(0.6, dtype=dtype),
        torch.tensor(200.0, dtype=dtype),
        torch.tensor(200.0, dtype=dtype),
    )

    assert large_n > small_n

    # Test gradient computation
    p1_grad = p1_values.clone().requires_grad_(True)
    p2_grad = p2_values.clone().requires_grad_(True)
    n1_grad = n1_values.clone().requires_grad_(True)
    n2_grad = n2_values.clone().requires_grad_(True)
    result_grad = beignet.statistics.proportion_two_sample_power(
        p1_grad, p2_grad, n1_grad, n2_grad
    )

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert p1_grad.grad is not None
    assert p2_grad.grad is not None
    assert n1_grad.grad is not None
    assert n2_grad.grad is not None

    # Test torch.compile compatibility
    compiled_proportion_two_sample_power = torch.compile(
        beignet.statistics.proportion_two_sample_power, fullgraph=True
    )
    result_compiled = compiled_proportion_two_sample_power(
        p1_values, p2_values, n1_values, n2_values
    )
    assert torch.allclose(result, result_compiled, atol=1e-5)

    # Test invalid alternative
    try:
        beignet.statistics.proportion_two_sample_power(
            torch.tensor(0.5, dtype=dtype),
            torch.tensor(0.6, dtype=dtype),
            torch.tensor(100.0, dtype=dtype),
            torch.tensor(100.0, dtype=dtype),
            alternative="invalid",
        )
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass

    # Test against known theoretical values
    # For moderate effect (0.5 vs 0.6) with n=100 each, power should be reasonable
    p1 = torch.tensor(0.5, dtype=dtype)
    p2 = torch.tensor(0.6, dtype=dtype)
    n1 = torch.tensor(100.0, dtype=dtype)
    n2 = torch.tensor(100.0, dtype=dtype)
    power = beignet.statistics.proportion_two_sample_power(
        p1, p2, n1, n2, alpha=0.05, alternative="two-sided"
    )

    # Should be somewhere between 0.15 and 0.8 for these parameters
    assert 0.15 < power < 0.8

    # Test edge cases - same proportions should give power ≈ alpha
    same_props = beignet.statistics.proportion_two_sample_power(
        p1, p1, n1, n2, alpha=0.05
    )
    assert torch.abs(same_props - 0.05) < 0.03

    # Skip statsmodels comparison - proportions_ztest_power function not available in statsmodels
    pass

    # Test edge cases
    # Test with very small proportions
    tiny_p1 = torch.tensor(0.001, dtype=dtype)
    tiny_p2 = torch.tensor(0.002, dtype=dtype)
    large_n = torch.tensor(1000.0, dtype=dtype)
    tiny_power = beignet.statistics.proportion_two_sample_power(
        tiny_p1, tiny_p2, large_n, large_n
    )
    assert 0.0 <= tiny_power <= 1.0

    # Test with proportions close to 1
    large_p1 = torch.tensor(0.998, dtype=dtype)
    large_p2 = torch.tensor(0.999, dtype=dtype)
    large_power = beignet.statistics.proportion_two_sample_power(
        large_p1, large_p2, large_n, large_n
    )
    assert 0.0 <= large_power <= 1.0

    # Test with very different sample sizes
    unequal_power = beignet.statistics.proportion_two_sample_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(0.6, dtype=dtype),
        torch.tensor(50.0, dtype=dtype),
        torch.tensor(500.0, dtype=dtype),
    )
    assert 0.0 <= unequal_power <= 1.0
