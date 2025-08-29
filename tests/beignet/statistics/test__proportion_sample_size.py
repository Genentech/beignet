import hypothesis
import hypothesis.strategies
import statsmodels  # noqa: F401
import torch

import beignet
import beignet.statistics


@hypothesis.given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=5),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_proportion_sample_size(batch_size, dtype):
    """Test proportion sample size calculation."""
    # Generate test parameters
    p0_values = (
        torch.tensor([0.4, 0.5, 0.6], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    p1_values = (
        torch.tensor([0.5, 0.6, 0.7], dtype=dtype).repeat(batch_size, 1).flatten()
    )

    # Test basic functionality - two-sided test
    result = beignet.statistics.proportion_sample_size(
        p0_values,
        p1_values,
        power=0.8,
        alpha=0.05,
        alternative="two-sided",
    )
    assert result.shape == p0_values.shape
    assert result.dtype == dtype
    assert torch.all(result >= 1.0)
    assert torch.all(result < 10000)  # Should be reasonable

    # Test one-sided tests
    result_greater = beignet.statistics.proportion_sample_size(
        p0_values,
        p1_values,
        power=0.8,
        alpha=0.05,
        alternative="greater",
    )
    result_less = beignet.statistics.proportion_sample_size(
        p0_values,
        p1_values,
        power=0.8,
        alpha=0.05,
        alternative="less",
    )

    assert torch.all(result_greater >= 1.0)
    assert torch.all(result_less >= 1.0)

    # One-sided tests should generally require smaller sample sizes than two-sided
    assert torch.all(result_greater <= result + 50)  # Allow some tolerance
    assert torch.all(result_less <= result + 50)

    # Test with out parameter
    out = torch.empty_like(p0_values)
    result_out = beignet.statistics.proportion_sample_size(
        p0_values,
        p1_values,
        power=0.8,
        alpha=0.05,
        alternative="two-sided",
        out=out,
    )
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test sample size decreases with effect size
    small_effect = beignet.statistics.proportion_sample_size(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(0.51, dtype=dtype),
        power=0.8,
    )
    large_effect = beignet.statistics.proportion_sample_size(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(0.6, dtype=dtype),
        power=0.8,
    )

    assert large_effect < small_effect

    # Test sample size increases with power
    low_power = beignet.statistics.proportion_sample_size(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(0.6, dtype=dtype),
        power=0.5,
    )
    high_power = beignet.statistics.proportion_sample_size(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(0.6, dtype=dtype),
        power=0.9,
    )

    assert high_power > low_power

    # Test gradient computation
    p0_grad = p0_values.clone().requires_grad_(True)
    p1_grad = p1_values.clone().requires_grad_(True)
    result_grad = beignet.statistics.proportion_sample_size(p0_grad, p1_grad, power=0.8)

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert p0_grad.grad is not None
    assert p1_grad.grad is not None

    # Test torch.compile compatibility
    compiled_proportion_sample_size = torch.compile(
        beignet.statistics.proportion_sample_size,
        fullgraph=True,
    )
    result_compiled = compiled_proportion_sample_size(p0_values, p1_values, power=0.8)
    assert torch.allclose(result, result_compiled, atol=1e-5)

    # Test invalid alternative
    try:
        beignet.statistics.proportion_sample_size(
            torch.tensor(0.5, dtype=dtype),
            torch.tensor(0.6, dtype=dtype),
            alternative="invalid",
        )
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass

    # Test against known theoretical values
    # For moderate effect (0.5 vs 0.6) with power=0.8, sample size should be reasonable
    p0 = torch.tensor(0.5, dtype=dtype)
    p1 = torch.tensor(0.6, dtype=dtype)
    n = beignet.statistics.proportion_sample_size(
        p0,
        p1,
        power=0.8,
        alpha=0.05,
        alternative="two-sided",
    )

    # Should be somewhere between 100 and 500 for these parameters
    assert 100 < n < 500

    # For large effect (0.5 vs 0.8), sample size should be smaller
    large_effect_n = beignet.statistics.proportion_sample_size(
        p0,
        torch.tensor(0.8, dtype=dtype),
        power=0.8,
    )
    assert large_effect_n < n

    # Test that sample size and power calculations are consistent
    # Calculate sample size for given power, then verify power with that sample size
    p0 = torch.tensor(0.5, dtype=dtype)
    p1 = torch.tensor(0.6, dtype=dtype)
    target_power = 0.8
    alpha = 0.05

    # Calculate required sample size
    n = beignet.statistics.proportion_sample_size(
        p0,
        p1,
        power=target_power,
        alpha=alpha,
    )

    # Calculate power with that sample size
    actual_power = beignet.statistics.proportion_power(p0, p1, n, alpha=alpha)

    # Should be close to target power (within tolerance for rounding)
    assert abs(float(actual_power) - target_power) < 0.12

    # Test edge cases
    # Test with very small effect size
    tiny_effect = beignet.statistics.proportion_sample_size(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(0.501, dtype=dtype),
        power=0.8,
    )
    assert tiny_effect > 1000  # Should require large sample size

    # Test with large effect size
    large_effect = beignet.statistics.proportion_sample_size(
        torch.tensor(0.2, dtype=dtype),
        torch.tensor(0.8, dtype=dtype),
        power=0.8,
    )
    assert large_effect < 100  # Should require small sample size

    # Test with very high power
    high_power = beignet.statistics.proportion_sample_size(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(0.6, dtype=dtype),
        power=0.99,
    )
    normal_power = beignet.statistics.proportion_sample_size(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(0.6, dtype=dtype),
        power=0.8,
    )
    assert high_power > normal_power
