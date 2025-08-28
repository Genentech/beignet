import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet.statistics


@given(
    batch_size=st.integers(min_value=1, max_value=3),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_paired_t_test_power(batch_size, dtype):
    """Test paired t-test power calculation."""

    # Generate test parameters
    effect_sizes = (
        torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    sample_sizes = (
        torch.tensor([10, 30, 50], dtype=dtype).repeat(batch_size, 1).flatten()
    )

    # Test basic functionality
    result = beignet.statistics.paired_t_test_power(effect_sizes, sample_sizes)
    assert result.shape == effect_sizes.shape
    assert result.dtype == dtype
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)

    # Test with out parameter
    out = torch.empty_like(effect_sizes)
    result_out = beignet.statistics.paired_t_test_power(
        effect_sizes,
        sample_sizes,
        out=out,
    )
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test that power increases with effect size
    small_effect = beignet.statistics.paired_t_test_power(
        torch.tensor(0.2, dtype=dtype),
        torch.tensor(30.0, dtype=dtype),
    )
    large_effect = beignet.statistics.paired_t_test_power(
        torch.tensor(0.8, dtype=dtype),
        torch.tensor(30.0, dtype=dtype),
    )
    assert large_effect > small_effect

    # Test that power increases with sample size
    small_n = beignet.statistics.paired_t_test_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(10.0, dtype=dtype),
    )
    large_n = beignet.statistics.paired_t_test_power(
        torch.tensor(0.5, dtype=dtype),
        torch.tensor(50.0, dtype=dtype),
    )
    assert large_n > small_n

    # Test different alternatives
    for alt in ["two-sided", "greater", "less"]:
        result_alt = beignet.statistics.paired_t_test_power(
            torch.tensor(0.5, dtype=dtype),
            torch.tensor(30.0, dtype=dtype),
            alternative=alt,
        )
        assert result_alt >= 0.0 and result_alt <= 1.0

    # Test gradient computation
    effect_grad = torch.tensor([0.5], dtype=dtype, requires_grad=True)
    sample_grad = torch.tensor([30.0], dtype=dtype, requires_grad=True)
    result_grad = beignet.statistics.paired_t_test_power(effect_grad, sample_grad)

    loss = result_grad.sum()
    loss.backward()

    assert effect_grad.grad is not None
    assert sample_grad.grad is not None
    assert torch.all(torch.isfinite(effect_grad.grad))
    assert torch.all(torch.isfinite(sample_grad.grad))

    # Test torch.compile compatibility
    compiled_func = torch.compile(
        beignet.statistics.paired_t_test_power,
        fullgraph=True,
    )
    result_compiled = compiled_func(effect_sizes[:1], sample_sizes[:1])
    result_normal = beignet.statistics.paired_t_test_power(
        effect_sizes[:1],
        sample_sizes[:1],
    )
    assert torch.allclose(result_compiled, result_normal, atol=1e-6)
