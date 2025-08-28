import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet.statistics


@given(
    batch_size=st.integers(min_value=1, max_value=3),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_mcnemars_test_power(batch_size, dtype):
    """Test McNemar's test power calculation."""

    # Generate test parameters
    p01_values = (
        torch.tensor([0.1, 0.2, 0.3], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    p10_values = (
        torch.tensor([0.05, 0.1, 0.15], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    sample_sizes = (
        torch.tensor([50, 100, 200], dtype=dtype).repeat(batch_size, 1).flatten()
    )

    # Test basic functionality
    result = beignet.statistics.mcnemars_test_power(
        p01_values,
        p10_values,
        sample_sizes,
    )
    assert result.shape == p01_values.shape
    assert result.dtype == dtype
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)

    # Test with out parameter
    out = torch.empty_like(p01_values)
    result_out = beignet.statistics.mcnemars_test_power(
        p01_values,
        p10_values,
        sample_sizes,
        out=out,
    )
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test that power increases with effect size (difference between p01 and p10)
    small_diff = beignet.statistics.mcnemars_test_power(
        torch.tensor(0.15, dtype=dtype),
        torch.tensor(0.1, dtype=dtype),
        torch.tensor(100.0, dtype=dtype),
    )
    large_diff = beignet.statistics.mcnemars_test_power(
        torch.tensor(0.25, dtype=dtype),
        torch.tensor(0.1, dtype=dtype),
        torch.tensor(100.0, dtype=dtype),
    )
    assert large_diff > small_diff

    # Test that power increases with sample size
    small_n = beignet.statistics.mcnemars_test_power(
        torch.tensor(0.2, dtype=dtype),
        torch.tensor(0.1, dtype=dtype),
        torch.tensor(50.0, dtype=dtype),
    )
    large_n = beignet.statistics.mcnemars_test_power(
        torch.tensor(0.2, dtype=dtype),
        torch.tensor(0.1, dtype=dtype),
        torch.tensor(200.0, dtype=dtype),
    )
    assert large_n > small_n

    # Test two-sided vs one-sided
    power_two_sided = beignet.statistics.mcnemars_test_power(
        torch.tensor(0.2, dtype=dtype),
        torch.tensor(0.1, dtype=dtype),
        torch.tensor(100.0, dtype=dtype),
        two_sided=True,
    )
    power_one_sided = beignet.statistics.mcnemars_test_power(
        torch.tensor(0.2, dtype=dtype),
        torch.tensor(0.1, dtype=dtype),
        torch.tensor(100.0, dtype=dtype),
        two_sided=False,
    )
    assert power_one_sided > power_two_sided

    # Test different alpha values
    power_05 = beignet.statistics.mcnemars_test_power(
        torch.tensor(0.2, dtype=dtype),
        torch.tensor(0.1, dtype=dtype),
        torch.tensor(100.0, dtype=dtype),
        alpha=0.05,
    )
    power_01 = beignet.statistics.mcnemars_test_power(
        torch.tensor(0.2, dtype=dtype),
        torch.tensor(0.1, dtype=dtype),
        torch.tensor(100.0, dtype=dtype),
        alpha=0.01,
    )
    assert power_05 > power_01

    # Test gradient computation
    p01_grad = torch.tensor([0.2], dtype=dtype, requires_grad=True)
    p10_grad = torch.tensor([0.1], dtype=dtype, requires_grad=True)
    n_grad = torch.tensor([100.0], dtype=dtype, requires_grad=True)
    result_grad = beignet.statistics.mcnemars_test_power(p01_grad, p10_grad, n_grad)

    loss = result_grad.sum()
    loss.backward()

    assert p01_grad.grad is not None
    assert p10_grad.grad is not None
    assert n_grad.grad is not None
    assert torch.all(torch.isfinite(p01_grad.grad))
    assert torch.all(torch.isfinite(p10_grad.grad))
    assert torch.all(torch.isfinite(n_grad.grad))

    # Test torch.compile compatibility
    compiled_func = torch.compile(
        beignet.statistics.mcnemars_test_power,
        fullgraph=True,
    )
    result_compiled = compiled_func(p01_values[:1], p10_values[:1], sample_sizes[:1])
    result_normal = beignet.statistics.mcnemars_test_power(
        p01_values[:1],
        p10_values[:1],
        sample_sizes[:1],
    )
    assert torch.allclose(result_compiled, result_normal, atol=1e-6)
