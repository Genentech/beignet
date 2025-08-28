import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet.statistics


@given(
    batch_size=st.integers(min_value=1, max_value=3),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_mann_whitney_u_test_power(batch_size, dtype):
    """Test Mann-Whitney U test power calculation."""

    # Generate test parameters - AUC should be > 0.5 for positive effect
    auc_values = (
        torch.tensor([0.6, 0.7, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    sample_sizes1 = (
        torch.tensor([15, 25, 35], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    sample_sizes2 = (
        torch.tensor([15, 25, 35], dtype=dtype).repeat(batch_size, 1).flatten()
    )

    # Test basic functionality
    result = beignet.statistics.mann_whitney_u_test_power(
        auc_values,
        sample_sizes1,
        sample_sizes2,
    )
    assert result.shape == auc_values.shape
    assert result.dtype == dtype
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)

    # Test with out parameter
    out = torch.empty_like(auc_values)
    result_out = beignet.statistics.mann_whitney_u_test_power(
        auc_values,
        sample_sizes1,
        sample_sizes2,
        out=out,
    )
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test that power increases with higher AUC
    small_auc = beignet.statistics.mann_whitney_u_test_power(
        torch.tensor(0.6, dtype=dtype),
        torch.tensor(25.0, dtype=dtype),
        torch.tensor(25.0, dtype=dtype),
    )
    large_auc = beignet.statistics.mann_whitney_u_test_power(
        torch.tensor(0.8, dtype=dtype),
        torch.tensor(25.0, dtype=dtype),
        torch.tensor(25.0, dtype=dtype),
    )
    assert large_auc > small_auc

    # Test that power increases with sample size
    small_n = beignet.statistics.mann_whitney_u_test_power(
        torch.tensor(0.7, dtype=dtype),
        torch.tensor(15.0, dtype=dtype),
        torch.tensor(15.0, dtype=dtype),
    )
    large_n = beignet.statistics.mann_whitney_u_test_power(
        torch.tensor(0.7, dtype=dtype),
        torch.tensor(35.0, dtype=dtype),
        torch.tensor(35.0, dtype=dtype),
    )
    assert large_n > small_n

    # Test different alpha values
    power_05 = beignet.statistics.mann_whitney_u_test_power(
        torch.tensor(0.7, dtype=dtype),
        torch.tensor(25.0, dtype=dtype),
        torch.tensor(25.0, dtype=dtype),
        alpha=0.05,
    )
    power_01 = beignet.statistics.mann_whitney_u_test_power(
        torch.tensor(0.7, dtype=dtype),
        torch.tensor(25.0, dtype=dtype),
        torch.tensor(25.0, dtype=dtype),
        alpha=0.01,
    )
    assert power_05 > power_01

    # Test gradient computation
    auc_grad = torch.tensor([0.7], dtype=dtype, requires_grad=True)
    n1_grad = torch.tensor([25.0], dtype=dtype, requires_grad=True)
    n2_grad = torch.tensor([25.0], dtype=dtype, requires_grad=True)
    result_grad = beignet.statistics.mann_whitney_u_test_power(
        auc_grad,
        n1_grad,
        n2_grad,
    )

    loss = result_grad.sum()
    loss.backward()

    assert auc_grad.grad is not None
    assert n1_grad.grad is not None
    assert n2_grad.grad is not None
    assert torch.all(torch.isfinite(auc_grad.grad))
    assert torch.all(torch.isfinite(n1_grad.grad))
    assert torch.all(torch.isfinite(n2_grad.grad))

    # Test torch.compile compatibility
    compiled_func = torch.compile(
        beignet.statistics.mann_whitney_u_test_power,
        fullgraph=True,
    )
    result_compiled = compiled_func(
        auc_values[:1],
        sample_sizes1[:1],
        sample_sizes2[:1],
    )
    result_normal = beignet.statistics.mann_whitney_u_test_power(
        auc_values[:1],
        sample_sizes1[:1],
        sample_sizes2[:1],
    )
    assert torch.allclose(result_compiled, result_normal, atol=1e-6)
