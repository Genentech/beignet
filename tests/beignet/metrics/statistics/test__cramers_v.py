import hypothesis
import hypothesis.strategies as st
import torch
from torchmetrics import Metric

import beignet.metrics.statistics


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=10),
    n_rows=st.integers(min_value=2, max_value=5),
    n_cols=st.integers(min_value=2, max_value=5),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_cramers_v(batch_size, n_rows, n_cols, dtype):
    """Test CramersV TorchMetrics class."""
    # Initialize the metric
    metric = beignet.metrics.statistics.CramersV()

    # Verify it's a proper TorchMetrics Metric
    assert isinstance(metric, Metric)

    # Create test inputs - contingency table
    contingency_table = torch.randint(
        1,
        50,
        (batch_size, n_rows, n_cols),
        dtype=torch.int64,
    ).to(dtype)

    # Test update method
    metric.update(contingency_table)

    # Test compute method
    result = metric.compute()

    # Verify output properties
    assert isinstance(result, torch.Tensor)
    assert result.shape == (batch_size,)
    assert result.dtype == dtype
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)  # Cramer's V is bounded [0, 1]

    # Test multiple updates
    new_table = torch.randint(
        1,
        50,
        (batch_size, n_rows, n_cols),
        dtype=torch.int64,
    ).to(dtype)
    metric.update(new_table)
    result2 = metric.compute()
    assert result2.shape == (batch_size,)

    # Test reset functionality
    metric.reset()

    # After reset, compute should raise an error
    try:
        metric.compute()
        raise AssertionError("Expected RuntimeError after reset")
    except RuntimeError:
        pass

    # Test metric state after reset and new update
    metric.update(contingency_table)
    result3 = metric.compute()
    assert result3.shape == (batch_size,)
    assert torch.allclose(result, result3, atol=1e-6)

    # Test with different dtypes
    if dtype == torch.float32:
        table_64 = contingency_table.to(torch.float64)
        metric_64 = beignet.metrics.statistics.CramersV()
        metric_64.update(table_64)
        result_64 = metric_64.compute()
        assert result_64.dtype == torch.float64

    # Test edge case - perfect independence
    uniform_table = torch.ones((batch_size, n_rows, n_cols), dtype=dtype) * 10
    metric_uniform = beignet.metrics.statistics.CramersV()
    metric_uniform.update(uniform_table)
    result_uniform = metric_uniform.compute()
    # Should be close to 0 for uniform distribution
    assert torch.all(result_uniform < 0.1)

    # Test repr
    repr_str = repr(metric)
    assert "CramersV" in repr_str

    # Test gradient computation
    table_grad = contingency_table.clone().requires_grad_(True)

    metric_grad = beignet.metrics.statistics.CramersV()
    metric_grad.update(table_grad)
    result_grad = metric_grad.compute()

    loss = result_grad.sum()
    loss.backward()

    assert table_grad.grad is not None
    assert table_grad.grad.shape == contingency_table.shape
