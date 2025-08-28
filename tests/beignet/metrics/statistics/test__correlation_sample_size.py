import hypothesis
import hypothesis.strategies as st
import torch
from torchmetrics import Metric

import beignet.metrics.statistics


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=10),
    effect_size=st.floats(min_value=0.1, max_value=0.9),
    power=st.floats(min_value=0.1, max_value=0.95),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_correlation_sample_size(batch_size, effect_size, power, alpha, dtype):
    """Test CorrelationSampleSize TorchMetrics class."""
    # Initialize the metric
    metric = beignet.metrics.statistics.CorrelationSampleSize(alpha=alpha)

    # Verify it's a proper TorchMetrics Metric
    assert isinstance(metric, Metric)

    # Create test inputs
    effect_size_tensor = torch.full((batch_size,), effect_size, dtype=dtype)
    power_tensor = torch.full((batch_size,), power, dtype=dtype)

    # Test update method
    metric.update(effect_size_tensor, power_tensor)

    # Test compute method
    result = metric.compute()

    # Verify output properties
    assert isinstance(result, torch.Tensor)
    assert result.shape == (batch_size,)
    assert result.dtype == dtype
    assert torch.all(result >= 3.0)  # Minimum for correlation

    # Test multiple updates
    metric.update(effect_size_tensor * 0.8, power_tensor)
    result2 = metric.compute()
    assert result2.shape == (batch_size * 2,)

    # Test reset functionality
    metric.reset()

    # After reset, compute should raise an error
    try:
        metric.compute()
        raise AssertionError("Expected RuntimeError after reset")
    except RuntimeError:
        pass

    # Test metric state after reset and new update
    metric.update(effect_size_tensor, power_tensor)
    result3 = metric.compute()
    assert result3.shape == (batch_size,)
    assert torch.allclose(result, result3, atol=1e-6)

    # Test repr
    repr_str = repr(metric)
    assert "CorrelationSampleSize" in repr_str
    assert f"alpha={alpha}" in repr_str

    # Test gradient computation
    effect_grad = effect_size_tensor.clone().requires_grad_(True)
    power_grad = power_tensor.clone().requires_grad_(True)

    metric_grad = beignet.metrics.statistics.CorrelationSampleSize(alpha=alpha)
    metric_grad.update(effect_grad, power_grad)
    result_grad = metric_grad.compute()

    loss = result_grad.sum()
    loss.backward()

    assert effect_grad.grad is not None
    assert power_grad.grad is not None
