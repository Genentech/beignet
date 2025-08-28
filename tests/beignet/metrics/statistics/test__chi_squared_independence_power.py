import hypothesis
import hypothesis.strategies as st
import torch
from torchmetrics import Metric

import beignet.metrics.statistics


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=10),
    effect_size=st.floats(min_value=0.1, max_value=2.0),
    sample_size=st.integers(min_value=10, max_value=100),
    degrees_of_freedom=st.integers(min_value=1, max_value=10),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_chi_squared_independence_power(
    batch_size,
    effect_size,
    sample_size,
    degrees_of_freedom,
    alpha,
    dtype,
):
    """Test ChiSquaredIndependencePower TorchMetrics class."""
    # Initialize the metric
    metric = beignet.metrics.statistics.ChiSquaredIndependencePower(alpha=alpha)

    # Verify it's a proper TorchMetrics Metric
    assert isinstance(metric, Metric)

    # Create test inputs
    effect_size_tensor = torch.full((batch_size,), effect_size, dtype=dtype)
    sample_size_tensor = torch.full((batch_size,), sample_size, dtype=torch.int64)
    degrees_of_freedom_tensor = torch.full(
        (batch_size,),
        degrees_of_freedom,
        dtype=torch.int64,
    )

    # Test update method
    metric.update(effect_size_tensor, sample_size_tensor, degrees_of_freedom_tensor)

    # Test compute method
    result = metric.compute()

    # Verify output properties
    assert isinstance(result, torch.Tensor)
    assert result.shape == (batch_size,)
    assert result.dtype == dtype
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)

    # Test multiple updates
    metric.update(
        effect_size_tensor * 0.5,
        sample_size_tensor,
        degrees_of_freedom_tensor,
    )
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
    metric.update(effect_size_tensor, sample_size_tensor, degrees_of_freedom_tensor)
    result3 = metric.compute()
    assert result3.shape == (batch_size,)
    assert torch.allclose(result, result3, atol=1e-6)

    # Test with different dtypes
    if dtype == torch.float32:
        float64_effect = effect_size_tensor.to(torch.float64)
        metric_64 = beignet.metrics.statistics.ChiSquaredIndependencePower(alpha=alpha)
        metric_64.update(float64_effect, sample_size_tensor, degrees_of_freedom_tensor)
        result_64 = metric_64.compute()
        assert result_64.dtype == torch.float64

    # Test edge cases
    # Very small effect size should give low power
    small_effect = torch.full((batch_size,), 0.01, dtype=dtype)
    metric_small = beignet.metrics.statistics.ChiSquaredIndependencePower(alpha=alpha)
    metric_small.update(small_effect, sample_size_tensor, degrees_of_freedom_tensor)
    power_small = metric_small.compute()
    assert torch.all(power_small < 0.5)  # Should be low power

    # Large effect size should give high power
    large_effect = torch.full((batch_size,), 2.0, dtype=dtype)
    metric_large = beignet.metrics.statistics.ChiSquaredIndependencePower(alpha=alpha)
    metric_large.update(large_effect, sample_size_tensor, degrees_of_freedom_tensor)
    power_large = metric_large.compute()
    assert torch.all(power_large > 0.5)  # Should be higher power

    # Test repr
    repr_str = repr(metric)
    assert "ChiSquaredIndependencePower" in repr_str
    assert f"alpha={alpha}" in repr_str

    # Test device consistency
    if torch.cuda.is_available():
        device = torch.device("cuda")
        metric_cuda = beignet.metrics.statistics.ChiSquaredIndependencePower(
            alpha=alpha,
        ).to(device)
        effect_cuda = effect_size_tensor.to(device)
        sample_cuda = sample_size_tensor.to(device)
        df_cuda = degrees_of_freedom_tensor.to(device)

        metric_cuda.update(effect_cuda, sample_cuda, df_cuda)
        result_cuda = metric_cuda.compute()
        assert result_cuda.device == device

        # Results should be close between CPU and CUDA
        assert torch.allclose(result.cpu(), result_cuda.cpu(), atol=1e-5)

    # Test gradient computation
    effect_grad = effect_size_tensor.clone().requires_grad_(True)
    sample_grad = sample_size_tensor.clone().float().requires_grad_(True)
    df_grad = degrees_of_freedom_tensor.clone().float().requires_grad_(True)

    metric_grad = beignet.metrics.statistics.ChiSquaredIndependencePower(alpha=alpha)
    metric_grad.update(effect_grad, sample_grad, df_grad)
    result_grad = metric_grad.compute()

    loss = result_grad.sum()
    loss.backward()

    assert effect_grad.grad is not None
    assert sample_grad.grad is not None
    assert df_grad.grad is not None
