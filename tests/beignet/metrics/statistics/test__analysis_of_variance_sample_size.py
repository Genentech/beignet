import hypothesis
import hypothesis.strategies as st
import torch
from torch import Tensor
from torchmetrics import Metric

import beignet.metrics.statistics


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=10),
    effect_size=st.floats(min_value=0.1, max_value=2.0),
    power=st.floats(min_value=0.1, max_value=0.95),
    groups=st.integers(min_value=3, max_value=8),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_analysis_of_variance_sample_size(
    batch_size,
    effect_size,
    power,
    groups,
    alpha,
    dtype,
):
    """Test AnalysisOfVarianceSampleSize TorchMetrics class."""
    # Initialize the metric
    metric = beignet.metrics.statistics.AnalysisOfVarianceSampleSize(alpha=alpha)

    # Verify it's a proper TorchMetrics Metric
    assert isinstance(metric, Metric)

    # Create test inputs
    effect_size_tensor = torch.full((batch_size,), effect_size, dtype=dtype)
    power_tensor = torch.full((batch_size,), power, dtype=dtype)
    groups_tensor = torch.full((batch_size,), groups, dtype=torch.int64)

    # Test update method
    metric.update(effect_size_tensor, power_tensor, groups_tensor)

    # Test compute method
    result = metric.compute()

    # Verify output properties
    assert isinstance(result, Tensor)
    assert result.shape == (batch_size,)
    assert result.dtype == dtype
    assert torch.all(result >= 3.0)  # Should be at least 3 for ANOVA

    # Test multiple updates
    metric.update(effect_size_tensor * 0.8, power_tensor, groups_tensor)
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
    metric.update(effect_size_tensor, power_tensor, groups_tensor)
    result3 = metric.compute()
    assert result3.shape == (batch_size,)
    assert torch.allclose(result, result3, atol=1e-6)

    # Test with different dtypes
    if dtype == torch.float32:
        float64_effect = effect_size_tensor.to(torch.float64)
        float64_power = power_tensor.to(torch.float64)
        metric_64 = beignet.metrics.statistics.AnalysisOfVarianceSampleSize(alpha=alpha)
        metric_64.update(float64_effect, float64_power, groups_tensor)
        result_64 = metric_64.compute()
        assert result_64.dtype == torch.float64

    # Test edge cases
    # Small effect size should require larger sample size
    small_effect = torch.full((batch_size,), 0.1, dtype=dtype)
    metric_small = beignet.metrics.statistics.AnalysisOfVarianceSampleSize(alpha=alpha)
    metric_small.update(small_effect, power_tensor, groups_tensor)
    size_small = metric_small.compute()

    # Large effect size should require smaller sample size
    large_effect = torch.full((batch_size,), 1.5, dtype=dtype)
    metric_large = beignet.metrics.statistics.AnalysisOfVarianceSampleSize(alpha=alpha)
    metric_large.update(large_effect, power_tensor, groups_tensor)
    size_large = metric_large.compute()

    assert torch.all(size_small > size_large)

    # Test repr
    repr_str = repr(metric)
    assert "AnalysisOfVarianceSampleSize" in repr_str
    assert f"alpha={alpha}" in repr_str

    # Test device consistency
    if torch.cuda.is_available():
        device = torch.device("cuda")
        metric_cuda = beignet.metrics.statistics.AnalysisOfVarianceSampleSize(
            alpha=alpha,
        ).to(device)
        effect_cuda = effect_size_tensor.to(device)
        power_cuda = power_tensor.to(device)
        groups_cuda = groups_tensor.to(device)

        metric_cuda.update(effect_cuda, power_cuda, groups_cuda)
        result_cuda = metric_cuda.compute()
        assert result_cuda.device == device

        # Results should be close between CPU and CUDA
        assert torch.allclose(result.cpu(), result_cuda.cpu(), atol=1e-5)

    # Test gradient computation
    effect_grad = effect_size_tensor.clone().requires_grad_(True)
    power_grad = power_tensor.clone().requires_grad_(True)
    groups_grad = groups_tensor.clone().float().requires_grad_(True)

    metric_grad = beignet.metrics.statistics.AnalysisOfVarianceSampleSize(alpha=alpha)
    metric_grad.update(effect_grad, power_grad, groups_grad)
    result_grad = metric_grad.compute()

    loss = result_grad.sum()
    loss.backward()

    assert effect_grad.grad is not None
    assert power_grad.grad is not None
    assert groups_grad.grad is not None
