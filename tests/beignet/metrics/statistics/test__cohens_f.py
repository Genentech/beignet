import hypothesis
import hypothesis.strategies as st
import torch
from torch import Tensor
from torchmetrics import Metric

import beignet.metrics.statistics


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=10),
    sample_size_per_group=st.integers(min_value=5, max_value=20),
    num_groups=st.integers(min_value=3, max_value=6),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_cohens_f(batch_size, sample_size_per_group, num_groups, dtype):
    """Test CohensF TorchMetrics class."""
    # Initialize the metric
    metric = beignet.metrics.statistics.CohensF()

    # Verify it's a proper TorchMetrics Metric
    assert isinstance(metric, Metric)

    # Create test inputs - list of groups with different means
    groups = []
    for i in range(num_groups):
        group = torch.randn(batch_size, sample_size_per_group, dtype=dtype) + float(i)
        groups.append(group)

    # Test update method
    metric.update(groups)

    # Test compute method
    result = metric.compute()

    # Verify output properties
    assert isinstance(result, Tensor)
    assert result.shape == (batch_size,)
    assert result.dtype == dtype
    assert torch.all(result >= 0.0)  # Cohen's f should be non-negative

    # Test multiple updates
    groups_new = []
    for i in range(num_groups):
        group = (
            torch.randn(batch_size, sample_size_per_group, dtype=dtype) + float(i) * 0.5
        )
        groups_new.append(group)

    metric.update(groups_new)
    result2 = metric.compute()

    # Should now have accumulated both batches
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
    metric.update(groups)
    result3 = metric.compute()
    assert result3.shape == (batch_size,)
    assert torch.allclose(result, result3, atol=1e-6)

    # Test with different dtypes
    if dtype == torch.float32:
        groups_64 = [group.to(torch.float64) for group in groups]
        metric_64 = beignet.metrics.statistics.CohensF()
        metric_64.update(groups_64)
        result_64 = metric_64.compute()
        assert result_64.dtype == torch.float64

    # Test edge cases - identical groups should give Cohen's f near zero
    identical_groups = []
    for _ in range(num_groups):
        group = torch.ones(batch_size, sample_size_per_group, dtype=dtype)
        identical_groups.append(group)

    metric_identical = beignet.metrics.statistics.CohensF()
    metric_identical.update(identical_groups)
    result_identical = metric_identical.compute()
    assert torch.allclose(
        result_identical,
        torch.zeros_like(result_identical),
        atol=1e-6,
    )

    # Test with large effect - groups with very different means
    large_effect_groups = []
    for i in range(num_groups):
        group = torch.full(
            (batch_size, sample_size_per_group),
            float(i * 3),
            dtype=dtype,
        )
        large_effect_groups.append(group)

    metric_large = beignet.metrics.statistics.CohensF()
    metric_large.update(large_effect_groups)
    result_large = metric_large.compute()
    assert torch.all(result_large > result)  # Should be larger effect

    # Test repr
    repr_str = repr(metric)
    assert "CohensF" in repr_str

    # Test device consistency
    if torch.cuda.is_available():
        device = torch.device("cuda")
        metric_cuda = beignet.metrics.statistics.CohensF().to(device)
        groups_cuda = [group.to(device) for group in groups]

        metric_cuda.update(groups_cuda)
        result_cuda = metric_cuda.compute()
        assert result_cuda.device == device

        # Results should be close between CPU and CUDA
        assert torch.allclose(result.cpu(), result_cuda.cpu(), atol=1e-5)

    # Test gradient computation
    groups_grad = [group.clone().requires_grad_(True) for group in groups]

    metric_grad = beignet.metrics.statistics.CohensF()
    metric_grad.update(groups_grad)
    result_grad = metric_grad.compute()

    loss = result_grad.sum()
    loss.backward()

    for group_grad in groups_grad:
        assert group_grad.grad is not None
        assert group_grad.grad.shape == group_grad.shape
