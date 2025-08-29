import hypothesis
import hypothesis.strategies as st
import torch
from torch import Tensor
from torchmetrics import Metric

import beignet.metrics.statistics


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=10),
    sample_size_group1=st.integers(min_value=5, max_value=30),
    sample_size_group2=st.integers(min_value=5, max_value=30),
    pooled=st.booleans(),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_cohens_d(batch_size, sample_size_group1, sample_size_group2, pooled, dtype):
    """Test CohensD TorchMetrics class."""
    # Initialize the metric
    metric = beignet.metrics.statistics.CohensD(pooled=pooled)

    # Verify it's a proper TorchMetrics Metric
    assert isinstance(metric, Metric)

    # Create test inputs
    group1 = torch.randn(batch_size, sample_size_group1, dtype=dtype)
    group2 = torch.randn(batch_size, sample_size_group2, dtype=dtype)

    # Test update method
    metric.update(group1, group2)

    # Test compute method
    result = metric.compute()

    # Verify output properties
    assert isinstance(result, Tensor)
    if batch_size == 1:
        assert result.shape == () or result.shape == (
            1,
        )  # Could be scalar or 1-element tensor
    else:
        assert result.shape == (batch_size,)
    assert result.dtype == dtype

    # Test multiple updates
    group1_new = torch.randn(batch_size, sample_size_group1, dtype=dtype)
    group2_new = torch.randn(batch_size, sample_size_group2, dtype=dtype)
    metric.update(group1_new, group2_new)
    metric.compute()  # Test that compute works after multiple updates

    # Test reset functionality
    metric.reset()

    # After reset, compute should raise an error
    try:
        metric.compute()
        raise AssertionError("Expected RuntimeError after reset")
    except RuntimeError:
        pass

    # Test metric state after reset and new update
    metric.update(group1, group2)
    result3 = metric.compute()
    if batch_size == 1:
        assert result3.shape == () or result3.shape == (1,)
    else:
        assert result3.shape == (batch_size,)
    assert torch.allclose(result, result3, atol=1e-6)

    # Test with different dtypes
    if dtype == torch.float32:
        float64_group1 = group1.to(torch.float64)
        float64_group2 = group2.to(torch.float64)
        metric_64 = beignet.metrics.statistics.CohensD(pooled=pooled)
        metric_64.update(float64_group1, float64_group2)
        result_64 = metric_64.compute()
        assert result_64.dtype == torch.float64

    # Test edge cases - constant groups should give Cohen's d near zero
    constant_group1 = torch.ones(batch_size, sample_size_group1, dtype=dtype)
    constant_group2 = torch.ones(batch_size, sample_size_group2, dtype=dtype)
    metric_identical = beignet.metrics.statistics.CohensD(pooled=pooled)
    metric_identical.update(constant_group1, constant_group2)
    result_identical = metric_identical.compute()
    # For constant groups with same values, Cohen's d should be 0 or NaN (due to zero variance)
    assert torch.all(torch.isnan(result_identical) | torch.eq(result_identical, 0))

    # Test symmetry: Cohen's d(A, B) = -Cohen's d(B, A) for non-constant groups
    # Create groups with clear different means
    group_a = torch.zeros(batch_size, sample_size_group1, dtype=dtype)
    group_b = torch.ones(batch_size, sample_size_group2, dtype=dtype)

    metric_forward = beignet.metrics.statistics.CohensD(pooled=pooled)
    metric_backward = beignet.metrics.statistics.CohensD(pooled=pooled)

    metric_forward.update(group_a, group_b)
    metric_backward.update(group_b, group_a)

    forward_result = metric_forward.compute()
    backward_result = metric_backward.compute()

    # Convert to same shape for comparison
    forward_flat = forward_result.flatten()
    backward_flat = backward_result.flatten()
    # Only test symmetry if results are not NaN (which can happen with zero variance)
    valid_mask = ~torch.isnan(forward_flat) & ~torch.isnan(backward_flat)
    if torch.any(valid_mask):
        assert torch.allclose(
            forward_flat[valid_mask],
            -backward_flat[valid_mask],
            atol=1e-5,
        )

    # Test repr
    repr_str = repr(metric)
    assert "CohensD" in repr_str
    assert f"pooled={pooled}" in repr_str

    # Test device consistency
    if torch.cuda.is_available():
        device = torch.device("cuda")
        metric_cuda = beignet.metrics.statistics.CohensD(pooled=pooled).to(device)
        group1_cuda = group1.to(device)
        group2_cuda = group2.to(device)

        metric_cuda.update(group1_cuda, group2_cuda)
        result_cuda = metric_cuda.compute()
        assert result_cuda.device == device

        # Results should be close between CPU and CUDA
        assert torch.allclose(result.cpu(), result_cuda.cpu(), atol=1e-5)

    # Test gradient computation
    group1_grad = group1.clone().requires_grad_(True)
    group2_grad = group2.clone().requires_grad_(True)

    metric_grad = beignet.metrics.statistics.CohensD(pooled=pooled)
    metric_grad.update(group1_grad, group2_grad)
    result_grad = metric_grad.compute()

    loss = result_grad.sum()
    loss.backward()

    assert group1_grad.grad is not None
    assert group2_grad.grad is not None
    assert group1_grad.grad.shape == group1.shape
    assert group2_grad.grad.shape == group2.shape

    # Test with different effect sizes
    different_group1 = torch.zeros(batch_size, sample_size_group1, dtype=dtype)
    different_group2 = torch.ones(batch_size, sample_size_group2, dtype=dtype) * 2.0

    metric_different = beignet.metrics.statistics.CohensD(pooled=pooled)
    metric_different.update(different_group1, different_group2)
    result_different = metric_different.compute()

    # Basic sanity check - result should be finite or NaN
    assert torch.all(torch.isfinite(result_different) | torch.isnan(result_different))
