import hypothesis
import hypothesis.strategies
import torch
from torch import Tensor
from torchmetrics import Metric

import beignet.metrics.statistics


@hypothesis.given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=10),
    sample_size_group1=hypothesis.strategies.integers(min_value=5, max_value=30),
    sample_size_group2=hypothesis.strategies.integers(min_value=5, max_value=30),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_hedges_g(batch_size, sample_size_group1, sample_size_group2, dtype):
    """Test HedgesG TorchMetrics class."""
    metric = beignet.metrics.statistics.HedgesG()
    assert isinstance(metric, Metric)

    group1 = torch.randn(batch_size, sample_size_group1, dtype=dtype)
    group2 = torch.randn(batch_size, sample_size_group2, dtype=dtype)

    metric.update(group1, group2)
    output = metric.compute()

    assert isinstance(output, Tensor)
    assert output.shape == (batch_size,)
    assert output.dtype == dtype

    # Test symmetry: Hedges' g(A, B) = -Hedges' g(B, A)
    metric_forward = beignet.metrics.statistics.HedgesG()
    metric_backward = beignet.metrics.statistics.HedgesG()

    metric_forward.update(group1, group2)
    metric_backward.update(group2, group1)

    forward_result = metric_forward.compute()
    backward_result = metric_backward.compute()

    assert torch.allclose(forward_result, -backward_result, atol=1e-6)

    # Test identical groups should give near zero
    identical_group1 = torch.ones(batch_size, sample_size_group1, dtype=dtype)
    identical_group2 = torch.ones(batch_size, sample_size_group2, dtype=dtype)
    metric_identical = beignet.metrics.statistics.HedgesG()
    metric_identical.update(identical_group1, identical_group2)
    result_identical = metric_identical.compute()
    assert torch.allclose(
        result_identical,
        torch.zeros_like(result_identical),
        atol=1e-6,
    )

    metric.reset()
    try:
        metric.compute()
        raise AssertionError("Expected RuntimeError after reset")
    except RuntimeError:
        pass

    repr_str = repr(metric)
    assert "HedgesG" in repr_str
