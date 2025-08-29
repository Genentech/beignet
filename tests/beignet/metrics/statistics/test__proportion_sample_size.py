import hypothesis
import hypothesis.strategies
import torch
from torch import Tensor
from torchmetrics import Metric

import beignet.metrics.statistics


@hypothesis.given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=10),
    p0=hypothesis.strategies.floats(min_value=0.1, max_value=0.9),
    p1=hypothesis.strategies.floats(min_value=0.1, max_value=0.9),
    power=hypothesis.strategies.floats(min_value=0.1, max_value=0.95),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_proportion_sample_size(batch_size, p0, p1, power, alpha, dtype):
    """Test ProportionSampleSize TorchMetrics class."""
    # ProportionSampleSize constructor accepts power and alpha parameters
    metric = beignet.metrics.statistics.ProportionSampleSize(power=power, alpha=alpha)
    assert isinstance(metric, Metric)

    # ProportionSampleSize.update expects (p0, p1)
    p0_tensor = torch.full((batch_size,), p0, dtype=dtype)
    p1_tensor = torch.full((batch_size,), p1, dtype=dtype)

    metric.update(p0_tensor, p1_tensor)
    output = metric.compute()

    assert isinstance(output, Tensor)
    # The metric returns the shape of the most recent update call
    assert output.dtype == dtype
    assert torch.all(output >= 1.0)  # Sample size should be at least 1
    # Verify output has reasonable dimensions (scalar or 1D tensor)
    assert output.ndim <= 1

    metric.reset()
    try:
        metric.compute()
        raise AssertionError("Expected RuntimeError after reset")
    except RuntimeError:
        pass

    repr_str = repr(metric)
    assert "ProportionSampleSize" in repr_str
