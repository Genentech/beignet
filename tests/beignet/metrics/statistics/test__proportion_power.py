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
    sample_size=hypothesis.strategies.integers(min_value=10, max_value=100),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_proportion_power(batch_size, p0, p1, sample_size, alpha, dtype):
    """Test ProportionPower TorchMetrics class."""
    # ProportionPower constructor requires p0 parameter
    metric = beignet.metrics.statistics.ProportionPower(p0=p0, alpha=alpha)
    assert isinstance(metric, Metric)

    # ProportionPower.update expects (p1, sample_size)
    p1_tensor = torch.full((batch_size,), p1, dtype=dtype)
    sample_size_tensor = torch.full((batch_size,), sample_size, dtype=torch.int64)

    metric.update(p1_tensor, sample_size_tensor)
    output = metric.compute()

    assert isinstance(output, Tensor)
    # The metric returns the shape of the most recent update call
    assert output.dtype == dtype
    assert torch.all(output >= 0.0)
    assert torch.all(output <= 1.0)
    # Verify output has reasonable dimensions (scalar or 1D tensor)
    assert output.ndim <= 1

    metric.reset()
    try:
        metric.compute()
        raise AssertionError("Expected RuntimeError after reset")
    except RuntimeError:
        pass

    repr_str = repr(metric)
    assert "ProportionPower" in repr_str
