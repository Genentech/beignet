import hypothesis
import hypothesis.strategies
import torch
from torch import Tensor
from torchmetrics import Metric

import beignet.metrics.statistics


@hypothesis.given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=10),
    effect_size=hypothesis.strategies.floats(min_value=0.1, max_value=2.0),
    power=hypothesis.strategies.floats(min_value=0.1, max_value=0.95),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_z_test_sample_size(batch_size, effect_size, power, alpha, dtype):
    """Test ZTestSampleSize TorchMetrics class."""
    metric = beignet.metrics.statistics.ZTestSampleSize(alpha=alpha)
    assert isinstance(metric, Metric)

    effect_size_tensor = torch.full((batch_size,), effect_size, dtype=dtype)
    power_tensor = torch.full((batch_size,), power, dtype=dtype)

    metric.update(effect_size_tensor, power_tensor)
    output = metric.compute()

    assert isinstance(output, Tensor)
    if batch_size == 1:
        assert output.shape == ()  # Scalar output for single element
        assert output.item() >= 1.0
    else:
        assert output.shape == (batch_size,)
        assert torch.all(output >= 1.0)
    assert output.dtype == dtype

    repr_str = repr(metric)
    assert "ZTestSampleSize" in repr_str
