import hypothesis
import hypothesis.strategies
import torch
from torch import Tensor
from torchmetrics import Metric

import beignet.metrics.statistics


@hypothesis.given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=10),
    effect_size=hypothesis.strategies.floats(min_value=0.1, max_value=0.8),
    power=hypothesis.strategies.floats(min_value=0.1, max_value=0.95),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_proportion_two_sample_sample_size(
    batch_size,
    effect_size,
    power,
    alpha,
    dtype,
):
    """Test ProportionTwoSampleSampleSize TorchMetrics class."""
    metric = beignet.metrics.statistics.ProportionTwoSampleSampleSize(alpha=alpha)
    assert isinstance(metric, Metric)

    effect_size_tensor = torch.full((batch_size,), effect_size, dtype=dtype)
    power_tensor = torch.full((batch_size,), power, dtype=dtype)

    metric.update(effect_size_tensor, power_tensor)
    output = metric.compute()

    assert isinstance(output, Tensor)
    assert output.shape == (batch_size,)
    assert output.dtype == dtype
    assert torch.all(output >= 2.0)

    repr_str = repr(metric)
    assert "ProportionTwoSampleSampleSize" in repr_str
