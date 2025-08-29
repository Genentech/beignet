import hypothesis
import hypothesis.strategies
import torch
from torch import Tensor
from torchmetrics import Metric

import beignet.metrics.statistics


@hypothesis.given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=10),
    effect_size=hypothesis.strategies.floats(min_value=0.1, max_value=0.8),
    sample_size=hypothesis.strategies.integers(min_value=10, max_value=100),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_proportion_two_sample_power(
    batch_size,
    effect_size,
    sample_size,
    alpha,
    dtype,
):
    """Test ProportionTwoSamplePower TorchMetrics class."""
    metric = beignet.metrics.statistics.ProportionTwoSamplePower(alpha=alpha)
    assert isinstance(metric, Metric)

    effect_size_tensor = torch.full((batch_size,), effect_size, dtype=dtype)
    sample_size_tensor = torch.full((batch_size,), sample_size, dtype=torch.int64)

    metric.update(effect_size_tensor, sample_size_tensor)
    output = metric.compute()

    assert isinstance(output, Tensor)
    assert output.shape == (batch_size,)
    assert output.dtype == dtype
    assert torch.all(output >= 0.0)
    assert torch.all(output <= 1.0)

    repr_str = repr(metric)
    assert "ProportionTwoSamplePower" in repr_str
