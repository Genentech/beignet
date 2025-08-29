import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor
from torchmetrics import Metric

import beignet.metrics.statistics


@hypothesis.given(
    correlation=hypothesis.strategies.floats(min_value=0.1, max_value=0.9),
    sample_size=hypothesis.strategies.integers(min_value=10, max_value=100),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_correlation_power(correlation, sample_size, alpha, dtype):
    metric = beignet.metrics.statistics.CorrelationPower(alpha=alpha)
    assert isinstance(metric, Metric)

    correlation_tensor = torch.tensor(correlation, dtype=dtype)
    sample_size_tensor = torch.tensor(sample_size, dtype=dtype)

    metric.update(correlation_tensor, sample_size_tensor)
    output = metric.compute()

    assert isinstance(output, Tensor)
    assert 0.0 <= output.item() <= 1.0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
