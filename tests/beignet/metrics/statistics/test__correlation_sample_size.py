import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor
from torchmetrics import Metric

import beignet.metrics.statistics


@hypothesis.given(
    correlation=hypothesis.strategies.floats(min_value=0.1, max_value=0.9),
    power=hypothesis.strategies.floats(min_value=0.7, max_value=0.95),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_correlation_sample_size(correlation, power, alpha, dtype):
    metric = beignet.metrics.statistics.CorrelationSampleSize(power=power, alpha=alpha)
    assert isinstance(metric, Metric)

    correlation_tensor = torch.tensor(correlation, dtype=dtype)

    metric.update(correlation_tensor)
    output = metric.compute()

    assert isinstance(output, Tensor)
    assert output.item() >= 3.0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
