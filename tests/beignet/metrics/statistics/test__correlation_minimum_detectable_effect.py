import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import CorrelationMinimumDetectableEffect


@hypothesis.given(
    sample_size=hypothesis.strategies.integers(min_value=10, max_value=200),
    power=hypothesis.strategies.floats(min_value=0.7, max_value=0.95),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_correlation_minimum_detectable_effect(
    sample_size,
    power,
    alpha,
    dtype,
):
    metric = CorrelationMinimumDetectableEffect(power=power, alpha=alpha)

    sample_size_tensor = torch.tensor(sample_size, dtype=dtype)

    metric.update(sample_size_tensor)
    output = metric.compute()

    assert isinstance(output, Tensor)
    assert 0.0 <= output.item() <= 1.0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
