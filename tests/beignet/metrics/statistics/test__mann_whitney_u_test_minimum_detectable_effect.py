import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import MannWhitneyUTestMinimumDetectableEffect


@hypothesis.given(
    sample_size_1=hypothesis.strategies.integers(min_value=10, max_value=100),
    sample_size_2=hypothesis.strategies.integers(min_value=10, max_value=100),
    power=hypothesis.strategies.floats(min_value=0.7, max_value=0.95),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_mann_whitney_u_test_minimum_detectable_effect(
    sample_size_1,
    sample_size_2,
    power,
    alpha,
    dtype,
):
    metric = MannWhitneyUTestMinimumDetectableEffect(power=power, alpha=alpha)

    sample_size_1_tensor = torch.tensor(sample_size_1, dtype=dtype)
    sample_size_2_tensor = torch.tensor(sample_size_2, dtype=dtype)

    metric.update(sample_size_1_tensor, sample_size_2_tensor)
    output = metric.compute()

    assert isinstance(output, Tensor)
    assert output.item() >= 0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
