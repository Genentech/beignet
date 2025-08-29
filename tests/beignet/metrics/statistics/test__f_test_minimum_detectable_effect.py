import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import FTestMinimumDetectableEffect


@hypothesis.given(
    degrees_of_freedom_1=hypothesis.strategies.integers(min_value=1, max_value=10),
    degrees_of_freedom_2=hypothesis.strategies.integers(min_value=10, max_value=100),
    power=hypothesis.strategies.floats(min_value=0.7, max_value=0.95),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_f_test_minimum_detectable_effect(
    degrees_of_freedom_1,
    degrees_of_freedom_2,
    power,
    alpha,
    dtype,
):
    metric = FTestMinimumDetectableEffect(power=power, alpha=alpha)

    df1_tensor = torch.tensor(degrees_of_freedom_1, dtype=dtype)
    df2_tensor = torch.tensor(degrees_of_freedom_2, dtype=dtype)

    metric.update(df1_tensor, df2_tensor)
    result = metric.compute()

    assert isinstance(result, Tensor)
    assert result.item() >= 0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
