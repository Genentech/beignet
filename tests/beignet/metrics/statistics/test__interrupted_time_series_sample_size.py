import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import InterruptedTimeSeriesSampleSize


@hypothesis.given(
    effect_size=hypothesis.strategies.floats(min_value=0.1, max_value=1.0),
    time_points=hypothesis.strategies.integers(min_value=10, max_value=50),
    power=hypothesis.strategies.floats(min_value=0.7, max_value=0.95),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_interrupted_time_series_sample_size(
    effect_size,
    time_points,
    power,
    alpha,
    dtype,
):
    metric = InterruptedTimeSeriesSampleSize(power=power, alpha=alpha)

    effect_size_tensor = torch.tensor(effect_size, dtype=dtype)
    time_points_tensor = torch.tensor(time_points, dtype=dtype)

    metric.update(effect_size_tensor, time_points_tensor)
    result = metric.compute()

    assert isinstance(result, Tensor)
    assert result.item() > 0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
