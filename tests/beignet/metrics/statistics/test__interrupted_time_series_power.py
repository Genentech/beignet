import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import InterruptedTimeSeriesPower


@hypothesis.given(
    effect_size=hypothesis.strategies.floats(min_value=0.1, max_value=1.0),
    n_time_points=hypothesis.strategies.integers(min_value=10, max_value=50),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_interrupted_time_series_power(
    effect_size,
    n_time_points,
    alpha,
    dtype,
):
    metric = InterruptedTimeSeriesPower(alpha=alpha)

    effect_size_tensor = torch.tensor(effect_size, dtype=dtype)
    n_time_points_tensor = torch.tensor(n_time_points, dtype=dtype)
    n_pre_intervention_tensor = torch.tensor(max(1, n_time_points // 2), dtype=dtype)

    metric.update(effect_size_tensor, n_time_points_tensor, n_pre_intervention_tensor)
    output = metric.compute()

    assert isinstance(output, Tensor)
    assert 0.0 <= output.item() <= 1.0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
