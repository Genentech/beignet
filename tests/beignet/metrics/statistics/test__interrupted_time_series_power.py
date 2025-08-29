"""Test InterruptedTimeSeriesPower metric."""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from beignet.metrics.statistics import InterruptedTimeSeriesPower


@given(
    effect_size=st.floats(min_value=0.1, max_value=1.0),
    sample_size=st.integers(min_value=30, max_value=200),
    time_points=st.integers(min_value=10, max_value=50),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
def test_interrupted_time_series_power(
    effect_size,
    sample_size,
    time_points,
    alpha,
    dtype,
):
    metric = InterruptedTimeSeriesPower(alpha=alpha)

    effect_size_tensor = torch.tensor(effect_size, dtype=dtype)
    sample_size_tensor = torch.tensor(sample_size, dtype=dtype)
    time_points_tensor = torch.tensor(time_points, dtype=dtype)

    metric.update(effect_size_tensor, sample_size_tensor, time_points_tensor)
    result = metric.compute()

    assert isinstance(result, torch.Tensor)
    assert 0.0 <= result.item() <= 1.0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
