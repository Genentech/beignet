"""Test FTestMinimumDetectableEffect metric."""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from beignet.metrics.statistics import FTestMinimumDetectableEffect


@given(
    degrees_of_freedom_1=st.integers(min_value=1, max_value=10),
    degrees_of_freedom_2=st.integers(min_value=10, max_value=100),
    power=st.floats(min_value=0.7, max_value=0.95),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
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

    assert isinstance(result, torch.Tensor)
    assert result.item() >= 0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
