"""Test McnemarsTestMinimumDetectableEffect metric."""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from beignet.metrics.statistics import McnemarsTestMinimumDetectableEffect


@given(
    sample_size=st.integers(min_value=20, max_value=200),
    power=st.floats(min_value=0.7, max_value=0.95),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
def test_mcnemars_test_minimum_detectable_effect(
    sample_size,
    power,
    alpha,
    dtype,
):
    metric = McnemarsTestMinimumDetectableEffect(power=power, alpha=alpha)

    sample_size_tensor = torch.tensor(sample_size, dtype=dtype)

    metric.update(sample_size_tensor)
    result = metric.compute()

    assert isinstance(result, torch.Tensor)
    assert result.item() >= 0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
