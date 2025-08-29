"""Test PartialEtaSquared metric."""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from beignet.metrics.statistics import PartialEtaSquared


@given(
    sum_squares_effect=st.floats(min_value=1.0, max_value=100.0),
    sum_squares_error=st.floats(min_value=1.0, max_value=100.0),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
def test_partial_eta_squared(sum_squares_effect, sum_squares_error, dtype):
    metric = PartialEtaSquared()

    ss_effect_tensor = torch.tensor(sum_squares_effect, dtype=dtype)
    ss_error_tensor = torch.tensor(sum_squares_error, dtype=dtype)

    metric.update(ss_effect_tensor, ss_error_tensor)
    result = metric.compute()

    assert isinstance(result, torch.Tensor)
    assert 0.0 <= result.item() <= 1.0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
