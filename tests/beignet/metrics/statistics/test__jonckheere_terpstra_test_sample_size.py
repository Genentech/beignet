"""Test JonckheereTerpstraTestSampleSize metric."""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from beignet.metrics.statistics import JonckheereTerpstraTestSampleSize


@given(
    effect_size=st.floats(min_value=0.1, max_value=1.0),
    groups=st.integers(min_value=3, max_value=6),
    power=st.floats(min_value=0.7, max_value=0.95),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
def test_jonckheere_terpstra_test_sample_size(
    effect_size,
    groups,
    power,
    alpha,
    dtype,
):
    metric = JonckheereTerpstraTestSampleSize(power=power, alpha=alpha)

    effect_size_tensor = torch.tensor(effect_size, dtype=dtype)
    groups_tensor = torch.tensor(groups, dtype=dtype)

    metric.update(effect_size_tensor, groups_tensor)
    result = metric.compute()

    assert isinstance(result, torch.Tensor)
    assert result.item() > 0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
