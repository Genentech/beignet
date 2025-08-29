"""Test IntraclassCorrelationPower metric."""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from beignet.metrics.statistics import IntraclassCorrelationPower


@given(
    icc=st.floats(min_value=0.1, max_value=0.9),
    sample_size=st.integers(min_value=20, max_value=200),
    groups=st.integers(min_value=3, max_value=10),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
def test_intraclass_correlation_power(
    icc,
    sample_size,
    groups,
    alpha,
    dtype,
):
    metric = IntraclassCorrelationPower(alpha=alpha)

    icc_tensor = torch.tensor(icc, dtype=dtype)
    sample_size_tensor = torch.tensor(sample_size, dtype=dtype)
    groups_tensor = torch.tensor(groups, dtype=dtype)

    metric.update(icc_tensor, sample_size_tensor, groups_tensor)
    result = metric.compute()

    assert isinstance(result, torch.Tensor)
    assert 0.0 <= result.item() <= 1.0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
