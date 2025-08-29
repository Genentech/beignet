"""Test IntraclassCorrelationSampleSize metric."""
import torch
from hypothesis import given, strategies as st
import pytest

from beignet.metrics.statistics import IntraclassCorrelationSampleSize


@given(
    icc=st.floats(min_value=0.1, max_value=0.9),
    groups=st.integers(min_value=3, max_value=10),
    power=st.floats(min_value=0.7, max_value=0.95),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64])
)
def test_intraclass_correlation_sample_size(
    icc, groups, power, alpha, dtype
):
    metric = IntraclassCorrelationSampleSize(power=power, alpha=alpha)
    
    icc_tensor = torch.tensor(icc, dtype=dtype)
    groups_tensor = torch.tensor(groups, dtype=dtype)
    
    metric.update(icc_tensor, groups_tensor)
    result = metric.compute()
    
    assert isinstance(result, torch.Tensor)
    assert result.item() > 0
    
    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()