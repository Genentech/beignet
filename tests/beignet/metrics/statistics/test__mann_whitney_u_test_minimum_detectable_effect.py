"""Test MannWhitneyUTestMinimumDetectableEffect metric."""
import torch
from hypothesis import given, strategies as st
import pytest

from beignet.metrics.statistics import MannWhitneyUTestMinimumDetectableEffect


@given(
    sample_size_1=st.integers(min_value=10, max_value=100),
    sample_size_2=st.integers(min_value=10, max_value=100),
    power=st.floats(min_value=0.7, max_value=0.95),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64])
)
def test_mann_whitney_u_test_minimum_detectable_effect(
    sample_size_1, sample_size_2, power, alpha, dtype
):
    metric = MannWhitneyUTestMinimumDetectableEffect(power=power, alpha=alpha)
    
    sample_size_1_tensor = torch.tensor(sample_size_1, dtype=dtype)
    sample_size_2_tensor = torch.tensor(sample_size_2, dtype=dtype)
    
    metric.update(sample_size_1_tensor, sample_size_2_tensor)
    result = metric.compute()
    
    assert isinstance(result, torch.Tensor)
    assert result.item() >= 0
    
    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()