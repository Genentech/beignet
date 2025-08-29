"""Test MannWhitneyUTestPower metric."""
import torch
from hypothesis import given, strategies as st
import pytest

from beignet.metrics.statistics import MannWhitneyUTestPower


@given(
    effect_size=st.floats(min_value=0.1, max_value=1.0),
    sample_size_1=st.integers(min_value=10, max_value=100),
    sample_size_2=st.integers(min_value=10, max_value=100),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64])
)
def test_mann_whitney_u_test_power(
    effect_size, sample_size_1, sample_size_2, alpha, dtype
):
    metric = MannWhitneyUTestPower(alpha=alpha)
    
    effect_size_tensor = torch.tensor(effect_size, dtype=dtype)
    sample_size_1_tensor = torch.tensor(sample_size_1, dtype=dtype)
    sample_size_2_tensor = torch.tensor(sample_size_2, dtype=dtype)
    
    metric.update(effect_size_tensor, sample_size_1_tensor, sample_size_2_tensor)
    result = metric.compute()
    
    assert isinstance(result, torch.Tensor)
    assert 0.0 <= result.item() <= 1.0
    
    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()