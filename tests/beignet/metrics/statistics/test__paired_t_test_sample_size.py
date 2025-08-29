"""Test PairedTTestSampleSize metric."""
import torch
from hypothesis import given, strategies as st
import pytest

from beignet.metrics.statistics import PairedTTestSampleSize


@given(
    effect_size=st.floats(min_value=0.1, max_value=1.5),
    power=st.floats(min_value=0.7, max_value=0.95),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64])
)
def test_paired_t_test_sample_size(effect_size, power, alpha, dtype):
    metric = PairedTTestSampleSize(power=power, alpha=alpha)
    
    effect_size_tensor = torch.tensor(effect_size, dtype=dtype)
    
    metric.update(effect_size_tensor)
    result = metric.compute()
    
    assert isinstance(result, torch.Tensor)
    assert result.item() > 0
    
    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()