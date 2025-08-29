"""Test McnemarsTestSampleSize metric."""
import torch
from hypothesis import given, strategies as st
import pytest

from beignet.metrics.statistics import McnemarsTestSampleSize


@given(
    discordant_proportion=st.floats(min_value=0.05, max_value=0.5),
    power=st.floats(min_value=0.7, max_value=0.95),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64])
)
def test_mcnemars_test_sample_size(
    discordant_proportion, power, alpha, dtype
):
    metric = McnemarsTestSampleSize(power=power, alpha=alpha)
    
    discordant_prop_tensor = torch.tensor(discordant_proportion, dtype=dtype)
    
    metric.update(discordant_prop_tensor)
    result = metric.compute()
    
    assert isinstance(result, torch.Tensor)
    assert result.item() > 0
    
    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()