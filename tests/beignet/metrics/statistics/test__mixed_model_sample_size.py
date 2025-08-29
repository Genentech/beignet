"""Test MixedModelSampleSize metric."""
import torch
from hypothesis import given, strategies as st
import pytest

from beignet.metrics.statistics import MixedModelSampleSize


@given(
    effect_size=st.floats(min_value=0.1, max_value=1.0),
    cluster_size=st.integers(min_value=5, max_value=50),
    intraclass_correlation=st.floats(min_value=0.01, max_value=0.5),
    power=st.floats(min_value=0.7, max_value=0.95),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64])
)
def test_mixed_model_sample_size(
    effect_size, cluster_size, intraclass_correlation, power, alpha, dtype
):
    metric = MixedModelSampleSize(power=power, alpha=alpha)
    
    effect_size_tensor = torch.tensor(effect_size, dtype=dtype)
    cluster_size_tensor = torch.tensor(cluster_size, dtype=dtype)
    icc_tensor = torch.tensor(intraclass_correlation, dtype=dtype)
    
    metric.update(effect_size_tensor, cluster_size_tensor, icc_tensor)
    result = metric.compute()
    
    assert isinstance(result, torch.Tensor)
    assert result.item() > 0
    
    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()