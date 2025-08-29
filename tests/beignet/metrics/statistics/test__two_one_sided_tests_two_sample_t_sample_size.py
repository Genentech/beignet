"""Test TwoOneSidedTestsTwoSampleTSampleSize metric."""
import torch
from hypothesis import given, strategies as st
import pytest

from beignet.metrics.statistics import TwoOneSidedTestsTwoSampleTSampleSize


@given(
    effect_size=st.floats(min_value=0.1, max_value=1.0),
    equivalence_margin=st.floats(min_value=0.1, max_value=0.5),
    sample_ratio=st.floats(min_value=0.5, max_value=2.0),
    power=st.floats(min_value=0.7, max_value=0.95),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64])
)
def test_two_one_sided_tests_two_sample_t_sample_size(
    effect_size, equivalence_margin, sample_ratio, power, alpha, dtype
):
    metric = TwoOneSidedTestsTwoSampleTSampleSize(power=power, alpha=alpha)
    
    effect_size_tensor = torch.tensor(effect_size, dtype=dtype)
    margin_tensor = torch.tensor(equivalence_margin, dtype=dtype)
    ratio_tensor = torch.tensor(sample_ratio, dtype=dtype)
    
    metric.update(effect_size_tensor, margin_tensor, ratio_tensor)
    result = metric.compute()
    
    assert isinstance(result, torch.Tensor)
    assert result.item() > 0
    
    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()