
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from beignet.metrics.statistics import AnalysisOfVarianceMinimumDetectableEffect


@given(
    sample_size=st.integers(min_value=30, max_value=300),
    groups=st.integers(min_value=2, max_value=6),
    power=st.floats(min_value=0.7, max_value=0.95),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
def test_analysis_of_variance_minimum_detectable_effect(
    sample_size,
    groups,
    power,
    alpha,
    dtype,
):
    metric = AnalysisOfVarianceMinimumDetectableEffect(power=power, alpha=alpha)

    sample_size_tensor = torch.tensor(sample_size, dtype=dtype)
    groups_tensor = torch.tensor(groups, dtype=dtype)

    metric.update(sample_size_tensor, groups_tensor)
    result = metric.compute()

    assert isinstance(result, torch.Tensor)
    assert result.item() > 0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
