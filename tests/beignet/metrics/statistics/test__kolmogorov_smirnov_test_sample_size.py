
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from beignet.metrics.statistics import KolmogorovSmirnovTestSampleSize


@given(
    effect_size=st.floats(min_value=0.1, max_value=0.8),
    sample_ratio=st.floats(min_value=0.5, max_value=2.0),
    power=st.floats(min_value=0.7, max_value=0.95),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
def test_kolmogorov_smirnov_test_sample_size(
    effect_size,
    sample_ratio,
    power,
    alpha,
    dtype,
):
    metric = KolmogorovSmirnovTestSampleSize(power=power, alpha=alpha)

    effect_size_tensor = torch.tensor(effect_size, dtype=dtype)
    sample_ratio_tensor = torch.tensor(sample_ratio, dtype=dtype)

    metric.update(effect_size_tensor, sample_ratio_tensor)
    result = metric.compute()

    assert isinstance(result, torch.Tensor)
    assert result.item() > 0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
