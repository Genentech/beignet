
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from beignet.metrics.statistics import PoissonRegressionSampleSize


@given(
    rate_ratio=st.floats(min_value=1.1, max_value=3.0),
    baseline_rate=st.floats(min_value=0.01, max_value=1.0),
    power=st.floats(min_value=0.7, max_value=0.95),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
def test_poisson_regression_sample_size(
    rate_ratio,
    baseline_rate,
    power,
    alpha,
    dtype,
):
    metric = PoissonRegressionSampleSize(power=power, alpha=alpha)

    rate_ratio_tensor = torch.tensor(rate_ratio, dtype=dtype)
    baseline_rate_tensor = torch.tensor(baseline_rate, dtype=dtype)

    metric.update(rate_ratio_tensor, baseline_rate_tensor)
    result = metric.compute()

    assert isinstance(result, torch.Tensor)
    assert result.item() > 0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
