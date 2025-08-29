
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from beignet.metrics.statistics import PoissonRegressionPower


@given(
    rate_ratio=st.floats(min_value=1.1, max_value=3.0),
    sample_size=st.integers(min_value=20, max_value=500),
    baseline_rate=st.floats(min_value=0.01, max_value=1.0),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
def test_poisson_regression_power(
    rate_ratio,
    sample_size,
    baseline_rate,
    alpha,
    dtype,
):
    metric = PoissonRegressionPower(alpha=alpha)

    rate_ratio_tensor = torch.tensor(rate_ratio, dtype=dtype)
    sample_size_tensor = torch.tensor(sample_size, dtype=dtype)
    baseline_rate_tensor = torch.tensor(baseline_rate, dtype=dtype)

    metric.update(rate_ratio_tensor, sample_size_tensor, baseline_rate_tensor)
    result = metric.compute()

    assert isinstance(result, torch.Tensor)
    assert 0.0 <= result.item() <= 1.0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
