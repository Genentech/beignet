
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from beignet.metrics.statistics import LogisticRegressionPower


@given(
    odds_ratio=st.floats(min_value=1.2, max_value=5.0),
    sample_size=st.integers(min_value=50, max_value=500),
    baseline_probability=st.floats(min_value=0.1, max_value=0.9),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
def test_logistic_regression_power(
    odds_ratio,
    sample_size,
    baseline_probability,
    alpha,
    dtype,
):
    metric = LogisticRegressionPower(alpha=alpha)

    odds_ratio_tensor = torch.tensor(odds_ratio, dtype=dtype)
    sample_size_tensor = torch.tensor(sample_size, dtype=dtype)
    baseline_prob_tensor = torch.tensor(baseline_probability, dtype=dtype)

    metric.update(odds_ratio_tensor, sample_size_tensor, baseline_prob_tensor)
    result = metric.compute()

    assert isinstance(result, torch.Tensor)
    assert 0.0 <= result.item() <= 1.0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
