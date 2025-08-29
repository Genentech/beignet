import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import LogisticRegressionPower


@hypothesis.given(
    odds_ratio=hypothesis.strategies.floats(min_value=1.2, max_value=5.0),
    sample_size=hypothesis.strategies.integers(min_value=50, max_value=500),
    baseline_probability=hypothesis.strategies.floats(min_value=0.1, max_value=0.9),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
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

    assert isinstance(result, Tensor)
    assert 0.0 <= result.item() <= 1.0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
