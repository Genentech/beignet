import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import LogisticRegressionSampleSize


@hypothesis.given(
    odds_ratio=hypothesis.strategies.floats(min_value=1.2, max_value=5.0),
    baseline_probability=hypothesis.strategies.floats(min_value=0.1, max_value=0.9),
    power=hypothesis.strategies.floats(min_value=0.7, max_value=0.95),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_logistic_regression_sample_size(
    odds_ratio,
    baseline_probability,
    power,
    alpha,
    dtype,
):
    metric = LogisticRegressionSampleSize(power=power, alpha=alpha)

    odds_ratio_tensor = torch.tensor(odds_ratio, dtype=dtype)
    baseline_prob_tensor = torch.tensor(baseline_probability, dtype=dtype)

    metric.update(odds_ratio_tensor, baseline_prob_tensor)
    result = metric.compute()

    assert isinstance(result, Tensor)
    assert result.item() > 0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
