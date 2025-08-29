import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import PoissonRegressionSampleSize


@hypothesis.given(
    rate_ratio=hypothesis.strategies.floats(min_value=1.1, max_value=3.0),
    baseline_rate=hypothesis.strategies.floats(min_value=0.01, max_value=1.0),
    power=hypothesis.strategies.floats(min_value=0.7, max_value=0.95),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
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

    assert isinstance(result, Tensor)
    assert result.item() > 0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
