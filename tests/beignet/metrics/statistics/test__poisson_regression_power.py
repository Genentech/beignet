import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import PoissonRegressionPower


@hypothesis.given(
    rate_ratio=hypothesis.strategies.floats(min_value=1.1, max_value=3.0),
    sample_size=hypothesis.strategies.integers(min_value=20, max_value=500),
    baseline_rate=hypothesis.strategies.floats(min_value=0.01, max_value=1.0),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
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
    output = metric.compute()

    assert isinstance(output, Tensor)
    assert 0.0 <= output.item() <= 1.0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
