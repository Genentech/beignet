import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import AnalysisOfCovariancePower


@hypothesis.given(
    effect_size=hypothesis.strategies.floats(min_value=0.1, max_value=1.0),
    sample_size=hypothesis.strategies.integers(min_value=30, max_value=300),
    groups=hypothesis.strategies.integers(min_value=2, max_value=6),
    covariate_r2=hypothesis.strategies.floats(min_value=0.1, max_value=0.8),
    num_covariates=hypothesis.strategies.integers(min_value=1, max_value=5),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_analysis_of_covariance_power(
    effect_size,
    sample_size,
    groups,
    covariate_r2,
    num_covariates,
    alpha,
    dtype,
):
    metric = AnalysisOfCovariancePower(alpha=alpha)

    metric.update(
        torch.tensor(effect_size, dtype=dtype),
        torch.tensor(sample_size, dtype=dtype),
        torch.tensor(groups, dtype=dtype),
        torch.tensor(covariate_r2, dtype=dtype),
        torch.tensor(num_covariates, dtype=dtype),
    )

    output = metric.compute()

    assert isinstance(output, Tensor)

    assert 0.0 <= output.item() <= 1.0

    metric.reset()

    with pytest.raises(RuntimeError):
        metric.compute()
