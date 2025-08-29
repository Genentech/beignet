import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import AnalysisOfCovarianceSampleSize


@hypothesis.given(
    effect_size=hypothesis.strategies.floats(min_value=0.1, max_value=1.0),
    groups=hypothesis.strategies.integers(min_value=2, max_value=6),
    covariate_r2=hypothesis.strategies.floats(min_value=0.1, max_value=0.8),
    num_covariates=hypothesis.strategies.integers(min_value=1, max_value=5),
    power=hypothesis.strategies.floats(min_value=0.7, max_value=0.95),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_analysis_of_covariance_sample_size(
    effect_size,
    groups,
    covariate_r2,
    num_covariates,
    power,
    alpha,
    dtype,
):
    metric = AnalysisOfCovarianceSampleSize(power=power, alpha=alpha)

    effect_size_tensor = torch.tensor(effect_size, dtype=dtype)
    groups_tensor = torch.tensor(groups, dtype=dtype)
    covariate_r2_tensor = torch.tensor(covariate_r2, dtype=dtype)
    num_covariates_tensor = torch.tensor(num_covariates, dtype=dtype)

    metric.update(
        effect_size_tensor,
        groups_tensor,
        covariate_r2_tensor,
        num_covariates_tensor,
    )

    output = metric.compute()

    assert isinstance(output, Tensor)

    assert output.item() > 0

    metric.reset()

    with pytest.raises(RuntimeError):
        metric.compute()
