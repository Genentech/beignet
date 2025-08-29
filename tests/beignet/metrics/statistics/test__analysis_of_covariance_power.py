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

    effect_size_tensor = torch.tensor(effect_size, dtype=dtype)
    sample_size_tensor = torch.tensor(sample_size, dtype=dtype)
    groups_tensor = torch.tensor(groups, dtype=dtype)
    covariate_r2_tensor = torch.tensor(covariate_r2, dtype=dtype)
    num_covariates_tensor = torch.tensor(num_covariates, dtype=dtype)

    metric.update(
        effect_size_tensor,
        sample_size_tensor,
        groups_tensor,
        covariate_r2_tensor,
        num_covariates_tensor,
    )
    result = metric.compute()

    assert isinstance(result, Tensor)
    assert 0.0 <= result.item() <= 1.0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
