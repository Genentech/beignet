import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import AnalysisOfVarianceMinimumDetectableEffect


@hypothesis.given(
    sample_size=hypothesis.strategies.integers(min_value=30, max_value=300),
    groups=hypothesis.strategies.integers(min_value=2, max_value=6),
    power=hypothesis.strategies.floats(min_value=0.7, max_value=0.95),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_analysis_of_variance_minimum_detectable_effect(
    sample_size,
    groups,
    power,
    alpha,
    dtype,
):
    metric = AnalysisOfVarianceMinimumDetectableEffect(power=power, alpha=alpha)

    sample_size_tensor = torch.tensor(sample_size, dtype=dtype)

    groups_tensor = torch.tensor(groups, dtype=dtype)

    metric.update(sample_size_tensor, groups_tensor)

    output = metric.compute()

    assert isinstance(output, Tensor)

    assert output.item() > 0

    metric.reset()

    with pytest.raises(RuntimeError):
        metric.compute()
