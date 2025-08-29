import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import AnalysisOfVarianceSampleSize


@hypothesis.given(
    effect_size=hypothesis.strategies.floats(min_value=0.1, max_value=2.0),
    groups=hypothesis.strategies.integers(min_value=3, max_value=8),
    power=hypothesis.strategies.floats(min_value=0.7, max_value=0.95),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_analysis_of_variance_sample_size(
    effect_size,
    groups,
    power,
    alpha,
    dtype,
):
    metric = AnalysisOfVarianceSampleSize(power=power, alpha=alpha)

    effect_size_tensor = torch.tensor(effect_size, dtype=dtype)
    groups_tensor = torch.tensor(groups, dtype=dtype)

    metric.update(effect_size_tensor, groups_tensor)
    output = metric.compute()

    assert isinstance(output, Tensor)
    assert output.item() > 0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
