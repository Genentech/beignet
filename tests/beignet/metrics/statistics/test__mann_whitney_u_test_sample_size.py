import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import MannWhitneyUTestSampleSize


@hypothesis.given(
    effect_size=hypothesis.strategies.floats(min_value=0.1, max_value=1.0),
    sample_ratio=hypothesis.strategies.floats(min_value=0.5, max_value=2.0),
    power=hypothesis.strategies.floats(min_value=0.7, max_value=0.95),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_mann_whitney_u_test_sample_size(
    effect_size,
    sample_ratio,
    power,
    alpha,
    dtype,
):
    metric = MannWhitneyUTestSampleSize(power=power, alpha=alpha)

    effect_size_tensor = torch.tensor(effect_size, dtype=dtype)
    sample_ratio_tensor = torch.tensor(sample_ratio, dtype=dtype)

    metric.update(effect_size_tensor, sample_ratio_tensor)
    result = metric.compute()

    assert isinstance(result, Tensor)
    assert result.item() > 0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
