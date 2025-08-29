import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import KruskalWallisTestPower


@hypothesis.given(
    effect_size=hypothesis.strategies.floats(min_value=0.1, max_value=1.0),
    sample_size=hypothesis.strategies.integers(min_value=30, max_value=200),
    groups=hypothesis.strategies.integers(min_value=3, max_value=6),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_kruskal_wallis_test_power(
    effect_size,
    sample_size,
    groups,
    alpha,
    dtype,
):
    metric = KruskalWallisTestPower(alpha=alpha)

    effect_size_tensor = torch.tensor(effect_size, dtype=dtype)
    sample_size_tensor = torch.tensor(sample_size, dtype=dtype)
    groups_tensor = torch.tensor(groups, dtype=dtype)

    metric.update(effect_size_tensor, sample_size_tensor, groups_tensor)
    output = metric.compute()

    assert isinstance(output, Tensor)
    assert 0.0 <= output.item() <= 1.0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
