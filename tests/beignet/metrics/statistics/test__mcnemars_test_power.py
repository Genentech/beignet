import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import McnemarsTestPower


@hypothesis.given(
    discordant_proportion=hypothesis.strategies.floats(min_value=0.05, max_value=0.5),
    sample_size=hypothesis.strategies.integers(min_value=20, max_value=200),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_mcnemars_test_power(
    discordant_proportion,
    sample_size,
    alpha,
    dtype,
):
    metric = McnemarsTestPower(alpha=alpha)

    discordant_prop_tensor = torch.tensor(discordant_proportion, dtype=dtype)
    sample_size_tensor = torch.tensor(sample_size, dtype=dtype)

    metric.update(discordant_prop_tensor, sample_size_tensor)
    result = metric.compute()

    assert isinstance(result, Tensor)
    assert 0.0 <= result.item() <= 1.0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
