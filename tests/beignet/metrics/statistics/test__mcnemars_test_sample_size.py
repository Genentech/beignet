import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import McnemarsTestSampleSize


@hypothesis.given(
    discordant_proportion=hypothesis.strategies.floats(min_value=0.05, max_value=0.5),
    power=hypothesis.strategies.floats(min_value=0.7, max_value=0.95),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_mcnemars_test_sample_size(
    discordant_proportion,
    power,
    alpha,
    dtype,
):
    metric = McnemarsTestSampleSize(power=power, alpha=alpha)

    discordant_prop_tensor = torch.tensor(discordant_proportion, dtype=dtype)

    metric.update(discordant_prop_tensor)
    output = metric.compute()

    assert isinstance(output, Tensor)
    assert output.item() > 0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
