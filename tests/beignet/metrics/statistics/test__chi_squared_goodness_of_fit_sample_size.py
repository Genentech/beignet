import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import ChiSquareGoodnessOfFitSampleSize


@hypothesis.given(
    effect_size=hypothesis.strategies.floats(min_value=0.1, max_value=2.0),
    degrees_of_freedom=hypothesis.strategies.integers(min_value=1, max_value=10),
    power=hypothesis.strategies.floats(min_value=0.7, max_value=0.95),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_chi_squared_goodness_of_fit_sample_size(
    effect_size,
    degrees_of_freedom,
    power,
    alpha,
    dtype,
):
    metric = ChiSquareGoodnessOfFitSampleSize(power=power, alpha=alpha)

    effect_size_tensor = torch.tensor(effect_size, dtype=dtype)
    degrees_of_freedom_tensor = torch.tensor(degrees_of_freedom, dtype=dtype)

    metric.update(effect_size_tensor, degrees_of_freedom_tensor)
    output = metric.compute()

    assert isinstance(output, Tensor)
    assert output.item() > 0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()