import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import TwoOneSidedTestsOneSampleTSampleSize


@hypothesis.given(
    effect_size=hypothesis.strategies.floats(min_value=0.1, max_value=1.0),
    equivalence_margin=hypothesis.strategies.floats(min_value=0.1, max_value=0.5),
    power=hypothesis.strategies.floats(min_value=0.7, max_value=0.95),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_two_one_sided_tests_one_sample_t_sample_size(
    effect_size,
    equivalence_margin,
    power,
    alpha,
    dtype,
):
    metric = TwoOneSidedTestsOneSampleTSampleSize(power=power, alpha=alpha)

    effect_size_tensor = torch.tensor(effect_size, dtype=dtype)

    margin_tensor = torch.tensor(equivalence_margin, dtype=dtype)

    metric.update(effect_size_tensor, margin_tensor)

    output = metric.compute()

    assert isinstance(output, Tensor)

    assert output.item() > 0

    metric.reset()

    with pytest.raises(RuntimeError):
        metric.compute()
