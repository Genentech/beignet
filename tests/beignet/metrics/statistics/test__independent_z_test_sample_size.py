import hypothesis
import hypothesis.strategies
import torch
from torch import Tensor
from torchmetrics import Metric

import beignet.metrics.statistics


@hypothesis.given(
    effect_size=hypothesis.strategies.floats(min_value=0.1, max_value=2.0),
    power=hypothesis.strategies.floats(min_value=0.7, max_value=0.95),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_independent_z_test_sample_size(effect_size, power, alpha, dtype):
    metric = beignet.metrics.statistics.IndependentZTestSampleSize(
        power=power,
        alpha=alpha,
    )
    assert isinstance(metric, Metric)

    effect_size_tensor = torch.tensor(effect_size, dtype=dtype)

    metric.update(effect_size_tensor)
    output = metric.compute()

    assert isinstance(output, Tensor)
    assert output.item() >= 2.0
