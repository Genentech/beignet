import hypothesis
import hypothesis.strategies
import torch
from torch import Tensor
from torchmetrics import Metric

import beignet.metrics.statistics


@hypothesis.given(
    input_value=hypothesis.strategies.floats(min_value=0.1, max_value=2.0),
    df1=hypothesis.strategies.integers(min_value=1, max_value=10),
    power=hypothesis.strategies.floats(min_value=0.7, max_value=0.95),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_f_test_sample_size(input_value, df1, power, alpha, dtype):
    metric = beignet.metrics.statistics.FTestSampleSize(power=power, alpha=alpha)
    assert isinstance(metric, Metric)

    input_tensor = torch.tensor(input_value, dtype=dtype)
    df1_tensor = torch.tensor(df1, dtype=dtype)

    metric.update(input_tensor, df1_tensor)
    output = metric.compute()

    assert isinstance(output, Tensor)
    assert output.item() > 0
