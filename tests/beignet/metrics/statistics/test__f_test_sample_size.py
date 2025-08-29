import hypothesis
import hypothesis.strategies
import torch
from torch import Tensor
from torchmetrics import Metric

import beignet.metrics.statistics


@hypothesis.given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=10),
    effect_size=hypothesis.strategies.floats(min_value=0.1, max_value=2.0),
    power=hypothesis.strategies.floats(min_value=0.1, max_value=0.95),
    df1=hypothesis.strategies.integers(min_value=1, max_value=10),
    df2=hypothesis.strategies.integers(min_value=10, max_value=50),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_f_test_sample_size(batch_size, effect_size, power, df1, df2, alpha, dtype):
    """Test FTestSampleSize TorchMetrics class."""
    metric = beignet.metrics.statistics.FTestSampleSize(alpha=alpha)
    assert isinstance(metric, Metric)

    effect_size_tensor = torch.full((batch_size,), effect_size, dtype=dtype)
    power_tensor = torch.full((batch_size,), power, dtype=dtype)
    df1_tensor = torch.full((batch_size,), df1, dtype=torch.int64)
    df2_tensor = torch.full((batch_size,), df2, dtype=torch.int64)

    metric.update(effect_size_tensor, power_tensor, df1_tensor, df2_tensor)
    output = metric.compute()

    assert isinstance(output, Tensor)
    assert output.shape == (batch_size,)
    assert output.dtype == dtype
    assert torch.all(output >= 1.0)

    metric.reset()
    try:
        metric.compute()
        raise AssertionError("Expected RuntimeError after reset")
    except RuntimeError:
        pass

    repr_str = repr(metric)
    assert "FTestSampleSize" in repr_str
