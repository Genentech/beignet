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
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    ratio=hypothesis.strategies.floats(min_value=0.5, max_value=2.0),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_independent_t_test_sample_size(batch_size, effect_size, power, alpha, ratio, dtype):
    """Test IndependentTTestSampleSize TorchMetrics class."""
    # IndependentTTestSampleSize constructor accepts power parameter
    metric = beignet.metrics.statistics.IndependentTTestSampleSize(power=power, alpha=alpha)
    assert isinstance(metric, Metric)

    # IndependentTTestSampleSize.update expects (effect_size, ratio) optionally
    effect_size_tensor = torch.full((batch_size,), effect_size, dtype=dtype)
    ratio_tensor = torch.full((batch_size,), ratio, dtype=dtype)

    metric.update(effect_size_tensor, ratio_tensor)
    output = metric.compute()

    assert isinstance(output, Tensor)
    # The metric returns the shape of the most recent update call
    assert output.dtype == dtype
    assert torch.all(output >= 2.0)  # Sample size should be at least 2
    # Verify output has reasonable dimensions (scalar or 1D tensor)
    assert output.ndim <= 1

    # Small effect should require larger sample size
    small_effect = torch.full((batch_size,), 0.1, dtype=dtype)
    metric_small = beignet.metrics.statistics.IndependentTTestSampleSize(power=power, alpha=alpha)
    metric_small.update(small_effect, ratio_tensor)
    size_small = metric_small.compute()

    large_effect = torch.full((batch_size,), 1.5, dtype=dtype)
    metric_large = beignet.metrics.statistics.IndependentTTestSampleSize(power=power, alpha=alpha)
    metric_large.update(large_effect, ratio_tensor)
    size_large = metric_large.compute()

    assert torch.all(size_small > size_large)

    metric.reset()
    try:
        metric.compute()
        raise AssertionError("Expected RuntimeError after reset")
    except RuntimeError:
        pass

    repr_str = repr(metric)
    assert "IndependentTTestSampleSize" in repr_str
