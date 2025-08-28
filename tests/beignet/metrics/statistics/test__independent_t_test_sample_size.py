import hypothesis
import hypothesis.strategies as st
import torch
from torchmetrics import Metric

import beignet.metrics.statistics


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=10),
    effect_size=st.floats(min_value=0.1, max_value=2.0),
    power=st.floats(min_value=0.1, max_value=0.95),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_independent_t_test_sample_size(batch_size, effect_size, power, alpha, dtype):
    """Test IndependentTTestSampleSize TorchMetrics class."""
    metric = beignet.metrics.statistics.IndependentTTestSampleSize(alpha=alpha)
    assert isinstance(metric, Metric)

    effect_size_tensor = torch.full((batch_size,), effect_size, dtype=dtype)
    power_tensor = torch.full((batch_size,), power, dtype=dtype)

    metric.update(effect_size_tensor, power_tensor)
    result = metric.compute()

    assert isinstance(result, torch.Tensor)
    assert result.shape == (batch_size,)
    assert result.dtype == dtype
    assert torch.all(result >= 2.0)

    # Small effect should require larger sample size
    small_effect = torch.full((batch_size,), 0.1, dtype=dtype)
    metric_small = beignet.metrics.statistics.IndependentTTestSampleSize(alpha=alpha)
    metric_small.update(small_effect, power_tensor)
    size_small = metric_small.compute()

    large_effect = torch.full((batch_size,), 1.5, dtype=dtype)
    metric_large = beignet.metrics.statistics.IndependentTTestSampleSize(alpha=alpha)
    metric_large.update(large_effect, power_tensor)
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
