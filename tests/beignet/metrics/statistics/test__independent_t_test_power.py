import hypothesis
import hypothesis.strategies as st
import torch
from torch import Tensor
from torchmetrics import Metric

import beignet.metrics.statistics


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=10),
    effect_size=st.floats(min_value=0.1, max_value=2.0),
    sample_size=st.integers(min_value=10, max_value=50),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_independent_t_test_power(batch_size, effect_size, sample_size, alpha, dtype):
    """Test IndependentTTestPower TorchMetrics class."""
    metric = beignet.metrics.statistics.IndependentTTestPower(alpha=alpha)
    assert isinstance(metric, Metric)

    effect_size_tensor = torch.full((batch_size,), effect_size, dtype=dtype)
    sample_size_tensor = torch.full((batch_size,), sample_size, dtype=torch.int64)

    metric.update(effect_size_tensor, sample_size_tensor)
    result = metric.compute()

    assert isinstance(result, Tensor)
    assert result.shape == (batch_size,)
    assert result.dtype == dtype
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)

    # Test edge cases
    small_effect = torch.full((batch_size,), 0.01, dtype=dtype)
    metric_small = beignet.metrics.statistics.IndependentTTestPower(alpha=alpha)
    metric_small.update(small_effect, sample_size_tensor)
    power_small = metric_small.compute()
    assert torch.all(power_small < 0.5)

    metric.reset()
    try:
        metric.compute()
        raise AssertionError("Expected RuntimeError after reset")
    except RuntimeError:
        pass

    repr_str = repr(metric)
    assert "IndependentTTestPower" in repr_str
