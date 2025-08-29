import hypothesis
import hypothesis.strategies as st
import torch
from torch import Tensor
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
def test_t_test_sample_size(batch_size, effect_size, power, alpha, dtype):
    """Test TTestSampleSize TorchMetrics class."""
    metric = beignet.metrics.statistics.TTestSampleSize(alpha=alpha)
    assert isinstance(metric, Metric)

    effect_size_tensor = torch.full((batch_size,), effect_size, dtype=dtype)
    power_tensor = torch.full((batch_size,), power, dtype=dtype)

    metric.update(effect_size_tensor, power_tensor)
    result = metric.compute()

    assert isinstance(result, Tensor)
    assert result.shape == (batch_size,)
    assert result.dtype == dtype
    assert torch.all(result >= 3.0)

    repr_str = repr(metric)
    assert "TTestSampleSize" in repr_str
