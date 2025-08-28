import hypothesis
import hypothesis.strategies as st
import torch
from torchmetrics import Metric

import beignet.metrics.statistics


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=10),
    effect_size=st.floats(min_value=0.1, max_value=0.8),
    sample_size=st.integers(min_value=10, max_value=100),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_proportion_power(batch_size, effect_size, sample_size, alpha, dtype):
    """Test ProportionPower TorchMetrics class."""
    metric = beignet.metrics.statistics.ProportionPower(alpha=alpha)
    assert isinstance(metric, Metric)

    effect_size_tensor = torch.full((batch_size,), effect_size, dtype=dtype)
    sample_size_tensor = torch.full((batch_size,), sample_size, dtype=torch.int64)

    metric.update(effect_size_tensor, sample_size_tensor)
    result = metric.compute()

    assert isinstance(result, torch.Tensor)
    assert result.shape == (batch_size,)
    assert result.dtype == dtype
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)

    repr_str = repr(metric)
    assert "ProportionPower" in repr_str
