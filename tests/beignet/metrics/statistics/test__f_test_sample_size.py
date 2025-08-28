import hypothesis
import hypothesis.strategies as st
import torch
from torchmetrics import Metric

import beignet.metrics.statistics


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=10),
    effect_size=st.floats(min_value=0.1, max_value=2.0),
    power=st.floats(min_value=0.1, max_value=0.95),
    df1=st.integers(min_value=1, max_value=10),
    df2=st.integers(min_value=10, max_value=50),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
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
    result = metric.compute()

    assert isinstance(result, torch.Tensor)
    assert result.shape == (batch_size,)
    assert result.dtype == dtype
    assert torch.all(result >= 1.0)

    metric.reset()
    try:
        metric.compute()
        raise AssertionError("Expected RuntimeError after reset")
    except RuntimeError:
        pass

    repr_str = repr(metric)
    assert "FTestSampleSize" in repr_str
