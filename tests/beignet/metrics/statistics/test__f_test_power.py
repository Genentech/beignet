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
    df1=st.integers(min_value=1, max_value=10),
    df2=st.integers(min_value=10, max_value=50),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_f_test_power(batch_size, effect_size, sample_size, df1, df2, alpha, dtype):
    """Test FTestPower TorchMetrics class."""
    metric = beignet.metrics.statistics.FTestPower(alpha=alpha)
    assert isinstance(metric, Metric)

    effect_size_tensor = torch.full((batch_size,), effect_size, dtype=dtype)
    sample_size_tensor = torch.full((batch_size,), sample_size, dtype=torch.int64)
    df1_tensor = torch.full((batch_size,), df1, dtype=torch.int64)
    df2_tensor = torch.full((batch_size,), df2, dtype=torch.int64)

    metric.update(effect_size_tensor, sample_size_tensor, df1_tensor, df2_tensor)
    result = metric.compute()

    assert isinstance(result, Tensor)
    assert result.shape == (batch_size,)
    assert result.dtype == dtype
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)

    metric.reset()
    try:
        metric.compute()
        raise AssertionError("Expected RuntimeError after reset")
    except RuntimeError:
        pass

    repr_str = repr(metric)
    assert "FTestPower" in repr_str
