import hypothesis
import hypothesis.strategies as st
import torch
from torch import Tensor
from torchmetrics import Metric

import beignet.metrics.statistics


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=10),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_phi_coefficient(batch_size, dtype):
    """Test PhiCoefficient TorchMetrics class."""
    metric = beignet.metrics.statistics.PhiCoefficient()
    assert isinstance(metric, Metric)

    # Create 2x2 contingency tables
    contingency_table = torch.randint(1, 50, (batch_size, 2, 2), dtype=torch.int64).to(
        dtype,
    )

    metric.update(contingency_table)
    result = metric.compute()

    assert isinstance(result, Tensor)
    assert result.shape == (batch_size,)
    assert result.dtype == dtype
    assert torch.all(result >= -1.0)
    assert torch.all(result <= 1.0)

    metric.reset()
    try:
        metric.compute()
        raise AssertionError("Expected RuntimeError after reset")
    except RuntimeError:
        pass

    repr_str = repr(metric)
    assert "PhiCoefficient" in repr_str
