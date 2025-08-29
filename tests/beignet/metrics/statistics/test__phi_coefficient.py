import hypothesis
import hypothesis.strategies
import torch
from torch import Tensor
from torchmetrics import Metric

import beignet.metrics.statistics


@hypothesis.given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=10),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
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
    output = metric.compute()

    assert isinstance(output, Tensor)
    assert output.shape == (batch_size,)
    assert output.dtype == dtype
    assert torch.all(output >= -1.0)
    assert torch.all(output <= 1.0)

    metric.reset()
    try:
        metric.compute()
        raise AssertionError("Expected RuntimeError after reset")
    except RuntimeError:
        pass

    repr_str = repr(metric)
    assert "PhiCoefficient" in repr_str
