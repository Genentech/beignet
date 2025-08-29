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

    # Create chi-square values and sample sizes (PhiCoefficient expects these directly)
    chi_square = (
        torch.rand(batch_size, dtype=dtype) * 10.0
    )  # Random chi-square values 0-10
    sample_size = torch.randint(
        50,
        200,
        (batch_size,),
        dtype=dtype,
    )  # Random sample sizes 50-200

    metric.update(chi_square, sample_size)
    output = metric.compute()

    assert isinstance(output, Tensor)
    # The metric returns the shape of the most recent update call
    assert output.dtype == dtype
    assert torch.all(output >= 0.0)  # Phi coefficient is absolute value, so >= 0
    assert torch.all(output <= 1.0)
    # Verify output has reasonable dimensions (scalar or 1D tensor)
    assert output.ndim <= 1

    metric.reset()
    try:
        metric.compute()
        raise AssertionError("Expected RuntimeError after reset")
    except RuntimeError:
        pass

    repr_str = repr(metric)
    assert "PhiCoefficient" in repr_str
