import hypothesis
import hypothesis.strategies as st
import torch

import beignet.metrics.functional.statistics
from beignet.metrics.statistics import EtaSquared


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=10),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_eta_squared_metric(batch_size, dtype):
    """Test EtaSquared metric wrapper."""
    # Create test data - sum of squares values
    ss_between = torch.abs(torch.randn(batch_size, dtype=dtype)) * 25.0
    ss_total = ss_between + torch.abs(torch.randn(batch_size, dtype=dtype)) * 75.0

    # Initialize metric
    metric = EtaSquared()

    # Update metric
    metric.update(ss_between, ss_total)

    # Compute result
    result_metric = metric.compute()

    # Compare with functional implementation
    result_functional = beignet.metrics.functional.statistics.eta_squared(
        ss_between,
        ss_total,
    )

    # Verify results are close (TorchMetrics may squeeze single-element tensors to scalars)
    if result_functional.shape == torch.Size([1]) and result_metric.shape == torch.Size(
        []
    ):
        assert torch.allclose(result_metric, result_functional.squeeze(), atol=1e-6)
    else:
        assert torch.allclose(result_metric, result_functional, atol=1e-6)
        assert result_metric.shape == result_functional.shape
    assert result_metric.dtype == result_functional.dtype

    # Verify eta squared is between 0 and 1
    assert torch.all(result_metric >= 0.0)
    assert torch.all(result_metric <= 1.0)

    # Test metric reset
    metric.reset()
    try:
        metric.compute()
        assert False, "Should raise RuntimeError after reset"
    except RuntimeError:
        pass  # Expected

    # Test metric with new data after reset
    metric.update(ss_between, ss_total)
    result_after_reset = metric.compute()
    assert torch.allclose(result_after_reset, result_functional, atol=1e-6)
