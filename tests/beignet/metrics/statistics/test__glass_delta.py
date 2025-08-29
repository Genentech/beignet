import hypothesis
import hypothesis.strategies as st
import torch

import beignet.metrics.functional.statistics
from beignet.metrics.statistics import GlassDelta


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=10),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_glass_delta_metric(batch_size, dtype):
    """Test GlassDelta metric wrapper."""
    # Create test data with some effect
    group1 = torch.randn(batch_size, 20, dtype=dtype)  # Control group
    group2 = torch.randn(batch_size, 20, dtype=dtype) + 0.5  # Treatment group

    # Initialize metric
    metric = GlassDelta()

    # Update metric
    metric.update(group1, group2)

    # Compute result
    result_metric = metric.compute()

    # Compare with functional implementation
    result_functional = beignet.metrics.functional.statistics.glass_delta(
        group1.view(batch_size, -1),
        group2.view(batch_size, -1),
    )

    # Verify results are close (TorchMetrics may squeeze single-element tensors to scalars)
    if result_functional.shape == torch.Size([1]) and result_metric.shape == torch.Size(
        [],
    ):
        assert torch.allclose(result_metric, result_functional.squeeze(), atol=1e-6)
    else:
        assert torch.allclose(result_metric, result_functional, atol=1e-6)
        assert result_metric.shape == result_functional.shape
    assert result_metric.dtype == result_functional.dtype

    # Test metric reset
    metric.reset()
    try:
        metric.compute()
        assert False, "Should raise RuntimeError after reset"
    except RuntimeError:
        pass  # Expected

    # Test metric with new data after reset
    metric.update(group1, group2)
    result_after_reset = metric.compute()
    assert torch.allclose(result_after_reset, result_functional, atol=1e-6)
