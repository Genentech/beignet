import hypothesis
import hypothesis.strategies
import torch

import beignet.metrics.functional.statistics
from beignet.metrics.statistics import WelchTTestPower


@hypothesis.given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=10),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_welch_t_test_power_metric(batch_size, dtype):
    """Test WelchTTestPower metric wrapper."""
    # Create reasonable test parameters
    effect_size = (
        torch.abs(torch.randn(batch_size, dtype=dtype)) * 0.8 + 0.2
    )  # Between 0.2 and 1.0
    sample_size1 = torch.randint(10, 50, (batch_size,), dtype=dtype)
    sample_size2 = torch.randint(10, 50, (batch_size,), dtype=dtype)

    alpha = 0.05
    alternative = "two-sided"

    # Initialize metric
    metric = WelchTTestPower(alpha=alpha, alternative=alternative)

    # Update metric
    metric.update(effect_size, sample_size1, sample_size2)

    # Compute output
    result_metric = metric.compute()

    # Compare with functional implementation
    result_functional = beignet.metrics.functional.statistics.welch_t_test_power(
        effect_size,
        sample_size1,
        sample_size2,
        alpha=alpha,
        alternative=alternative,
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

    # Verify power is between 0 and 1
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
    metric.update(effect_size, sample_size1, sample_size2)
    result_after_reset = metric.compute()
    assert torch.allclose(result_after_reset, result_functional, atol=1e-6)

    # Test parameter validation
    try:
        WelchTTestPower(alpha=1.5)
        assert False, "Should raise ValueError for invalid alpha"
    except ValueError:
        pass  # Expected

    try:
        WelchTTestPower(alternative="invalid")
        assert False, "Should raise ValueError for invalid alternative"
    except ValueError:
        pass  # Expected
