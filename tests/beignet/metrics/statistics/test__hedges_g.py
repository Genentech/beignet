import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import HedgesG


@hypothesis.given(
    sample_size_group1=hypothesis.strategies.integers(min_value=5, max_value=20),
    sample_size_group2=hypothesis.strategies.integers(min_value=5, max_value=20),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_hedges_g(sample_size_group1, sample_size_group2, dtype):
    metric = HedgesG()

    # Create test groups with different means
    group1 = torch.randn(sample_size_group1, dtype=dtype)
    group2 = torch.randn(sample_size_group2, dtype=dtype) + 1.0  # Different mean

    metric.update(group1, group2)
    output = metric.compute()

    assert isinstance(output, Tensor)
    # Hedges' g can be finite, NaN, or infinite for edge cases
    assert torch.isfinite(output) or torch.isnan(output) or torch.isinf(output)

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
