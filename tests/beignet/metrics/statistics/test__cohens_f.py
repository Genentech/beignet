import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import CohensF


@hypothesis.given(
    num_groups=hypothesis.strategies.integers(min_value=3, max_value=6),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_cohens_f(num_groups, dtype):
    metric = CohensF()

    # Create test group means with some variation
    group_means = torch.randn(num_groups, dtype=dtype)
    pooled_std = torch.tensor(1.0, dtype=dtype)

    metric.update(group_means, pooled_std)
    output = metric.compute()

    assert isinstance(output, Tensor)
    assert output.item() >= 0.0  # Cohen's f should be non-negative

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
