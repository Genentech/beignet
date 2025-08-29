import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import IntraclassCorrelationPower


@hypothesis.given(
    icc=hypothesis.strategies.floats(min_value=0.1, max_value=0.9),
    sample_size=hypothesis.strategies.integers(min_value=20, max_value=200),
    groups=hypothesis.strategies.integers(min_value=3, max_value=10),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_intraclass_correlation_power(
    icc,
    sample_size,
    groups,
    alpha,
    dtype,
):
    metric = IntraclassCorrelationPower(alpha=alpha)

    icc_tensor = torch.tensor(icc, dtype=dtype)
    sample_size_tensor = torch.tensor(sample_size, dtype=dtype)
    groups_tensor = torch.tensor(groups, dtype=dtype)

    metric.update(icc_tensor, sample_size_tensor, groups_tensor)
    result = metric.compute()

    assert isinstance(result, Tensor)
    assert 0.0 <= result.item() <= 1.0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
