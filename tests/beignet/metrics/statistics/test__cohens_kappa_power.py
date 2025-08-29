import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import CohensKappaPower


@hypothesis.given(
    kappa=hypothesis.strategies.floats(min_value=0.1, max_value=0.9),
    sample_size=hypothesis.strategies.integers(min_value=20, max_value=200),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_cohens_kappa_power(
    kappa,
    sample_size,
    alpha,
    dtype,
):
    metric = CohensKappaPower(alpha=alpha)

    kappa_tensor = torch.tensor(kappa, dtype=dtype)
    sample_size_tensor = torch.tensor(sample_size, dtype=dtype)

    metric.update(kappa_tensor, sample_size_tensor)
    output = metric.compute()

    assert isinstance(output, Tensor)
    assert 0.0 <= output.item() <= 1.0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
