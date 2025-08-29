import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import CohensKappaSampleSize


@hypothesis.given(
    kappa=hypothesis.strategies.floats(min_value=0.1, max_value=0.9),
    categories=hypothesis.strategies.integers(min_value=2, max_value=5),
    power=hypothesis.strategies.floats(min_value=0.7, max_value=0.95),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_cohens_kappa_sample_size(
    kappa,
    categories,
    power,
    alpha,
    dtype,
):
    metric = CohensKappaSampleSize(power=power, alpha=alpha)

    kappa_tensor = torch.tensor(kappa, dtype=dtype)
    categories_tensor = torch.tensor(categories, dtype=dtype)

    metric.update(kappa_tensor, categories_tensor)
    result = metric.compute()

    assert isinstance(result, Tensor)
    assert result.item() > 0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
